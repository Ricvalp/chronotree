"""Matched two-tower local scorers with immutable candidate-key caching."""

from __future__ import annotations
import math
from copy import deepcopy
import torch
from torch import nn
from stmem.models.event_encoder import command_conditioning
from stmem.models.metadata_adapter import metadata, METADATA_DIM


class LocalRouter(nn.Module):
    requires_children = False
    requires_current = False

    def __init__(self, width=256, key_width=128):
        super().__init__()
        self.query_dim = 2 * width + 27  # current visual, text, proprio, command+mask, clocks, B
        self.query = nn.Sequential(
            nn.Linear(self.query_dim, key_width),
            nn.SiLU(),
            nn.Linear(key_width, key_width),
            nn.LayerNorm(key_width),
        )
        self.key = nn.Sequential(
            nn.Linear(2 * width + METADATA_DIM, key_width),
            nn.SiLU(),
            nn.Linear(key_width, key_width),
            nn.LayerNorm(key_width),
        )
        self.bias = nn.Linear(METADATA_DIM, 1)
        self.scale = key_width**-0.5
        self.width = width

    def encode_query(self, query_input, *, current=None, current_valid=None):
        return self.query(query_input.float())

    def encode_candidates(self, candidate_input, *, children=None):
        return (self.key(candidate_input.float()),
                self.bias(candidate_input[..., -METADATA_DIM:]).squeeze(-1))

    def score_encoded(self, query, keys, bias, *, score_head=None):
        return (query * keys).sum(-1) * self.scale + bias

    def candidate_input(self, records):
        if not records:
            device = next(self.parameters()).device
            return torch.empty(0, 2 * self.width + METADATA_DIM, device=device)
        device = next(self.parameters()).device
        # No age in immutable keys: absolute source-local time enters both towers.
        # Stack at the archive device before transfer: one bulk transfer for
        # each field rather than one small accelerator copy per record.
        archive_device = records[0].content_sum.device
        m = torch.stack([metadata(r, {}, device=archive_device) for r in records]).to(device)
        m[:, 11:13] = 0  # read-time ages are intentionally excluded
        content = torch.stack([r.content.float() for r in records]).to(device)
        maxima = torch.stack([r.max_content.float() for r in records]).to(device)
        return torch.cat([content, maxima, m], -1)

    def forward(self, query_input, candidate_input, *, children=None, current=None, current_valid=None):
        q = self.encode_query(query_input, current=current, current_valid=current_valid)
        k, bias = self.encode_candidates(candidate_input, children=children)
        return self.score_encoded(q, k, bias)


class CurrentQueryBlock(nn.Module):
    """A few task-conditioned queries read only the current observation."""

    def __init__(self, width, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))

    def forward(self, queries, context, valid):
        normalized = self.context_norm(context)
        update, _ = self.attention(self.query_norm(queries), normalized, normalized,
                                   key_padding_mask=~valid, need_weights=False)
        queries = queries + update
        return queries + self.ff(self.ff_norm(queries))


class ArchitectureRouter(LocalRouter):
    """Efficient architecture ablations; none reads all historical tokens.

    Every non-dot variant shares the same nonlinear query/candidate interaction.
    Children are encoded separately before symmetric pooling, preserving their
    feature/geometry associations. Current attention retains several learned
    queries and reads only current visual tokens plus the task/state/clock token.
    """

    def __init__(self, width=256, key_width=128, *, architecture="mlp", heads=4,
                 query_tokens=8, attention_layers=2):
        super().__init__(width, key_width)
        if architecture not in ("mlp", "children", "current_attention", "children_current_attention"):
            raise ValueError(f"Unknown router architecture: {architecture}")
        self.architecture = architecture
        self.requires_children = architecture in ("children", "children_current_attention")
        self.requires_current = architecture in ("current_attention", "children_current_attention")
        self.score_head = nn.Sequential(nn.Linear(4 * key_width, key_width), nn.SiLU(), nn.Linear(key_width, 1))
        if self.requires_children:
            self.child_encoder = nn.Sequential(
                nn.Linear(2 * width + METADATA_DIM, key_width), nn.SiLU(),
                nn.Linear(key_width, key_width), nn.LayerNorm(key_width),
            )
            self.child_fusion = nn.Sequential(nn.Linear(3 * key_width, key_width), nn.SiLU(),
                                               nn.Linear(key_width, key_width), nn.LayerNorm(key_width))
        if self.requires_current:
            if (not isinstance(heads, int) or heads < 1 or key_width % heads
                    or not isinstance(query_tokens, int) or query_tokens < 1
                    or not isinstance(attention_layers, int) or attention_layers < 1):
                raise ValueError("Current attention needs positive layers/query count and heads dividing width")
            # The task/state/clock pathway deliberately excludes the old visual
            # mean. Individual current contents and metadata enter below.
            self.query = nn.Sequential(nn.Linear(width + 27, key_width), nn.SiLU(),
                                       nn.Linear(key_width, key_width), nn.LayerNorm(key_width))
            self.current_projection = nn.Sequential(nn.Linear(width + METADATA_DIM, key_width),
                                                    nn.SiLU(), nn.LayerNorm(key_width))
            self.query_seeds = nn.Parameter(torch.randn(query_tokens, key_width) * 0.02)
            self.current_blocks = nn.ModuleList(
                CurrentQueryBlock(key_width, heads) for _ in range(attention_layers)
            )
            self.current_output_norm = nn.LayerNorm(key_width)

    def child_input(self, records):
        if any(len(record.children) != 2 for record in records):
            raise ValueError("Child-aware candidates must be splittable binary parents")
        return self.candidate_input([child for record in records for child in record.children]).reshape(
            len(records), 2, 2 * self.width + METADATA_DIM
        )

    def encode_candidates(self, candidate_input, *, children=None):
        keys, bias = super().encode_candidates(candidate_input)
        if self.requires_children:
            expected = (*candidate_input.shape[:-1], 2, candidate_input.shape[-1])
            if children is None or tuple(children.shape) != expected:
                raise ValueError(f"Child-aware router requires child features shaped {expected}")
            child_keys = self.child_encoder(children.float())
            keys = self.child_fusion(torch.cat((keys, child_keys.mean(-2), child_keys.amax(-2)), -1))
        return keys, bias

    def encode_query(self, query_input, *, current=None, current_valid=None):
        if not self.requires_current:
            return super().encode_query(query_input)
        if current is None:
            raise ValueError("Current-attention router requires individual current tokens and metadata")
        single = query_input.ndim == 1
        query_input = query_input.unsqueeze(0) if single else query_input
        current = current.unsqueeze(0) if current.ndim == 2 else current
        if (query_input.ndim != 2 or current.ndim != 3
                or current.shape[0] != query_input.shape[0]
                or current.shape[-1] != self.width + METADATA_DIM):
            raise ValueError("Current tokens must have shape [batch, tokens, width + metadata]")
        if current_valid is None:
            current_valid = torch.ones(current.shape[:2], dtype=torch.bool, device=current.device)
        elif current_valid.ndim == 1:
            current_valid = current_valid.unsqueeze(0)
        if current_valid.dtype != torch.bool or tuple(current_valid.shape) != tuple(current.shape[:2]):
            raise ValueError("Current validity must be a boolean [batch, tokens] mask")
        current = torch.where(current_valid[..., None], current.float(), 0.0)
        global_query = self.query(query_input[..., self.width:].float())
        # The task/state/clock token is always valid, including an empty scene;
        # attention never receives an all-masked context.
        context = torch.cat((global_query[:, None], self.current_projection(current)), 1)
        valid = torch.cat((torch.ones_like(current_valid[:, :1]) if current_valid.shape[1]
                           else torch.ones((len(current), 1), dtype=torch.bool, device=current.device),
                           current_valid), 1)
        queries = global_query[:, None] + self.query_seeds[None]
        for block in self.current_blocks:
            queries = block(queries, context, valid)
        queries = self.current_output_norm(queries)
        return queries[0] if single else queries

    def score_encoded(self, query, keys, bias, *, score_head=None):
        if self.requires_current:
            if query.ndim == 2:
                # One cached decision query, any number of encountered nodes.
                weights = torch.softmax(keys @ query.transpose(-1, -2) * self.scale, -1)
                query = weights @ query
            else:
                weights = torch.softmax((query * keys.unsqueeze(-2)).sum(-1) * self.scale, -1)
                query = (weights[..., None] * query).sum(-2)
        query, keys = torch.broadcast_tensors(query, keys)
        head = self.score_head if score_head is None else score_head
        joint = torch.cat((query, keys, query * keys, (query - keys).abs()), -1)
        return head(joint).squeeze(-1) + bias


def build_router(cfg):
    model = cfg["model"]
    architecture = model.get("router_architecture", "dot")
    if architecture == "dot":
        return LocalRouter(model["width"], model["router_width"])
    if architecture in ("children", "children_current_attention") and cfg.get("reader", "tree") != "tree":
        raise ValueError("Child-aware routing is a TREE architecture diagnostic")
    return ArchitectureRouter(
        model["width"], model["router_width"], architecture=architecture,
        heads=model.get("router_attention_heads", 4), query_tokens=model.get("router_query_tokens", 8),
        attention_layers=model.get("router_attention_layers", 2),
    )


def current_token_features(records, clocks, *, width, device):
    """Allowed current contents and metadata, exactly as represented in caches."""
    if not records:
        return torch.empty((0, width + METADATA_DIM), device=device)
    return torch.cat((torch.stack([record.content.to(device) for record in records]),
                      torch.stack([metadata(record, clocks, device=device) for record in records])), -1)


class CachedScorer:
    """Online CPU key cache: batched encoding, one query transfer, no split GPU sync.

    Cache is per frozen router instance, and reset with the episode. Keys for
    replaced pending prefixes are evicted by retain(). Cache bytes and full
    leaf scans remain part of recorded routing costs.
    """

    def __init__(self, router: LocalRouter, checkpoint_hash: str):
        if router.training or any(p.requires_grad for p in router.parameters()):
            raise ValueError("Key caching requires an eval-mode frozen router")
        self.router = router
        self.checkpoint_hash = checkpoint_hash
        self.keys: dict[str, tuple[torch.Tensor, float]] = {}
        self.query_value = None
        self.key_transferred_bytes = 0
        self.feature_transferred_bytes = 0
        self.query_transferred_bytes = 0
        self.keys_encoded = 0
        # Only the small interaction head is copied; query and candidate towers
        # remain on their original device. Selective traversal has no GPU sync.
        self.cpu_score_head = (deepcopy(router.score_head).cpu().eval().requires_grad_(False)
                               if hasattr(router, "score_head") else None)

    @torch.no_grad()
    def set_query(self, query_input, *, current=None, current_valid=None):
        device = next(self.router.parameters()).device

        def transfer(value):
            if value is None:
                return None
            if value.device != device:
                self.query_transferred_bytes += value.numel() * value.element_size()
            return value.to(device)

        q = self.router.encode_query(
            transfer(query_input), current=transfer(current), current_valid=transfer(current_valid),
        ).float().cpu()
        self.query_value = q
        if next(self.router.parameters()).device.type != "cpu":
            self.query_transferred_bytes += q.numel() * q.element_size()

    def retain(self, records):
        ids = {r.node_id for r in records}
        self.keys = {k: v for k, v in self.keys.items() if k in ids}

    @torch.no_grad()
    def prepare(self, records, *, chunk_size=4096, children=None):
        """Batch all new immutable keys during archive update, before traversal.

        TREE supplies all archive nodes; FLAT supplies all fine leaves. The
        resulting O(N) key-storage/update cost is explicit, even when a decision
        subsequently visits only O(B) tree nodes. Completed keys are reused;
        replaced prefix keys should be removed with retain() at the same update.
        """
        if chunk_size < 1:
            raise ValueError("Key-preparation chunk size must be positive")
        records = list(records)
        if children is not None and len(children) != len(records):
            raise ValueError("Prepared child features must align with candidate records")
        missing_indices = [i for i, r in enumerate(records) if r.node_id not in self.keys
                           and (not self.router.requires_children or children is not None or len(r.children) == 2)]
        missing = [records[i] for i in missing_indices]
        for start in range(0, len(missing), chunk_size):
            batch = missing[start : start + chunk_size]
            features = self.router.candidate_input(batch)
            child_features = None
            if self.router.requires_children:
                child_features = (children[missing_indices[start:start + chunk_size]].to(features.device)
                                  if children is not None else self.router.child_input(batch))
            encoded, offsets = self.router.encode_candidates(features, children=child_features)
            keys, bias = encoded.float().cpu(), offsets.float().cpu().flatten()
            self.keys_encoded += len(batch)
            if features.device.type != "cpu":
                self.key_transferred_bytes += keys.numel() * 4 + bias.numel() * 4
                if batch[0].content_sum.device.type == "cpu":
                    self.feature_transferred_bytes += features.numel() * features.element_size()
                    if child_features is not None:
                        self.feature_transferred_bytes += child_features.numel() * child_features.element_size()
            for r, key, b in zip(batch, keys, bias):
                self.keys[r.node_id] = key, float(b)

    @torch.no_grad()
    def __call__(self, records):
        if self.query_value is None:
            raise RuntimeError("set_query must precede scoring")
        if not records:
            return []
        if any(r.node_id not in self.keys for r in records):
            raise RuntimeError(
                "Call prepare(archive.records) for TREE or prepare(archive.leaves) for FLAT before routing"
            )
        keys = torch.stack([self.keys[r.node_id][0] for r in records])
        bias = torch.tensor([self.keys[r.node_id][1] for r in records])
        return self.router.score_encoded(self.query_value, keys, bias, score_head=self.cpu_score_head).tolist()

    @property
    def storage_bytes(self):
        head_bytes = (sum(p.numel() * p.element_size() for p in self.cpu_score_head.parameters())
                      if self.cpu_score_head is not None else 0)
        return head_bytes + sum(v[0].numel() * 4 + 8 + len(k.encode()) for k, v in self.keys.items())

    @property
    def transferred_bytes(self):
        return self.feature_transferred_bytes + self.key_transferred_bytes + self.query_transferred_bytes


def query_features(
    current_mean,
    text,
    current,
    clocks,
    budget,
    *,
    condition_on_previous_joint_commands=True,
    condition_on_previous_commands=True,
    proprioception_mode="all",
):
    """Only allowed observations. No target, noisy action, episode ID, or duration."""
    device = current_mean.device
    extra = torch.tensor(
        [
            float(current.action_available) if condition_on_previous_commands else 0.0,
            clocks.get("conditioning", 0.0) / 100.0,
            clocks.get("execution", 0.0) / 100.0,
            math.log1p(budget),
        ],
        device=device,
    )
    return torch.cat(
        [
            current_mean,
            text,
            current.proprio.to(device).float()
            if proprioception_mode != "none"
            else torch.zeros_like(current.proprio, device=device).float(),
            command_conditioning(
                current.previous_action.to(device).float(),
                include_joint_commands=condition_on_previous_joint_commands,
                include_commands=condition_on_previous_commands,
            ),
            extra,
        ]
    )
