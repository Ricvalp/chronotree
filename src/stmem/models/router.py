"""Matched two-tower local scorers with immutable candidate-key caching."""

from __future__ import annotations
import math
import torch
from torch import nn
from stmem.models.metadata_adapter import metadata, METADATA_DIM


class LocalRouter(nn.Module):
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

    def forward(self, query_input, candidate_input):
        q = self.query(query_input.float())
        k = self.key(candidate_input.float())
        return (q * k).sum(-1) * self.scale + self.bias(candidate_input[..., -METADATA_DIM:]).squeeze(-1)


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

    @torch.no_grad()
    def set_query(self, query_input):
        q = self.router.query(query_input.to(next(self.router.parameters()).device)).float().cpu()
        self.query_value = q
        if next(self.router.parameters()).device.type != "cpu":
            self.query_transferred_bytes += q.numel() * q.element_size()

    def retain(self, records):
        ids = {r.node_id for r in records}
        self.keys = {k: v for k, v in self.keys.items() if k in ids}

    @torch.no_grad()
    def prepare(self, records, *, chunk_size=4096):
        """Batch all new immutable keys during archive update, before traversal.

        TREE supplies all archive nodes; FLAT supplies all fine leaves. The
        resulting O(N) key-storage/update cost is explicit, even when a decision
        subsequently visits only O(B) tree nodes. Completed keys are reused;
        replaced prefix keys should be removed with retain() at the same update.
        """
        if chunk_size < 1:
            raise ValueError("Key-preparation chunk size must be positive")
        missing = [r for r in records if r.node_id not in self.keys]
        for start in range(0, len(missing), chunk_size):
            batch = missing[start : start + chunk_size]
            features = self.router.candidate_input(batch)
            keys = self.router.key(features).float().cpu()
            bias = self.router.bias(features[:, -METADATA_DIM:]).float().cpu().flatten()
            self.keys_encoded += len(batch)
            if features.device.type != "cpu":
                self.key_transferred_bytes += keys.numel() * 4 + bias.numel() * 4
                if batch[0].content_sum.device.type == "cpu":
                    self.feature_transferred_bytes += features.numel() * features.element_size()
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
        return (keys @ self.query_value * self.router.scale + bias).tolist()

    @property
    def storage_bytes(self):
        return sum(v[0].numel() * 4 + 8 + len(k.encode()) for k, v in self.keys.items())

    @property
    def transferred_bytes(self):
        return self.feature_transferred_bytes + self.key_transferred_bytes + self.query_transferred_bytes


def query_features(current_mean, text, current, clocks, budget):
    """Only allowed observations. No target, noisy action, episode ID, or duration."""
    device = current_mean.device
    extra = torch.tensor(
        [
            float(current.action_available),
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
            current.proprio.to(device).float(),
            current.previous_action.to(device).float(),
            extra,
        ]
    )
