"""One metadata schema for leaves, tree nodes, and disconnected residuals."""

from __future__ import annotations
import math
import torch
from torch import nn

METADATA_DIM = 23


def metadata(record, clocks: dict[str, float] | None = None, *, device=None) -> torch.Tensor:
    """Static source-local times; read-time ages never enter frozen candidate keys."""
    device = device or record.content.device
    clocks = clocks or {}
    now = clocks.get(record.source, 0.0)
    scalars = [
        record.time_min / 100.0,
        record.time_max / 100.0,
        (now - record.time_min) / 100.0,
        (now - record.time_max) / 100.0,
        math.log(max(record.count, 1)),
        float(record.depth) / 16.0,
        math.log1p(record.scale),
        float(record.completed),
        float(record.disconnected),
        float(record.source == "conditioning"),
        float(record.source == "execution"),
        float(record.modality == "visual"),
        float(record.modality == "event"),
        float(record.count == 1),
    ]
    return torch.cat(
        [
            record.anchor.to(device).float(),
            record.bounds_min.to(device).float(),
            record.bounds_max.to(device).float(),
            torch.tensor(scalars, device=device),
        ]
    ).float()


class MetadataAdapter(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.content = nn.Linear(width, width)
        self.meta = nn.Sequential(nn.Linear(METADATA_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, records: list, clocks: dict, device, *, timeless=False) -> torch.Tensor:
        if not records:
            return torch.empty(0, self.content.out_features, device=device)
        latent = torch.stack([r.content.to(device).float() for r in records])
        meta = torch.stack([metadata(r, clocks, device=device) for r in records])
        if timeless:
            # Current-only ablations keep observed geometry and token type,
            # but cannot infer episode progress from absolute clocks or ages.
            meta[:, 9:13] = 0
        return self.norm(self.content(latent) + self.meta(meta))
