"""Small teacher-trained byte encoder, without external pretraining."""

import torch
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, width, heads=4, layers=2, max_bytes=256):
        super().__init__()
        self.max_bytes = max_bytes
        self.embedding = nn.Embedding(257, width)
        self.position = nn.Parameter(torch.randn(max_bytes + 1, width) * 0.01)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, text: str):
        ids = torch.tensor(
            [256] + list(text.encode("utf-8")[: self.max_bytes]),
            device=self.position.device,
            dtype=torch.long,
        )
        x = self.embedding(ids) + self.position[: len(ids)]
        return self.norm(self.transformer(x[None]).mean(1)).squeeze(0)
