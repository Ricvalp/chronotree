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

    def forward(self, text: str | list[str]):
        """Encode strings together without allowing padding into pooled features."""
        unbatched = isinstance(text, str)
        texts = [text] if unbatched else text
        if not texts or not all(isinstance(item, str) for item in texts):
            raise ValueError("TextEncoder requires a string or a nonempty list of strings")
        sequences = [[256] + list(item.encode("utf-8")[: self.max_bytes]) for item in texts]
        length = max(map(len, sequences))
        ids = torch.tensor(
            [sequence + [0] * (length - len(sequence)) for sequence in sequences],
            device=self.position.device,
            dtype=torch.long,
        )
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=ids.device)
        padding = torch.arange(length, device=ids.device)[None] >= lengths[:, None]
        x = self.embedding(ids) + self.position[None, :length]
        encoded = self.transformer(x, src_key_padding_mask=padding)
        pooled = encoded.masked_fill(padding[..., None], 0).sum(1) / lengths[:, None]
        result = self.norm(pooled)
        return result[0] if unbatched else result
