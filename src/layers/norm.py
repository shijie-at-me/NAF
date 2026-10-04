"""Normalization layers."""

from torch import nn


class CastRMSNorm(nn.RMSNorm):
    """RMSNorm computed in the dtype of its weight and returned in that of the input (e.g. bf16 under autocast)."""

    def forward(self, x):
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)


class ChannelNorm(nn.Module):
    """LayerNorm over the channels of [B, C, H, W] maps."""

    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
