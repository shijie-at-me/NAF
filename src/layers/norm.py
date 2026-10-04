"""Normalization layers."""

from torch import nn


class CastRMSNorm(nn.RMSNorm):
    """RMSNorm computed in the dtype of its weight and returned in that of the input (e.g. bf16 under autocast)."""

    def forward(self, x):
        return super().forward(x.to(self.weight.dtype)).to(x.dtype)
