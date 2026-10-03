"""Learned 2x (or ``scale`` x) feature upsampling layers, ported from PixelUp (MIT License,
https://github.com/deepankkumar/PixelUp): nearest + conv, pixel shuffle, and CARAFE (Wang et al., ICCV 2019)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NNConvUpsample", "PixelShuffleUpsample", "CARAFEUpsample", "build_up_module"]


class NNConvUpsample(nn.Module):
    """Nearest-neighbor upsampling followed by a 3x3 conv."""

    def __init__(self, c_in: int, d_out: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(c_in, d_out, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=self.scale, mode="nearest"))


class PixelShuffleUpsample(nn.Sequential):
    """3x3 conv to ``scale**2`` times the channels, rearranged into a ``scale`` x larger map."""

    def __init__(self, c_in: int, d_out: int, scale: int = 2):
        super().__init__(nn.Conv2d(c_in, d_out * scale * scale, 3, padding=1, bias=False), nn.PixelShuffle(scale))


class CARAFEUpsample(nn.Module):
    """Content-aware reassembly: each output pixel is a predicted ``k_up`` x ``k_up`` kernel over its neighbors."""

    def __init__(self, c_in: int, d_out: int, scale: int = 2, cm: int = 64, k_encoder: int = 3, k_up: int = 5):
        super().__init__()
        self.scale, self.k_up = scale, k_up
        self.compress = nn.Conv2d(c_in, cm, 1, bias=False)
        self.content_encoder = nn.Conv2d(
            cm, (scale * scale) * (k_up * k_up), k_encoder, padding=k_encoder // 2, bias=False
        )
        self.proj = nn.Conv2d(c_in, d_out, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        s, k = self.scale, self.k_up
        weights = self.content_encoder(self.compress(x)).view(b, s * s, k * k, h, w).softmax(dim=2)
        windows = F.unfold(x, kernel_size=k, padding=k // 2).view(b, c, k * k, h, w)
        out = torch.einsum("bskhw, bckhw -> bcshw", weights, windows)
        out = out.view(b, c, s, s, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * s, w * s)
        return self.proj(out)


UP_MODULES = {"nnconv": NNConvUpsample, "pixshuffle": PixelShuffleUpsample, "carafe": CARAFEUpsample}


def build_up_module(method: str, c_in: int, d_out: int, scale: int = 2) -> nn.Module:
    """The upsampling layer named ``method`` (``nnconv``, ``pixshuffle`` or ``carafe``, case-insensitive)."""
    if method.lower() not in UP_MODULES:
        raise ValueError(f"unknown up_method={method!r}, expected one of {sorted(UP_MODULES)}")
    return UP_MODULES[method.lower()](c_in, d_out, scale)
