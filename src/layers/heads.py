"""Layouts of multi-head attention tensors: feature maps [B, n * D, H, W] <-> heads [B, H, W, n, D]."""

import torch
import torch.nn.functional as F
from einops import rearrange


def to_heads(x, num_heads):
    """[B, n * D, H, W] -> [B, H, W, n, D]."""
    return rearrange(x, "b (n d) h w -> b h w n d", n=num_heads)


def from_heads(x):
    """[B, H, W, n, D] -> [B, n * D, H, W]."""
    return rearrange(x, "b h w n d -> b (n d) h w")


def low_res_heads(x, num_heads, dtype):
    """[B, n * D, h, w] -> [B, h, w, n, D] in ``dtype``."""
    return x.to(dtype).permute(0, 2, 3, 1).unflatten(-1, (num_heads, -1))


def upsample_to_heads(x, size, num_heads, dtype):
    """Nearest-exact upsample [B, n * D, h, w] to ``size`` and split heads: [B, H, W, n, D] in ``dtype``.

    Casting and switching to channels-last happen at the low resolution, so the only full-resolution
    tensor allocated is the upsampled output, already in the [B, H, W, C] layout NATTEN reads (nearest
    interpolation only copies values, so the result is identical to casting afterwards).
    """
    x = x.to(dtype).contiguous(memory_format=torch.channels_last)
    # Autocast would run the interpolation in fp32; nearest only copies values, so keep ``dtype``
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = F.interpolate(x, size=size, mode="nearest-exact")
    return x.permute(0, 2, 3, 1).unflatten(-1, (num_heads, -1))
