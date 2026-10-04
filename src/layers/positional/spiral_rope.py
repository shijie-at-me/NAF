"""Spiral 2D rotary position embedding, ported from PixelUp (MIT License, https://github.com/deepankkumar/PixelUp).

Axial 2D RoPE varies its angles along x and y only; here channel groups use frequency fields rotated to
``num_directions`` directions between 0 and 90 degrees.
"""

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SpiralRoPE2D"]


def rotate_freqs(freqs, angle_deg):
    """Rotate an [n, n, ...] frequency field by ``angle_deg`` about its center (bilinear, zeros outside)."""
    n, _, d1, d2 = freqs.shape
    angle = math.radians(angle_deg)
    flat = freqs.reshape(n, n, -1).permute(2, 0, 1).unsqueeze(0).float()
    theta = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0]],
        dtype=torch.float32,
        device=freqs.device,
    ).unsqueeze(0)
    grid = F.affine_grid(theta, flat.size(), align_corners=True)
    rot = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return rot.squeeze(0).permute(1, 2, 0).reshape(n, n, d1, d2)


class SpiralRoPE2D(nn.Module):
    """2D RoPE whose channel groups use frequencies rotated to ``num_directions`` directions (not only x and y).

    Channels are rotated in interleaved pairs (2i, 2i + 1), which always share the same angle: the sin/cos tables
    are kept at half width, one value per pair, and cached for the last ``cache_size`` grid sizes.
    """

    def __init__(
        self, dim: int, num_directions: int = 16, theta: float = 10000.0, pt_seq_len: int = 14, cache_size: int = 8
    ):
        super().__init__()
        self.dim, self.num_directions, self.pt_seq_len = dim, num_directions, pt_seq_len
        eff = dim // num_directions
        min_f = 1.0 / (theta ** ((eff - 2) / eff))
        half = eff // 2
        freqs = min_f * (theta ** (torch.arange(half - 1, -1, -1).float() / max(half - 1, 1)))
        self.register_buffer("freqs", freqs)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    def build_freqs_2d(self, h, w, device, dtype):
        """Angles of every channel at every position: [h, w, dim]."""
        k, n = self.num_directions, max(h, w)
        t = torch.arange(n, device=device, dtype=dtype) / n * self.pt_seq_len
        freqs = torch.einsum("i, f -> i f", t, self.freqs.to(dtype)).repeat_interleave(2, dim=1)
        freqs = freqs.reshape(n, freqs.shape[1] // 4, 4).repeat_interleave(k, dim=1).reshape(n, -1)
        f2d = torch.cat([freqs[:, None, :].expand(n, n, -1), freqs[None, :, :].expand(n, n, -1)], dim=-1)
        group = k * 4
        f2d = f2d.view(n, n, f2d.shape[-1] // group, group)
        for i in range(1, k):
            f2d[..., i * 4 : (i + 1) * 4] = rotate_freqs(f2d[..., i * 4 : (i + 1) * 4], i * 90.0 / k)
        return f2d.view(n, n, -1)[:h, :w, : self.dim]

    def tables(self, h, w, device, dtype):
        """Half-width cos/sin tables [h, w, dim / 2] (one entry per channel pair), cached."""
        key = (h, w, dtype, device)
        entry = self._cache.get(key)
        if entry is None:
            f = self.build_freqs_2d(h, w, device, dtype)
            pairs = f.unflatten(-1, (-1, 2))
            assert torch.equal(pairs[..., 0], pairs[..., 1]), "the channels of a pair must share their angle"
            half = pairs[..., 0]
            entry = (half.cos().contiguous(), half.sin().contiguous())
            self._cache[key] = entry
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return entry

    def forward(self, x):
        """Rotate x [B, C, H, W]; the result has the same shape and memory layout."""
        _, c, h, w = x.shape
        cos, sin = self.tables(h, w, x.device, x.dtype)
        x = x.permute(0, 2, 3, 1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        if torch.is_grad_enabled() and x.requires_grad:
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
            out = x * cos.repeat_interleave(2, dim=-1) + rotated * sin.repeat_interleave(2, dim=-1)
        else:
            # Written pair by pair into the output (half-size temporaries only), rounded like the expression above
            out = torch.empty_like(x)
            pairs = out.unflatten(-1, (c // 2, 2))
            torch.sub(x1 * cos, x2 * sin, out=pairs[..., 0])
            torch.add(x2 * cos, x1 * sin, out=pairs[..., 1])
        return out.permute(0, 3, 1, 2)
