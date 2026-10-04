# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import torch
from torch import Tensor, nn


def rope_periods(
    head_dim: int,
    *,
    base: float = None,
    min_period: float = None,
    max_period: float = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Rotation periods, one per (u, v) frequency pair: shape [head_dim // 4]."""
    n = head_dim // 4
    if base is not None:
        return base ** (2 * torch.arange(n, device=device, dtype=dtype) / (head_dim // 2))
    return torch.logspace(math.log10(min_period), math.log10(max_period), steps=n, device=device, dtype=dtype)


def grid_coords(
    h: int,
    w: int,
    normalize: Literal["min", "max", "separate"] = "separate",
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Pixel-center coordinates of an h x w grid in [-1, 1]: shape [h * w, 2]."""
    if normalize == "max":
        denom_h = denom_w = max(h, w)
    elif normalize == "min":
        denom_h = denom_w = min(h, w)
    elif normalize == "separate":
        denom_h, denom_w = h, w
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize}")

    coords_h = torch.arange(0.5, h, device=device, dtype=dtype) / denom_h
    coords_w = torch.arange(0.5, w, device=device, dtype=dtype) / denom_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
    return 2.0 * coords - 1.0


def _log_uniform(n: int, max_factor: float, like: Tensor) -> Tensor:
    """``n`` samples, log-uniform in [1 / max_factor, max_factor]."""
    log_max = math.log(max_factor)
    return torch.empty(n, device=like.device, dtype=like.dtype).uniform_(-log_max, log_max).exp()


def augment_coords(coords: Tensor, shift: float = None, jitter: float = None, rescale: float = None) -> Tensor:
    """Randomly shift, jitter (per axis) and rescale (both axes) the coordinates. Does not modify ``coords``."""
    if shift is not None:
        coords = coords + torch.empty(2, device=coords.device, dtype=coords.dtype).uniform_(-shift, shift)
    if jitter is not None:
        coords = coords * _log_uniform(2, jitter, coords)
    if rescale is not None:
        coords = coords * _log_uniform(1, rescale, coords)
    return coords


def rope_sin_cos(coords: Tensor, periods: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
    """Half-width sin/cos tables laid out channel-first: each [head_dim // 2, h, w].

    Channel order is [u_1 .. u_{D/4}, v_1 .. v_{D/4}]; the full-width table would just repeat it twice.
    """
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]  # [HW, 2, D/4]
    angles = angles.flatten(1, 2).T.reshape(-1, h, w)  # [D/2, H, W]
    return angles.sin(), angles.cos()


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor, dim: int) -> Tensor:
    """Rotate each pair (x[i], x[i + D/2]) along ``dim`` by the angle whose half-width sin/cos are given.

    Equivalent to ``x * cat([cos, cos]) + rotate_half(x) * cat([sin, sin])`` without materializing
    the full-width tables or the rotated copy of ``x``.
    """
    x1, x2 = x.chunk(2, dim=dim)
    # addcmul fuses the multiply-add, saving one temporary per half
    return torch.cat([torch.addcmul(x1 * cos, x2, sin, value=-1), torch.addcmul(x2 * cos, x1, sin)], dim=dim)


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RoPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        min_period: float = None,
        max_period: float = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float = None,
        jitter_coords: float = None,
        rescale_coords: float = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.num_heads = num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype  # Don't rely on self.periods.dtype

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.register_buffer(
            "periods",
            rope_periods(
                self.D_head, base=base, min_period=min_period, max_period=max_period, device=device, dtype=dtype
            ),
            persistent=True,
        )

        # Deterministic tables, rebuilt when the grid size, device or periods change
        self._cache_key = None
        self._coords = None
        self._sin_cos = None

    @property
    def augments_coords(self) -> bool:
        return self.training and any(
            v is not None for v in (self.shift_coords, self.jitter_coords, self.rescale_coords)
        )

    def _refresh_cache(self, h: int, w: int):
        key = (h, w, self.periods.device, self.periods.dtype, self.periods._version)
        if key == self._cache_key:
            return

        # Build plain tensors even under inference_mode so they can later be reused in training
        with torch.inference_mode(False):
            self._coords = grid_coords(h, w, self.normalize_coords, device=self.periods.device, dtype=self.dtype)
            self._sin_cos = rope_sin_cos(self._coords, self.periods, h, w)
        self._cache_key = key

    def sin_cos(self, h: int, w: int) -> tuple[Tensor, Tensor]:
        """Half-width sin/cos tables for an h x w grid, with fresh random augmentation in training."""
        self._refresh_cache(h, w)
        if not self.augments_coords:
            return self._sin_cos

        coords = augment_coords(self._coords, self.shift_coords, self.jitter_coords, self.rescale_coords)
        return rope_sin_cos(coords, self.periods, h, w)

    def forward(self, x: Tensor, layout: str = "spatial") -> Tensor:
        """Rotate ``x`` of shape [B, num_heads * D_head, H, W].

        Returns [B, C, H, W] for ``layout="spatial"``, [B, H, W, num_heads, D_head] for ``"flatten"``,
        and [B, num_heads, H * W, D_head] otherwise.
        """
        b, _, h, w = x.shape
        sin, cos = self.sin_cos(h, w)

        # Rotate in the channel-first layout so no permuted copy of x is needed; compute in the
        # tables' precision and return in the input's (e.g. bf16 under autocast)
        x = x.unflatten(1, (self.num_heads, self.D_head))  # [B, n, D, H, W]
        x = rope_apply(x, sin, cos, dim=2).to(x.dtype)

        if layout == "spatial":
            return x.flatten(1, 2)
        if layout == "flatten":
            return x.permute(0, 3, 4, 1, 2)
        return x.flatten(3).transpose(2, 3)
