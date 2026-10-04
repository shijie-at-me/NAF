"""Index math of neighborhood attention over nearest-upsampled keys/values: which low-res tokens a query sees.

The keys/values are nearest-exact upsampled to the query resolution and every query attends to a dilated window
there (NATTEN's neighborhoods), with the dilation equal to the upsampling factor so the window covers
``kernel_size`` distinct low-res tokens per axis. These helpers give the low-res tokens of every window directly.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def upsampling_dilation(out_size, in_size, max_dilation: int | None = None) -> tuple[int, int]:
    """Window dilation (per axis) for queries of ``out_size`` over keys/values of ``in_size``: the integer
    upsampling factor, at least 1 and at most ``max_dilation``."""
    dilation = tuple(max(o // i, 1) for o, i in zip(out_size, in_size, strict=True))
    if max_dilation is not None:
        dilation = tuple(min(d, max_dilation) for d in dilation)
    return dilation


def natten_window(length: int, kernel_size: int, dilation: int, device=None) -> Tensor:
    """Indices of the ``kernel_size`` neighbors of every position along one axis: [length, kernel_size].

    Same neighborhoods as NATTEN: positions are split into ``dilation`` interleaved groups and, within its group,
    each position takes the window centered on it, shifted inwards at the borders.
    """
    idx = torch.arange(length, device=device)
    group, pos = idx % dilation, idx // dilation
    group_len = (length - group + dilation - 1) // dilation
    start = torch.minimum((pos - kernel_size // 2).clamp(min=0), group_len - kernel_size)
    return group[:, None] + (start[:, None] + torch.arange(kernel_size, device=device)) * dilation


def low_res_windows(out_len: int, in_len: int, kernel_size: int, dilation: int, device=None) -> Tensor:
    """Low-res index of every neighbor of every query along one axis: [out_len, kernel_size].

    The neighbors live on the nearest-exact upsampled grid; the source index of each upsampled position is read off
    ``F.interpolate`` itself, so the mapping matches the upsampling bit for bit.
    """
    src = torch.arange(in_len, device=device, dtype=torch.float32).view(1, 1, -1)
    src = F.interpolate(src, size=out_len, mode="nearest-exact").view(-1).long()
    return src[natten_window(out_len, kernel_size, dilation, device)]


def window_tokens(out_size, in_size, kernel_size, dilation, device=None) -> Tensor:
    """Flat low-res index of every token each query attends to: [H, W, kh * kw], in the order of the weights."""
    (hq, wq), (h, w) = out_size, in_size
    win_h = low_res_windows(hq, h, kernel_size[0], dilation[0], device)
    win_w = low_res_windows(wq, w, kernel_size[1], dilation[1], device)
    return (win_h[:, None, :, None] * w + win_w[None, :, None, :]).flatten(-2)


def own_tokens(out_size, in_size, device=None) -> Tensor:
    """Flat low-res index of the token each query copies under nearest-exact upsampling: [H, W]."""
    (hq, wq), (h, w) = out_size, in_size
    # A 1-wide, undilated window is the source position itself
    src_h, src_w = low_res_windows(hq, h, 1, 1, device)[:, 0], low_res_windows(wq, w, 1, 1, device)[:, 0]
    return src_h[:, None] * w + src_w[None, :]
