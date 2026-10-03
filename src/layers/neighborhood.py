"""Pure-PyTorch neighborhood cross-attention over nearest-upsampled keys/values.

NAF upsamples the low-res keys/values to the query resolution (nearest-exact) and runs NATTEN's dilated
neighborhood attention there. This module computes the same result straight from the low-res keys/values, so it
runs where NATTEN has no kernel (CPU, NATTEN builds without libnatten, unsupported head dims) and can return the
attention weights. NATTEN's fused kernels are faster: this is the fallback.

Queries are processed a chunk of rows at a time, so the gathered key/value windows of only one chunk exist at
once; with autograd, each chunk is recomputed in the backward pass instead of keeping its windows alive.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Max elements of the gathered key (or value) windows, or of the attention weights, of one chunk of query rows
MAX_CHUNK_ELEMENTS = 2**26


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


def _attend_rows(q, k, v, win_h, win_w, rows_per_window):
    """Attention of a chunk of query rows.

    q: [B, R, W, n, D] (already scaled); k: [B, h, w, n, D]; v: [B, h, w, n, Dv], or [B, h, w, 1, Dv] to apply the
    head-averaged weights to all value channels; win_h: [R / rows_per_window, kh]; win_w: [W', kw].
    Consecutive groups of ``rows_per_window`` query rows share a row of ``win_h``; when ``win_w`` has fewer rows
    than the queries have columns (integer ratio), groups of W / W' query columns share a row of ``win_w``.
    Returns the output [B, R, W, n or 1, Dv] and the softmax weights [B, R, W, n, kh * kw].
    """
    b, rows, wq, n, _ = q.shape
    kh, kw = win_h.shape[1], win_w.shape[1]
    q = q.view(b, rows // rows_per_window, rows_per_window, win_w.shape[0], wq // win_w.shape[0], n, -1)
    # Keys/values in the window of every (group of) query: [B, R', kh, W', kw, n, D]
    k = k[:, win_h][:, :, :, win_w]
    v = v[:, win_h][:, :, :, win_w]
    weights = torch.einsum("byixjnd,byaxcnd->byixjnac", q, k).flatten(-2).softmax(dim=-1)
    applied = weights.mean(dim=5, keepdim=True) if v.shape[-2] == 1 < n else weights
    out = torch.einsum("byixjnac,byaxcne->byixjne", applied.unflatten(-1, (kh, kw)), v)
    return out.reshape(b, rows, wq, v.shape[-2], -1), weights.reshape(b, rows, wq, n, -1)


def _attend_rows_output(q, k, v, win_h, win_w, rows_per_window):
    return _attend_rows(q, k, v, win_h, win_w, rows_per_window)[0]


def upsampled_neighborhood_attention(q, k, v, kernel_size, dilation, scale, need_weights=True, out_dtype=None):
    """Neighborhood attention of ``q`` over ``k``/``v`` nearest-exact upsampled to the resolution of ``q``.

    Equivalent to ``na2d(q, interp(k), interp(v), kernel_size, dilation)`` without materializing the upsampled
    keys/values.

    Args:
        q: [B, H, W, n, D] queries.
        k, v: [B, h, w, n, D] low-res keys and values (``v`` may have another head dim). A ``v`` with a single
            head [B, h, w, 1, Dv] gets the softmax weights averaged over the heads of ``q``/``k``.
        kernel_size, dilation: (height, width) pairs, as for NATTEN.
        scale: Query scaling (NATTEN uses ``D ** -0.5``).
        need_weights: Also return the attention weights (then no chunk is recomputed in the backward pass).
        out_dtype: dtype of the output (default: that of ``q``). Without autograd, each chunk is written straight
            into the output, cast on the fly, instead of being concatenated (and cast) at the end.

    Returns:
        Output [B, H, W, n (or 1), Dv], and softmax attention weights [B, n, H, W, kh * kw] (NATTEN's layout) or
        None.
    """
    (b, hq, wq, n, d), (h, w) = q.shape, k.shape[1:3]
    win_h = low_res_windows(hq, h, kernel_size[0], dilation[0], q.device)
    win_w = low_res_windows(wq, w, kernel_size[1], dilation[1], q.device)

    # With an integer ratio, the window only depends on the low-res cell of the query: attend per low-res cell
    rows_per_window = 1
    if hq % h == 0 and (hq // h) == dilation[0]:
        rows_per_window, win_h = hq // h, win_h[:: hq // h]
    if wq % w == 0 and (wq // w) == dilation[1]:
        win_w = win_w[:: wq // w]

    # Per row of windows, the largest tensor: the gathered keys (or values), the attention weights of its queries,
    # or their output
    num_offsets = kernel_size[0] * kernel_size[1]
    row_elements = b * max(
        win_w.shape[0] * num_offsets * max(n * d, v.shape[-2] * v.shape[-1]),
        rows_per_window * wq * max(n * num_offsets, v.shape[-2] * v.shape[-1]),
    )
    windows_per_chunk = max(1, MAX_CHUNK_ELEMENTS // row_elements)
    # The weights of a checkpointed chunk can't be returned: they are only kept when not training through them
    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    use_checkpoint = not need_weights and needs_grad

    if scale != 1:
        q = q * scale
    out_dtype = out_dtype or q.dtype
    # Without autograd the output is allocated once: concatenating the chunks would briefly need twice its memory
    out = None if needs_grad else q.new_empty((b, hq, wq, *v.shape[-2:]), dtype=out_dtype)
    outs, weights = [], []
    for start in range(0, win_h.shape[0], windows_per_chunk):
        win_rows = win_h[start : start + windows_per_chunk]
        q_rows = q[:, start * rows_per_window : (start + len(win_rows)) * rows_per_window]
        if use_checkpoint:
            # Recompute the chunk in the backward pass instead of storing its key/value windows
            outs.append(
                checkpoint(_attend_rows_output, q_rows, k, v, win_rows, win_w, rows_per_window, use_reentrant=False)
            )
        else:
            chunk, weight = _attend_rows(q_rows, k, v, win_rows, win_w, rows_per_window)
            if out is None:
                outs.append(chunk)
            else:
                out[:, start * rows_per_window : start * rows_per_window + chunk.shape[1]] = chunk
            if need_weights:
                weights.append(weight)

    if out is None:
        out = torch.cat(outs, dim=1).to(out_dtype)
    return out, torch.cat(weights, dim=1).permute(0, 3, 1, 2, 4) if need_weights else None
