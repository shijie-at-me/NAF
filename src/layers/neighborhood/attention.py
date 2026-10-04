"""Windowed cross-attention of high-res queries to low-res keys/values (NAF's upsampling attention).

Runs on NATTEN when it can (``natten_backend``), else on the pure-PyTorch fallback (``fallback``), which gives
the same results.

Also the multi-head tensor layouts: feature maps [B, n * D, H, W] <-> heads [B, H, W, n, D]."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .fallback import upsampled_neighborhood_attention
from .natten_backend import can_use_natten, natten_attention
from .windows import upsampling_dilation


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


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9),
        backend=None,
        **kwargs,
    ):
        """``backend`` is passed to NATTEN >= 0.20; ``None`` lets NATTEN pick the fastest one that can run.

        Inputs NATTEN can't handle (no NATTEN, CPU tensors, unsupported head dims, ...) fall back to a pure-PyTorch
        implementation with identical results, as does everything when NATTEN was built without its CUDA kernels
        and no ``backend`` is forced.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.backend = backend

        self.scale = (dim // num_heads) ** -0.5

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        """Each high-res query attends to a window of the low-res keys/values around its location.

        k and v are nearest-upsampled to the query resolution, and the window is dilated by the
        upsampling factor so that it covers ``kernel_size`` distinct low-res positions. With
        ``return_weights``, also returns the softmax attention weights [B, n, H, W, kh * kw].
        """
        size = q.shape[-2:]
        dilation = upsampling_dilation(size, k.shape[-2:])
        q = to_heads(q, self.num_heads)

        if not return_weights and can_use_natten(q, v, self.num_heads, self.backend):
            out = natten_attention(
                q,
                upsample_to_heads(k, size, self.num_heads, q.dtype),
                upsample_to_heads(v, size, self.num_heads, q.dtype),
                self.kernel_size,
                dilation,
                self.scale,
                self.backend,
            )
            if out is not None:
                return from_heads(out)

        # The fallback reads the low-res keys/values directly
        out, attn_weights = upsampled_neighborhood_attention(
            q,
            low_res_heads(k, self.num_heads, q.dtype),
            low_res_heads(v, self.num_heads, q.dtype),
            self.kernel_size,
            dilation,
            self.scale,
            need_weights=return_weights,
        )
        return (from_heads(out), attn_weights) if return_weights else from_heads(out)
