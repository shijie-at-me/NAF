"""Windowed cross-attention of high-res queries to low-res keys/values (NAF's upsampling attention).

Runs on NATTEN when it can (``natten_backend``), else on the pure-PyTorch fallback (``neighborhood``), which gives
the same results.
"""

import torch.nn as nn

from .heads import from_heads, low_res_heads, to_heads, upsample_to_heads
from .natten_backend import can_use_natten, natten_attention
from .neighborhood import upsampled_neighborhood_attention
from .windows import upsampling_dilation


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
