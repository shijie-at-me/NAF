import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .neighborhood import upsampled_neighborhood_attention

try:  # NATTEN < 0.20
    from natten.functional import na2d_av, na2d_qk

    NATTEN_VERSION = "legacy"
except ImportError:
    try:  # NATTEN >= 0.20
        from natten import na2d

        NATTEN_VERSION = "recent"
    except ImportError:
        NATTEN_VERSION = None

try:
    # Without its compiled kernels (libnatten), NATTEN >= 0.20 can only use Flex Attention, which uncompiled
    # materializes dense masks (hundreds of GB at 448x448): the PyTorch fallback is then the better default
    from natten import HAS_LIBNATTEN
except ImportError:
    HAS_LIBNATTEN = NATTEN_VERSION == "legacy"

# (device type, dtype, q/k head dim, v head dim) that NATTEN failed on; they go straight to the PyTorch fallback
_NATTEN_UNSUPPORTED = set()
if NATTEN_VERSION is None:
    warnings.warn("NATTEN is not installed: neighborhood attention uses the (slower) PyTorch fallback", stacklevel=1)


def natten_key(q, v, num_heads=None):
    """Key of ``_NATTEN_UNSUPPORTED`` for queries [B, H, W, n, D] and values in heads (or [B, n * D, h, w]) layout."""
    dim_v = v.shape[-1] if num_heads is None else v.shape[1] // num_heads
    return (q.device.type, q.dtype, q.shape[-1], dim_v)


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


def legacy_attention(q, k, v, kernel_size, dilation, scale=1):
    """Neighborhood attention with the pre-0.20 NATTEN ops; inputs and output are [B, H, W, n, D]."""
    q = rearrange(q * scale, "b h w n d -> b n h w d")  # scaling q is cheaper than scaling the K*K scores
    k = rearrange(k, "b h w n d -> b n h w d")
    v = rearrange(v, "b h w n d -> b n h w d")
    attn_weights = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation).softmax(dim=-1)
    features = na2d_av(attn_weights, v, kernel_size=kernel_size, dilation=dilation)
    return rearrange(features, "b n h w d -> b h w n d")


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

    def natten_attend(self, q, k, v, dilation):
        """NATTEN neighborhood attention on [B, H, W, n, D] tensors, or None if NATTEN can't run it."""
        key = natten_key(q, v)
        try:
            if NATTEN_VERSION == "legacy":
                return legacy_attention(q, k, v, self.kernel_size, dilation, scale=self.scale)
            # Modern na2d uses head_dim ** -0.5, which equals self.scale
            return na2d(q, k, v, kernel_size=self.kernel_size, dilation=dilation, stride=1, backend=self.backend)
        except torch.OutOfMemoryError:
            raise
        except (RuntimeError, NotImplementedError, ValueError, AssertionError) as e:
            _NATTEN_UNSUPPORTED.add(key)
            warnings.warn(f"NATTEN can't run {key}, using the PyTorch fallback: {e}", stacklevel=2)
            return None

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        """Each high-res query attends to a window of the low-res keys/values around its location.

        k and v are nearest-upsampled to the query resolution, and the window is dilated by the
        upsampling factor so that it covers ``kernel_size`` distinct low-res positions. With
        ``return_weights``, also returns the softmax attention weights [B, n, H, W, kh * kw].
        """
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        dilation = (max(hq // hk, 1), max(wq // wk, 1))
        q = to_heads(q, self.num_heads)

        use_natten = (
            NATTEN_VERSION is not None
            and (HAS_LIBNATTEN or self.backend is not None)
            and natten_key(q, v, self.num_heads) not in _NATTEN_UNSUPPORTED
        )
        if use_natten and not return_weights:
            out = self.natten_attend(
                q,
                upsample_to_heads(k, (hq, wq), self.num_heads, q.dtype),
                upsample_to_heads(v, (hq, wq), self.num_heads, q.dtype),
                dilation,
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
