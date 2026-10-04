"""NATTEN neighborhood attention, when it can run: version detection, the pre-0.20 API, and a memory of the inputs
NATTEN failed on (they go straight to the PyTorch fallback, ``fallback``, afterwards)."""

import warnings

import torch
from einops import rearrange

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

# (device type, dtype, q/k head dim, v head dim) that NATTEN failed on
_NATTEN_UNSUPPORTED = set()
if NATTEN_VERSION is None:
    warnings.warn("NATTEN is not installed: neighborhood attention uses the (slower) PyTorch fallback", stacklevel=1)


def natten_key(q, v, num_heads=None):
    """Key of ``_NATTEN_UNSUPPORTED`` for queries [B, H, W, n, D] and values in heads (or [B, n * D, h, w]) layout."""
    dim_v = v.shape[-1] if num_heads is None else v.shape[1] // num_heads
    return (q.device.type, q.dtype, q.shape[-1], dim_v)


def can_use_natten(q, v, num_heads=None, backend=None):
    """Whether to try NATTEN: it is installed, has its kernels (or ``backend`` forces one) and hasn't failed on
    such inputs before (see ``natten_key`` for the layouts)."""
    return (
        NATTEN_VERSION is not None
        and (HAS_LIBNATTEN or backend is not None)
        and natten_key(q, v, num_heads) not in _NATTEN_UNSUPPORTED
    )


def legacy_attention(q, k, v, kernel_size, dilation, scale=1):
    """Neighborhood attention with the pre-0.20 NATTEN ops; inputs and output are [B, H, W, n, D]."""
    q = rearrange(q * scale, "b h w n d -> b n h w d")  # scaling q is cheaper than scaling the K*K scores
    k = rearrange(k, "b h w n d -> b n h w d")
    v = rearrange(v, "b h w n d -> b n h w d")
    attn_weights = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation).softmax(dim=-1)
    features = na2d_av(attn_weights, v, kernel_size=kernel_size, dilation=dilation)
    return rearrange(features, "b n h w d -> b h w n d")


def natten_attention(q, k, v, kernel_size, dilation, scale, backend=None):
    """NATTEN neighborhood attention on [B, H, W, n, D] tensors, or None if NATTEN can't run it (remembered).

    ``scale`` must be ``D ** -0.5`` with NATTEN >= 0.20, whose ``na2d`` always uses it; ``backend`` is passed to it.
    """
    key = natten_key(q, v)
    try:
        if NATTEN_VERSION == "legacy":
            return legacy_attention(q, k, v, kernel_size, dilation, scale=scale)
        return na2d(q, k, v, kernel_size=kernel_size, dilation=dilation, stride=1, backend=backend)
    except torch.OutOfMemoryError:
        raise
    except (RuntimeError, NotImplementedError, ValueError, AssertionError) as e:
        _NATTEN_UNSUPPORTED.add(key)
        warnings.warn(f"NATTEN can't run {key}, using the PyTorch fallback: {e}", stacklevel=3)
        return None
