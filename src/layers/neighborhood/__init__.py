"""Neighborhood cross-attention of high-res queries to low-res keys / values (the core of NAF-style upsamplers).

- ``attention``: ``CrossAttention``, on NATTEN when it can run, else on the PyTorch fallback; multi-head layouts
- ``natten_backend``: NATTEN detection, its pre-0.20 API, and the inputs it failed on
- ``fallback``: ``upsampled_neighborhood_attention``, the same result straight from the low-res keys / values
- ``windows``: which low-res tokens each query's window holds
"""

from .attention import CrossAttention, from_heads, low_res_heads, to_heads, upsample_to_heads
from .fallback import MAX_CHUNK_ELEMENTS, upsampled_neighborhood_attention
from .natten_backend import HAS_LIBNATTEN, NATTEN_VERSION
from .windows import low_res_windows, natten_window, own_tokens, upsampling_dilation, window_tokens

__all__ = [
    "HAS_LIBNATTEN",
    "MAX_CHUNK_ELEMENTS",
    "NATTEN_VERSION",
    "CrossAttention",
    "from_heads",
    "low_res_heads",
    "low_res_windows",
    "natten_window",
    "own_tokens",
    "to_heads",
    "upsample_to_heads",
    "upsampled_neighborhood_attention",
    "upsampling_dilation",
    "window_tokens",
]
