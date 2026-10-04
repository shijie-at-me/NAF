"""Building blocks of the upsamplers.

- ``neighborhood``: ``CrossAttention``, windowed attention of high-res queries to low-res keys/values (NATTEN or the
  pure-PyTorch fallback), its windows and multi-head layouts;
- ``positional``: rotary position embeddings, axial (DINOv3) and spiral (PixelUp);
- ``convolutions``: conv encoders; ``upsample``: learned upsampling layers (PixelUp); ``norm``: normalizations;
- ``bilateral``: learned joint bilateral upsampling (FeatUp, needs its CUDA extension).
"""

from .bilateral import JBULearnedRange
from .convolutions import EncBlock, encoder, same_conv
from .neighborhood import (
    HAS_LIBNATTEN,
    NATTEN_VERSION,
    CrossAttention,
    from_heads,
    low_res_heads,
    low_res_windows,
    natten_window,
    own_tokens,
    to_heads,
    upsample_to_heads,
    upsampled_neighborhood_attention,
    upsampling_dilation,
    window_tokens,
)
from .norm import CastRMSNorm, ChannelNorm
from .positional import RoPE, SpiralRoPE2D
from .upsample import build_up_module
