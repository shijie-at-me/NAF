"""Building blocks of the upsamplers.

- ``attention``: ``CrossAttention``, windowed attention of high-res queries to low-res keys/values, on NATTEN
  (``natten_backend``) when it can run, else on the pure-PyTorch ``neighborhood`` attention;
- ``windows``: which low-res tokens each query's window holds; ``heads``: multi-head tensor layouts;
- ``rope`` (axial, DINOv3) and ``spiral_rope`` (PixelUp): rotary position embeddings;
- ``convolutions``: conv encoders; ``upsample``: learned upsampling layers (PixelUp); ``norm``: normalizations.
"""

from .attention import CrossAttention
from .convolutions import EncBlock, encoder, same_conv
from .heads import from_heads, low_res_heads, to_heads, upsample_to_heads
from .natten_backend import HAS_LIBNATTEN, NATTEN_VERSION
from .neighborhood import upsampled_neighborhood_attention
from .norm import CastRMSNorm
from .rope import RoPE
from .spiral_rope import SpiralRoPE2D
from .upsample import build_up_module
from .windows import low_res_windows, natten_window, own_tokens, upsampling_dilation, window_tokens
