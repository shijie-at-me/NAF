"""PixelUp: zero-shot semantic feature upsampling (Singh, Nihal and Hoskere, arXiv:2608.02792).

Port of https://github.com/deepankkumar/PixelUp (MIT License, v0.1.0, ``pixelup/models/pixelup.py``) to this
repository's upsampler interface. The architecture and parameter names are unchanged, so the released checkpoints
load as they are. Differences from the original:

- Neighborhood attention goes through ``src.layers``: NATTEN when it can run, else the PyTorch fallback. The
  original needs NATTEN < 0.20 for the split ``na2d_qk`` / ``na2d_av`` kernels of its decoder, which averages the
  softmax weights over heads before applying them to the values; the fallback does that directly on the low-res
  values (no upsampled copy of them, no chunking over value channels).
- ``checkpoint`` may be a URL (downloaded once to ``weights/pixelup/``) and the Semantic Encoder weights must come
  from the checkpoint (the released ones carry them) or from a local Hugging Face snapshot dir.
- Without autograd, the full-resolution tensors are updated in place, each one is released as soon as the next
  exists, and the decoder writes its float32 output directly: the same results with a lower peak memory.

The package splits it into ``blocks`` (the encoder and decoder modules), ``semantic_encoder`` (the frozen DINOv3
ConvNeXt pyramid), ``checkpoint`` (release assets, architecture resolution, Semantic Encoder weights) and ``model``
(``PixelUp``).

How it works: a frozen DINOv3 ConvNeXt (the Semantic Encoder, ``semantic_encoder.py``) encodes the image at
``semantic_scale`` x the output resolution. Pixel-encoder queries are refined coarse-to-fine through its four stages
(each: upsample the stage features, add them, windowed cross-attention to them), then a NAF-like decoder lets every
output pixel attend to a window of the backbone's low-res features (the values) and returns their weighted average.
"""

from .model import PixelUp

__all__ = ["PixelUp"]
