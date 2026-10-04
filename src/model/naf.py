"""NAF: an image encoder gives queries (at the output size) and keys (pooled to the features), the low-res
features are the values of a windowed cross-attention."""

import torch
import torch.nn.functional as F
from torch import nn

from src.layers import CrossAttention, RoPE, encoder
from src.model.base import BaseUpsampler
from src.utils.img import bilinear_resize

__all__ = ["NAF"]


def guide_size(image_size, output_size, max_ratio=4):
    """Size to downsample the guide image to, or None if it is already within ``max_ratio`` x ``output_size``.

    Both sides are capped at ``max_ratio`` times the smaller output side.
    """
    (h, w), (oh, ow) = image_size, output_size
    if h <= max_ratio * oh and w <= max_ratio * ow:
        return None
    cap = max_ratio * min(oh, ow)
    return (min(h, cap), min(w, cap))


class ImageEncoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256,
        heads_rope=1,
        use_encoder=True,
        rope_base=None,
        rope_rescale=None,
        img_layers=2,
    ):
        super().__init__()
        self.use_encoder = use_encoder
        self.out_channels = out_channels

        self.encoder = encoder(in_channels, out_channels // 2, kernel_size=1, ks_res=1, num_layers=img_layers)
        self.sem_encoder = encoder(in_channels, out_channels // 2, kernel_size=3, ks_res=3, num_layers=img_layers)

        self.rope = RoPE(embed_dim=out_channels, num_heads=heads_rope, base=rope_base, rescale_coords=rope_rescale)

    def forward_encoder(self, x, output_size):
        if not self.use_encoder:
            return F.adaptive_avg_pool2d(x, output_size=output_size)

        # Pool each branch before concatenating (pooling is per channel, so the result is the same):
        # the full-resolution concatenation is never materialized, and without grad the first branch's
        # full-resolution output is freed before the second branch runs
        return torch.cat(
            [
                F.adaptive_avg_pool2d(self.encoder(x), output_size=output_size),
                F.adaptive_avg_pool2d(self.sem_encoder(x), output_size=output_size),
            ],
            dim=1,
        )

    def forward(self, x, output_size):
        size = guide_size(x.shape[-2:], output_size)
        if size is not None:
            x = bilinear_resize(x, size)

        x = self.forward_encoder(x, output_size)
        return self.rope(x)


class NAF(BaseUpsampler):
    def __init__(
        self,
        dim=256,
        heads_attn=4,
        heads_rope=4,
        kernel_size=9,
        # ImageEncoder options
        use_encoder=True,
        rope_base=100.0,
        rope_rescale=2.0,
        img_layers=2,
        **kwargs,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            in_channels=3,
            out_channels=dim,
            heads_rope=heads_rope,
            use_encoder=use_encoder,
            rope_base=rope_base,
            img_layers=img_layers,
            rope_rescale=rope_rescale,
        )
        self.upsampler = CrossAttention(dim=dim, num_heads=heads_attn, kernel_size=(kernel_size, kernel_size))

    def forward(self, image, features, output_size, return_weights=False, *args, **kwargs):
        """Upsample ``features`` to ``output_size`` guided by ``image``.

        Queries are the encoded image at ``output_size``; keys are the same encoding pooled to the
        resolution of ``features``, which are the values. Returns ``(out, attn_weights)`` if
        ``return_weights``.
        """
        queries = self.image_encoder(image, output_size=output_size)
        keys = F.adaptive_avg_pool2d(queries, output_size=features.shape[-2:])
        return self.upsampler(queries, keys, features, image, return_weights=return_weights)
