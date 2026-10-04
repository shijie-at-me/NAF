"""The feature pipeline shared by training and evaluation: frozen backbone features, then an upsampler."""

import torch
import torch.nn.functional as F

from src.utils.image import normalize_pair

__all__ = ["backbone_features", "upsample_features"]


def backbone_features(backbone, images):
    """``(upsampler guide, low-res features)`` of ``images`` in [0, 1]: the ImageNet-normalized images, and the features
    of the frozen ``backbone`` (no gradient) for the images normalized its own way."""
    img_ups, img_bck = normalize_pair(images, backbone)
    with torch.no_grad():
        return img_ups, backbone(img_bck)


def upsample_features(backbone, upsampler, images, output_size, guide_size=None):
    """Features of the frozen ``backbone`` for ``images`` (in [0, 1]) upsampled to ``output_size``.

    The upsampler is guided by the ImageNet-normalized images, resized to ``guide_size`` if given. Gradients flow
    into the upsampler (wrap the call in ``torch.no_grad()`` when it is frozen) but never into the backbone.

    Returns:
        (high-res features, low-res backbone features)
    """
    img_ups, lr_feats = backbone_features(backbone, images)
    if guide_size is not None:
        img_ups = F.interpolate(img_ups, size=guide_size, mode="bicubic", align_corners=False)
    return upsampler(img_ups, lr_feats, output_size), lr_feats
