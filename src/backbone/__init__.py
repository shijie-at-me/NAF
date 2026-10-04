"""Frozen backbones (the vision foundation models whose features get upsampled).

- ``registry``: what a backbone name tells (family, fine-tune tag, patch size, feature dim, short name); no torch import
- ``families``: per family (timm, RADIO, Franca, CAPI), how to build the model and read its patch features
- ``wrapper``: ``PretrainedViTWrapper``, the backbone module the rest of the code uses, and fine-tuned weights
- ``loading``: frozen backbones from configs
- ``features``: the feature pipeline, frozen backbone features then an upsampler (``upsample_features``)
"""

from .families import create_backbone_model
from .features import backbone_features, upsample_features
from .loading import load_backbone, load_multiple_backbones
from .registry import (
    MODEL_LIST,
    SHORT_NAMES,
    feature_dim,
    get_backbone_family,
    infer_patch_size,
    short_name,
    split_finetune_tag,
)
from .wrapper import PretrainedViTWrapper

__all__ = [
    "MODEL_LIST",
    "SHORT_NAMES",
    "PretrainedViTWrapper",
    "backbone_features",
    "create_backbone_model",
    "feature_dim",
    "get_backbone_family",
    "infer_patch_size",
    "load_backbone",
    "load_multiple_backbones",
    "short_name",
    "split_finetune_tag",
    "upsample_features",
]
