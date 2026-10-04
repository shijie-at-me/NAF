"""Frozen backbones (the vision foundation models whose features get upsampled).

- ``registry``: what a backbone name tells (family, fine-tune tag, patch size, feature dim); no torch import
- ``families``: per family (timm, RADIO, Franca, CAPI), how to build the model and read its patch features
- ``wrapper``: ``PretrainedViTWrapper``, the backbone module the rest of the code uses, and fine-tuned weights
- ``loading``: frozen backbones from configs
- ``convnext``: PixelUp's Semantic Encoder (a frozen DINOv3 ConvNeXt pyramid)
"""

from .families import create_backbone_model
from .loading import load_backbone, load_multiple_backbones
from .registry import MODEL_LIST, feature_dim, get_backbone_family, infer_patch_size, split_finetune_tag
from .wrapper import PretrainedViTWrapper

__all__ = [
    "MODEL_LIST",
    "PretrainedViTWrapper",
    "create_backbone_model",
    "feature_dim",
    "get_backbone_family",
    "infer_patch_size",
    "load_backbone",
    "load_multiple_backbones",
    "split_finetune_tag",
]
