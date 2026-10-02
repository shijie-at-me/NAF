import re

import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, resolve_model_data_config
from torch import nn

# We provide a list of timm model names, more are available on their official repo.
MODEL_LIST = [
    # DINO
    "vit_base_patch16_224.dino",
    # DINOv2
    "vit_base_patch14_dinov2.lvd142m",
    # DINOv2-R
    "vit_base_patch14_reg4_dinov2",
    # Franca
    "franca_vitb14",
    # DINOv3-ViT
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_large_patch16_dinov3.lvd1689m",
    "vit_7b_patch16_dinov3.lvd1689m",
    # SigLIP2
    "vit_base_patch16_siglip_512.v2_webli",
    # PE Core
    "vit_pe_core_small_patch16_384.fb",
    # PE Spatial
    "vit_pe_spatial_tiny_patch16_512.fb",
    # RADIO
    "radio_v2.5-b",
    # CAPI
    "capi_vitl14_lvd",
    # MAE
    "vit_large_patch16_224.mae",
]

# Name prefixes marking fine-tuned weights to load on top of the pretrained backbone, e.g. "dvt_vit_base_patch14_dinov2".
FINETUNE_TAGS = ("dvt", "fit3d")

# Backbones loaded from torch.hub; everything else goes through timm.
HUB_FAMILIES = ("radio", "franca", "capi")


def split_finetune_tag(name: str) -> tuple[str | None, str]:
    """Split a fine-tune tag prefix off a backbone name: ``"dvt_vit_x"`` -> ``("dvt", "vit_x")``."""
    for tag in FINETUNE_TAGS:
        if name.startswith(f"{tag}_"):
            return tag, name.removeprefix(f"{tag}_")
    return None, name


def get_backbone_family(name: str) -> str:
    """Return the torch.hub family of a backbone name, or ``"timm"`` for timm models."""
    return next((family for family in HUB_FAMILIES if family in name), "timm")


def infer_patch_size(name: str) -> int:
    """Patch size from the model name (``patch14`` -> 14), with overrides for models that don't encode it."""
    if "convnext" in name:
        return 32
    if "franca" in name or "capi" in name:
        return 14
    match = re.search(r"patch(\d+)", name)
    return int(match.group(1)) if match else 16


def _imagenet_data_config(img_size: int) -> dict:
    return {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD, "input_size": (3, img_size, img_size)}


def _create_timm_model(name: str, patch_size: int, dynamic_img_size: bool, dynamic_img_pad: bool, **kwargs):
    timm_kwargs = dict(pretrained=True, num_classes=0, patch_size=patch_size)
    if "sam" not in name and "convnext" not in name:
        timm_kwargs["dynamic_img_size"] = dynamic_img_size
        timm_kwargs["dynamic_img_pad"] = dynamic_img_pad
    timm_kwargs.update(kwargs)
    return timm.create_model(name, **timm_kwargs)


def create_backbone_model(
    name: str,
    patch_size: int,
    dynamic_img_size: bool = True,
    dynamic_img_pad: bool = False,
    **kwargs,
) -> tuple[nn.Module, dict]:
    """Build a pretrained backbone in eval mode and its data config (``mean``, ``std``, ``input_size``).

    ``kwargs`` are forwarded to ``timm.create_model`` and ignored for torch.hub backbones.
    """
    family = get_backbone_family(name)
    if family == "radio":
        model = torch.hub.load("NVlabs/RADIO", "radio_model", version=name, progress=True, skip_validation=True)
        data_config = {"mean": torch.zeros(3), "std": torch.ones(3), "input_size": (3, 512, 512)}
    elif family == "franca":
        model = torch.hub.load("valeoai/Franca", name, use_rasa_head=True)
        data_config = _imagenet_data_config(448)
    elif family == "capi":
        model = torch.hub.load("facebookresearch/capi:main", name, force_reload=False)
        data_config = _imagenet_data_config(448)
    else:
        model = _create_timm_model(name, patch_size, dynamic_img_size, dynamic_img_pad, **kwargs)
        data_config = resolve_model_data_config(model=model)
    return model.eval(), data_config
