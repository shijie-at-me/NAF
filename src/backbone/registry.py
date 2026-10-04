"""What the code knows about a backbone from its name alone: family, fine-tune tag, patch size, feature dim, and
which timm options it supports.

Pure Python (no torch / timm import), so that config resolvers can use it at Hydra startup for free.
"""

import re

# Backbones tested with the upsamplers (timm names, or torch.hub ones); more are available in timm.
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

# Name prefixes marking fine-tuned weights to load on top of the pretrained backbone, e.g. "dvt_vit_base_patch14_dinov2"
FINETUNE_TAGS = ("dvt", "fit3d")

# Backbones loaded from torch.hub; everything else goes through timm
HUB_FAMILIES = ("radio", "franca", "capi")

# (name patterns, feature dim), checked in order
FEATURE_DIMS = (
    (("7b",), 4096),
    (("vits", "small"), 384),
    (("vitb", "base", "radio_v2.5-b"), 768),
    (("vitl", "large", "radio_v2.5-l"), 1024),
    (("so400m",), 1152),
    (("vith", "huge"), 1280),
    (("vitg", "giant"), 1536),
    (("tiny",), 192),
)


def split_finetune_tag(name: str) -> tuple[str | None, str]:
    """Split a fine-tune tag prefix off a backbone name: ``"dvt_vit_x"`` -> ``("dvt", "vit_x")``."""
    for tag in FINETUNE_TAGS:
        if name.startswith(f"{tag}_"):
            return tag, name.removeprefix(f"{tag}_")
    return None, name


def get_backbone_family(name: str) -> str:
    """The torch.hub family of a backbone name, or ``"timm"`` for timm models."""
    return next((family for family in HUB_FAMILIES if family in name), "timm")


def infer_patch_size(name: str) -> int:
    """Patch size from the model name (``patch14`` -> 14), with overrides for models that don't encode it."""
    if "convnext" in name:
        return 32
    if "franca" in name or "capi" in name:
        return 14
    match = re.search(r"patch(\d+)", name)
    return int(match.group(1)) if match else 16


def feature_dim(name: str) -> int:
    """Feature dimension of a backbone, from the first ``FEATURE_DIMS`` pattern found in its (lower-cased) name.

    Raises for a name that matches none: a model built with a made-up dimension would only fail much later.
    """
    lowered = name.lower()
    for patterns, dim in FEATURE_DIMS:
        if any(pattern in lowered for pattern in patterns):
            return dim
    known = ", ".join(f"{'/'.join(patterns)} -> {dim}" for patterns, dim in FEATURE_DIMS)
    raise ValueError(
        f"No known feature dimension for backbone {name!r} (name patterns: {known}). "
        "Add its pattern to FEATURE_DIMS in src/backbone/registry.py, or set the model's feature dim explicitly."
    )


def supports_dynamic_img_size(name: str) -> bool:
    """Whether timm's ``dynamic_img_size`` / ``dynamic_img_pad`` options apply (not to SAM or ConvNeXt models)."""
    return "sam" not in name and "convnext" not in name


def supports_prefix_tokens(name: str) -> bool:
    """Whether ``forward_intermediates`` can return the prefix (cls / register) tokens (not for SAM models)."""
    return "sam" not in name
