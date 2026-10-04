"""Everything family-specific about a backbone: how to build it and how to read its patch features.

Families: ``timm`` (any timm model) and the torch.hub ones, RADIO, Franca and CAPI. Adding one means a ``Family``
entry here and its name pattern in ``registry.HUB_FAMILIES``.
"""

from collections.abc import Callable
from dataclasses import dataclass

import timm
import torch
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, resolve_model_data_config
from torch import nn

from .registry import HUB_FAMILIES, get_backbone_family, supports_dynamic_img_size

__all__ = ["FAMILIES", "Family", "create_backbone_model", "imagenet_data_config"]


def imagenet_data_config(img_size: int) -> dict:
    """Data config of a model fed ImageNet-normalized square images of side ``img_size``."""
    return {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD, "input_size": (3, img_size, img_size)}


# --- building: (name, patch_size, dynamic_img_size, dynamic_img_pad, **timm kwargs) -> (model, data config) ---------


def _load_timm(name, patch_size, dynamic_img_size, dynamic_img_pad, **kwargs):
    """Pretrained timm model without classifier head; ``kwargs`` override the ``timm.create_model`` options."""
    timm_kwargs = dict(pretrained=True, num_classes=0, patch_size=patch_size)
    if supports_dynamic_img_size(name):
        timm_kwargs["dynamic_img_size"] = dynamic_img_size
        timm_kwargs["dynamic_img_pad"] = dynamic_img_pad
    timm_kwargs.update(kwargs)
    model = timm.create_model(name, **timm_kwargs)
    return model, resolve_model_data_config(model=model)


def _load_radio(name, *unused_args, **unused_kwargs):
    model = torch.hub.load("NVlabs/RADIO", "radio_model", version=name, progress=True, skip_validation=True)
    # RADIO normalizes its inputs itself
    return model, {"mean": torch.zeros(3), "std": torch.ones(3), "input_size": (3, 512, 512)}


def _load_franca(name, *unused_args, **unused_kwargs):
    return torch.hub.load("valeoai/Franca", name, use_rasa_head=True), imagenet_data_config(448)


def _load_capi(name, *unused_args, **unused_kwargs):
    return torch.hub.load("facebookresearch/capi:main", name, force_reload=False), imagenet_data_config(448)


# --- features: (model, x, *, patch_size, n, norm, return_prefix_tokens) -> [B, C, H / patch, W / patch] ------------


def intermediate_features(model, x, *, n=1, norm=True, return_prefix_tokens=False, **unused):
    """timm (and RADIO) models: the output of one block through ``forward_intermediates``.

    ``n`` must select a single block (as in ``forward_intermediates``: the last ``n``, or a list of indices).
    With ``return_prefix_tokens``, returns ``(patch map, prefix tokens)``.
    """
    kwargs = dict(norm=norm, output_fmt="NCHW", intermediates_only=True)
    if return_prefix_tokens:
        kwargs["return_prefix_tokens"] = True

    out = model.forward_intermediates(x, n, **kwargs)
    if not isinstance(out, (list, tuple)):
        return out
    assert len(out) == 1, f"Expected features from a single block, got {len(out)} (n={n})."
    return out[0]


def franca_features(model, x, *, patch_size, **unused):
    """Franca: the patch tokens of its RASA head."""
    h, w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
    feats = model.forward_features(x, use_rasa_head=True)["patch_token_rasa"]
    return rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)


def capi_features(model, x, **unused):
    """CAPI: its last output, channels last."""
    *_, feats = model(x)
    return feats.permute(0, 3, 1, 2)


# --- families -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    """How to build the backbones of a family and read their patch features (signatures in the sections above)."""

    load: Callable
    features: Callable


FAMILIES = {
    "timm": Family(_load_timm, intermediate_features),
    "radio": Family(_load_radio, intermediate_features),
    "franca": Family(_load_franca, franca_features),
    "capi": Family(_load_capi, capi_features),
}
assert set(FAMILIES) == {"timm", *HUB_FAMILIES}, "every family named in the registry needs an entry here"


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
    family = FAMILIES[get_backbone_family(name)]
    model, data_config = family.load(name, patch_size, dynamic_img_size, dynamic_img_pad, **kwargs)
    return model.eval(), data_config
