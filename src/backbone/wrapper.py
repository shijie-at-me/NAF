"""The frozen backbone the upsamplers read features from, whatever its family, with optional fine-tuned weights."""

import os

import torch
from torch import nn

from src.utils.checkpoint import FINETUNED_CKPT_DIR

from .families import FAMILIES, create_backbone_model
from .registry import get_backbone_family, infer_patch_size, split_finetune_tag, supports_prefix_tokens

__all__ = ["PretrainedViTWrapper", "finetuned_checkpoint_path", "load_finetuned_weights"]


def finetuned_checkpoint_path(tag: str, model_name: str, ckpt_dir: str = FINETUNED_CKPT_DIR) -> str:
    """``<ckpt_dir>/<tag>_<model name>.pth``, e.g. ``weights/finetuned/dvt_vit_base_patch14_dinov2.pth``."""
    return os.path.join(ckpt_dir, f"{tag}_{model_name}.pth")


def load_finetuned_weights(wrapper, ckpt_path: str, tag: str):
    """Load DVT / FiT3D fine-tuned weights into a ``PretrainedViTWrapper``.

    DVT checkpoints hold the whole wrapper under ``"model"``; FiT3D ones hold the bare backbone.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if tag == "dvt":
        wrapper.load_state_dict(ckpt["model"], strict=True)
    elif tag == "fit3d":
        wrapper.model.load_state_dict(ckpt, strict=True)
    else:
        raise ValueError(f"Unknown fine-tune tag {tag!r}")


class PretrainedViTWrapper(nn.Module):
    """A pretrained backbone (timm, RADIO, Franca or CAPI; ViT or not) that returns patch feature maps.

    ``name`` may carry a fine-tune tag (``dvt_...``, ``fit3d_...``): the fine-tuned weights are then loaded from
    ``ckpt_dir``. Exposes ``patch_size``, ``embed_dim`` and ``config`` (``mean``, ``std``, ``input_size``, ``ps``).
    ``layer`` is the block (1-based, timm backbones) whose output ``forward`` returns by default: None for the last one.
    """

    def __init__(
        self,
        name: str,
        norm: bool = True,
        layer: int | None = None,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        ckpt_dir: str = FINETUNED_CKPT_DIR,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.norm = norm
        self.layer = layer

        finetune_tag, model_name = split_finetune_tag(name)
        self.family = get_backbone_family(model_name)
        self.patch_size = infer_patch_size(model_name)

        self.model, self.config = create_backbone_model(
            model_name, self.patch_size, dynamic_img_size, dynamic_img_pad, **kwargs
        )
        self.config["ps"] = self.patch_size
        # Channels of the features: ``embed_dim`` for ViTs, ``num_features`` for timm models without it (ConvNeXt)
        self.embed_dim = getattr(self.model, "embed_dim", None) or self.model.num_features

        if finetune_tag is not None:
            load_finetuned_weights(self, finetuned_checkpoint_path(finetune_tag, model_name, ckpt_dir), finetune_tag)

    def forward(
        self,
        x: torch.Tensor,
        n: int | list[int] | tuple[int] | None = None,
        return_prefix_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the patch features of ``x`` as a ``(B, C, H / ps, W / ps)`` map.

        Args:
            x: Input image tensor.
            n: Block to take, as in ``forward_intermediates`` (timm backbones only); must select a single block.
                None takes ``layer`` (the last block if unset).
            return_prefix_tokens: Also return the prefix (cls / register) tokens (timm backbones only, not SAM).
        """
        if n is None:
            n = 1 if self.layer is None else [self.layer - 1]
        return FAMILIES[self.family].features(
            self.model,
            x,
            patch_size=self.patch_size,
            n=n,
            norm=self.norm,
            return_prefix_tokens=return_prefix_tokens and supports_prefix_tokens(self.name),
        )
