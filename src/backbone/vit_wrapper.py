import os

import torch
from einops import rearrange
from torch import nn

from .factory import create_backbone_model, get_backbone_family, infer_patch_size, split_finetune_tag

# Where the dvt_ / fit3d_ fine-tuned checkpoints live, as ``<tag>_<model name>.pth``.
FINETUNED_CKPT_DIR = "/home/lchambon/workspace/JAFAR/ckpts"


class PretrainedViTWrapper(nn.Module):
    def __init__(
        self,
        name: str,
        norm: bool = True,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        ckpt_dir: str = FINETUNED_CKPT_DIR,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.norm = norm

        finetune_tag, model_name = split_finetune_tag(name)
        self.family = get_backbone_family(model_name)
        self.patch_size = infer_patch_size(model_name)

        self.model, self.config = create_backbone_model(
            model_name, self.patch_size, dynamic_img_size, dynamic_img_pad, **kwargs
        )
        self.config["ps"] = self.patch_size
        self.embed_dim = self.model.embed_dim

        if finetune_tag is not None:
            self.load_finetuned_weights(os.path.join(ckpt_dir, f"{finetune_tag}_{model_name}.pth"), finetune_tag)

    def load_finetuned_weights(self, ckpt_path: str, tag: str):
        """DVT checkpoints hold the whole wrapper under ``"model"``; FiT3D ones hold the bare backbone."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if tag == "dvt":
            self.load_state_dict(ckpt["model"], strict=True)
        elif tag == "fit3d":
            self.model.load_state_dict(ckpt, strict=True)

    def forward(
        self,
        x: torch.Tensor,
        n: int | list[int] | tuple[int] = 1,
        return_prefix_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the patch features of ``x`` as a ``(B, C, H / ps, W / ps)`` map.

        Args:
            x: Input image tensor.
            n: Block to take, as in ``forward_intermediates`` (timm backbones only); must select a single block.
            return_prefix_tokens: Also return the prefix (cls / register) tokens (timm backbones only).
        """
        if self.family == "franca":
            return self._forward_franca(x)
        if self.family == "capi":
            return self._forward_capi(x)
        return self._forward_intermediates(x, n, return_prefix_tokens)

    def _forward_franca(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        feats = self.model.forward_features(x, use_rasa_head=True)["patch_token_rasa"]
        return rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)

    def _forward_capi(self, x: torch.Tensor) -> torch.Tensor:
        *_, feats = self.model(x)
        return feats.permute(0, 3, 1, 2)

    def _forward_intermediates(self, x: torch.Tensor, n, return_prefix_tokens: bool):
        kwargs = dict(norm=self.norm, output_fmt="NCHW", intermediates_only=True)
        # SAM models don't support prefix tokens
        if return_prefix_tokens and "sam" not in self.name:
            kwargs["return_prefix_tokens"] = True

        out = self.model.forward_intermediates(x, n, **kwargs)
        if not isinstance(out, (list, tuple)):
            return out
        assert len(out) == 1, f"Expected features from a single block, got {len(out)} (n={n})."
        return out[0]
