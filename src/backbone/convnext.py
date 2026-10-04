"""Frozen DINOv3 ConvNeXt feature pyramid, as used by PixelUp's Semantic Encoder (ported from
https://github.com/deepankkumar/PixelUp, MIT License)."""

import re
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["SemanticEncoder", "remap_convnext_hf_to_timm", "SEMANTIC_ENCODERS", "DEFAULT_SEMANTIC_ENCODER"]

# timm name -> channels of its four stages
SEMANTIC_ENCODERS = {
    "convnext_small.dinov3_lvd1689m": (96, 192, 384, 768),
    "convnext_tiny.dinov3_lvd1689m": (96, 192, 384, 768),
}
DEFAULT_SEMANTIC_ENCODER = "convnext_small.dinov3_lvd1689m"

# Hugging Face ConvNeXt parameter names -> timm's
_HF_TO_TIMM = (
    (".layers.", ".blocks."),
    (".depthwise_conv.", ".conv_dw."),
    (".layer_norm.", ".norm."),
    (".pointwise_conv1.", ".mlp.fc1."),
    (".pointwise_conv2.", ".mlp.fc2."),
)


def remap_convnext_hf_to_timm(hf_sd: dict) -> dict:
    """Rename a Hugging Face ``ConvNextModel`` state dict for timm's ``features_only`` ConvNeXt (final norm dropped)."""

    def renumber(key):
        return re.sub(r"^stages\.(\d+)\.", r"stages_\1.", key)

    out = {}
    for k, v in hf_sd.items():
        if k.startswith("stages.0.downsample_layers."):
            out["stem_" + k[len("stages.0.downsample_layers.") :]] = v
        elif ".downsample_layers." in k:
            out[renumber(k.replace(".downsample_layers.", ".downsample."))] = v
        elif ".layers." in k:
            for old, new in _HF_TO_TIMM:
                k = k.replace(old, new)
            out[renumber(k)] = v
        elif not k.startswith("layer_norm."):
            out[renumber(k)] = v
    return out


class SemanticEncoder(nn.Module):
    """Frozen DINOv3 ConvNeXt returning its four stage feature maps (strides 4, 8, 16, 32).

    The weights come from ``state_dict`` (timm names) or, without it, from the ``model.safetensors`` of a local
    Hugging Face snapshot in ``hf_dir``.
    """

    def __init__(self, arch: str = DEFAULT_SEMANTIC_ENCODER, state_dict: dict | None = None, hf_dir: str | None = None):
        super().__init__()
        import timm

        self.arch = arch
        self.model = timm.create_model(arch, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        self.model = self.model.to(memory_format=torch.channels_last)
        if state_dict is None:
            if hf_dir is None:
                raise RuntimeError(
                    f"No weights for the Semantic Encoder {arch!r}: use a checkpoint that carries them or pass the "
                    "local Hugging Face snapshot of facebook/dinov3-convnext-small-pretrain-lvd1689m as hf_dir"
                )
            from safetensors.torch import load_file

            state_dict = remap_convnext_hf_to_timm(load_file(str(Path(hf_dir) / "model.safetensors")))

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("head.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Semantic Encoder {arch!r} weight mismatch: missing={missing[:4]} unexpected={list(unexpected)[:4]}"
            )
        self.model.requires_grad_(False).eval()
        self.OUT_CH = tuple(self.model.feature_info.channels())

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, image):
        return tuple(self.model(image.contiguous(memory_format=torch.channels_last)))
