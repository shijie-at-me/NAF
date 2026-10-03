"""PixelUp checkpoints: release assets, the architecture they record, and their Semantic Encoder weights.

Port of https://github.com/deepankkumar/PixelUp (MIT License, v0.1.0). The model is in ``src/model/pixelup.py``.
"""

import os

import torch

from src.backbone.convnext import DEFAULT_SEMANTIC_ENCODER, SemanticEncoder
from utils.checkpoint import resolve_checkpoint

RELEASE_URL = "https://github.com/deepankkumar/PixelUp/releases/download/v0.1.0/{}.pth"

# Architecture of the released checkpoints; a checkpoint's embedded arch and its parameters override these
DEFAULT_ARCH = {
    "d_enc": 384,
    "num_heads_encoder": 6,
    "num_heads_decoder": 6,
    "rope_directions": 16,
    "chain_q_rope": False,
    "use_smooth": True,
    "fusion": "cross_only",
    "skip_attention_stages": "s0",
    "up_method": "nnconv",
    "up_scale": 2,
    "pixel_encoder_stride": 16,
    "q_norm_stages": False,
    "stage_pool_factors": "",
    "chain_global_attn": False,
}
# Older names of the arch fields in embedded checkpoint configs
_LEGACY_ARCH_FIELDS = {
    "num_heads_chain": "num_heads_encoder",
    "num_heads_down": "num_heads_decoder",
    "pixel_enc_scale": "pixel_encoder_stride",
    "drop_cross_stages": "skip_attention_stages",
    "fuse_mode": "fusion",
    "chain_scale": "semantic_scale",
    "guide_scale": "semantic_scale",
    "v_dim": "value_dim",
}


def load_checkpoint(checkpoint: str):
    """Load a PixelUp checkpoint given as a path, a URL, or the name of a release asset (e.g. "pixelup_convnext_s").

    URLs are downloaded once to the torch hub cache. The checkpoints hold their architecture next to the weights.
    """
    if not checkpoint.endswith(".pth") and not os.path.exists(checkpoint):
        checkpoint = RELEASE_URL.format(checkpoint)
    return torch.load(resolve_checkpoint(checkpoint), map_location="cpu", weights_only=False)


def read_embedded_arch(blob) -> dict:
    """The architecture recorded in a checkpoint (``arch``, or the ``arch`` of a config section), new field names."""
    if not isinstance(blob, dict):
        return {}
    arch = dict(blob.get("arch") or {})
    if not arch:
        for section in (blob.get("config") or {}).values():
            block = section.get("arch") if isinstance(section, dict) else None
            if block:
                arch = dict(block)
                break
    return {_LEGACY_ARCH_FIELDS.get(k, k): v for k, v in arch.items()}


def split_checkpoint(blob):
    """``(state dict, embedded arch)`` of a loaded checkpoint."""
    sd = (blob.get("state_dict") or blob.get("model") or blob) if isinstance(blob, dict) else blob
    return sd, read_embedded_arch(blob)


def detect_arch(sd) -> dict:
    """Arch fields that can be read off the parameters of a state dict."""
    keys = set(sd)
    det = {}
    if "pixel_enc.expand.weight" in sd:
        det["d_enc"] = int(sd["pixel_enc.expand.weight"].shape[0])
    has_fuse = any(k.startswith("fuse_s") for k in keys)
    has_cross = any(k.startswith("cross_s") for k in keys)
    det["fusion"] = "both" if (has_fuse and has_cross) else ("fuse_only" if has_fuse else "cross_only")
    det["skip_attention_stages"] = ",".join(
        s for s in ("s0", "s1", "s2", "s3") if has_cross and not any(k.startswith(f"cross_{s}.") for k in keys)
    )
    if any(k.startswith("up_s3.conv.") for k in keys):
        det["up_method"] = "nnconv"
    elif any(k.startswith("up_s3.compress.") for k in keys):
        det["up_method"] = "carafe"
    elif "up_s3.0.weight" in keys:
        det["up_method"] = "pixshuffle"
    det["chain_q_rope"] = any(k.startswith("q_rope.") for k in keys)
    det["use_smooth"] = "smooth.weight" in keys
    det["q_norm_stages"] = "q_norm.weight" in keys
    return det


def resolve_arch(sd, embedded: dict, overrides: dict) -> dict:
    """Defaults, then the checkpoint's embedded arch, then what its parameters imply, then explicit overrides."""
    arch = dict(DEFAULT_ARCH)
    arch.update({k: embedded[k] for k in DEFAULT_ARCH if embedded.get(k) is not None})
    if sd is not None:
        arch.update(detect_arch(sd))
    arch.update({k: v for k, v in overrides.items() if k in DEFAULT_ARCH})
    return arch


def encoder_state_dict(sd) -> dict:
    """The Semantic Encoder weights bundled in a PixelUp state dict (timm names), or {}."""
    if sd is None:
        return {}
    for prefix in ("semantic_encoder.model.", "convnext.model."):
        found = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        if found:
            return found
    return {}


def build_semantic_encoder(sd, embedded, spec, encoder_dir) -> SemanticEncoder:
    """The Semantic Encoder named by ``spec`` (an arch name or ``{"arch", "source"}``) or recorded in the checkpoint,
    with the checkpoint's weights unless ``source`` is "bundled", else those of ``encoder_dir``."""
    arch, source = None, "checkpoint"
    if isinstance(spec, str):
        arch, source = spec, "explicit"
    elif spec is not None:
        arch = spec.get("arch")
        source = spec.get("source") or "checkpoint"

    if arch is None:
        recorded = embedded.get("semantic_encoder")
        if isinstance(recorded, dict):
            arch = recorded.get("arch")
        arch = arch or embedded.get("semantic_encoder_arch") or DEFAULT_SEMANTIC_ENCODER

    weights = encoder_state_dict(sd) if source != "bundled" else {}
    if weights:
        return SemanticEncoder(arch, state_dict=weights)
    return SemanticEncoder(arch, hf_dir=encoder_dir)
