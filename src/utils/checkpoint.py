"""Checkpoint files: given as a path or a URL (downloaded once to ``WEIGHTS_DIR``), loaded, saved, and models built
from their config with one."""

import os
from urllib.parse import urlparse

import torch
from hydra.utils import instantiate

from src.utils.paths import WEIGHTS_DIR

__all__ = ["build_model", "load_checkpoint", "resolve_checkpoint", "save_checkpoint"]


def resolve_checkpoint(ckpt_path):
    """Local path of a checkpoint given as a path (``~`` expanded) or an http(s) URL; URLs are downloaded once to
    ``WEIGHTS_DIR`` under their file name."""
    if not ckpt_path.startswith(("http://", "https://")):
        return os.path.expanduser(ckpt_path)
    path = os.path.join(WEIGHTS_DIR, os.path.basename(urlparse(ckpt_path).path))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(ckpt_path, path)
    return path


def load_checkpoint(ckpt_path, device="cpu", weights_only=True):
    """Contents of a checkpoint given as a path or URL (``resolve_checkpoint``), mapped to ``device``.

    ``weights_only=False`` also unpickles non-tensor objects (e.g. the original FeatUp Lightning checkpoints);
    only use it with trusted files.
    """
    return torch.load(resolve_checkpoint(ckpt_path), map_location=device, weights_only=weights_only)


def build_model(model_cfg, device, ckpt_path=None, strict=True, weights_only=True):
    """Instantiate a model from its config and optionally load a checkpoint (path or URL) into it."""
    model = instantiate(model_cfg).to(device)
    if ckpt_path:
        model.load_state_dict(load_checkpoint(ckpt_path, device, weights_only), strict=strict)
    return model


def save_checkpoint(model, ckpt_dir, step):
    """Save the model weights as ``model_<step>steps.pth`` and return the path."""
    path = os.path.join(ckpt_dir, f"model_{step}steps.pth")
    torch.save(model.state_dict(), path)
    return path
