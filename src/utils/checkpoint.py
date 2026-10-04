"""Weights: where they live, checkpoint files given as a path or a URL (downloaded once to ``WEIGHTS_DIR``), loaded and
saved, and models built from their config with one."""

import os
from urllib.parse import urlparse

import torch
from hydra.utils import instantiate

__all__ = [
    "ROOT",
    "WEIGHTS_DIR",
    "FINETUNED_CKPT_DIR",
    "load_upsampler",
    "build_model",
    "load_checkpoint",
    "resolve_checkpoint",
    "save_checkpoint",
]


# Repository root (this file is <root>/src/utils/checkpoint.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Downloaded and trained checkpoints: <root>/weights (git-ignored), or $NAF_WEIGHTS_DIR
WEIGHTS_DIR = os.environ.get("NAF_WEIGHTS_DIR", os.path.join(ROOT, "weights"))
# The dvt_ / fit3d_ fine-tuned backbones, as <tag>_<model name>.pth: <WEIGHTS_DIR>/finetuned, or $NAF_FINETUNED_CKPT_DIR
FINETUNED_CKPT_DIR = os.environ.get("NAF_FINETUNED_CKPT_DIR", os.path.join(WEIGHTS_DIR, "finetuned"))


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


def load_upsampler(model_cfg, ckpt_path, device, console=None, trainable=False):
    """The upsampler of ``model_cfg`` with the weights of ``ckpt_path`` (path or URL; None: as built), in eval mode and
    frozen unless ``trainable``; says on ``console`` (if given) which weights it has."""
    model = build_model(model_cfg, device, ckpt_path, weights_only=False)
    if console is not None:
        if ckpt_path:
            console.print(f"[green]Loaded model from checkpoint: {ckpt_path}[/green]")
        else:
            console.print("[yellow]No model checkpoint provided, using the model as built[/yellow]")
    return model.requires_grad_(trainable).eval()
