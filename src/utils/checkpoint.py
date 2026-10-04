"""Weights: where they live, checkpoint files given as a path or a URL (downloaded once to ``WEIGHTS_DIR``), loaded and
saved, and models built from their config with one.

``WEIGHTS_DIR`` is organized by model: ``<model>/`` holds the checkpoints of a model (``naf/naf_release.pth``) and its
trained probes (``<model>/probes/``, see ``src.evaluation.probing.registry``); ``backbone/`` holds the fine-tuned
backbones. Files of the older flat layout (``WEIGHTS_DIR/<file>``, ``finetuned/``) are still found.
"""

import os
from urllib.parse import urlparse

import torch
from hydra.utils import instantiate

__all__ = [
    "ROOT",
    "WEIGHTS_DIR",
    "FINETUNED_CKPT_DIR",
    "model_weights_dir",
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


def _finetuned_dir():
    """``WEIGHTS_DIR/backbone``, or ``WEIGHTS_DIR/finetuned`` of the older layout while only it exists."""
    current, legacy = os.path.join(WEIGHTS_DIR, "backbone"), os.path.join(WEIGHTS_DIR, "finetuned")
    return legacy if os.path.isdir(legacy) and not os.path.isdir(current) else current


# The dvt_ / fit3d_ fine-tuned backbones, as <tag>_<model name>.pth: _finetuned_dir(), or $NAF_FINETUNED_CKPT_DIR
FINETUNED_CKPT_DIR = os.environ.get("NAF_FINETUNED_CKPT_DIR", _finetuned_dir())


def model_weights_dir(model, root=WEIGHTS_DIR):
    """Folder of the weights of ``model`` (its config ``name``, e.g. "naf"): ``<root>/<model>``."""
    return os.path.join(root, model)


def resolve_checkpoint(ckpt_path, model=None):
    """Local path of a checkpoint given as a path (``~`` expanded) or an http(s) URL.

    URLs are downloaded once to ``model_weights_dir(model)`` (``WEIGHTS_DIR`` without ``model``) under their file name.
    A file of the older flat layout, ``WEIGHTS_DIR/<file>``, is found in ``model_weights_dir(model)`` once moved there,
    and an older download still in ``WEIGHTS_DIR`` is reused.
    """
    is_url = ckpt_path.startswith(("http://", "https://"))
    if is_url:
        path = os.path.join(WEIGHTS_DIR, os.path.basename(urlparse(ckpt_path).path))
    else:
        path = os.path.expanduser(ckpt_path)
    # Downloads and files of the flat layout belong in the model's folder
    if model is not None and os.path.dirname(os.path.abspath(path)) == os.path.abspath(WEIGHTS_DIR):
        in_model_dir = os.path.join(model_weights_dir(model), os.path.basename(path))
        if os.path.exists(in_model_dir) or (is_url and not os.path.exists(path)):
            path = in_model_dir
    if is_url and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(ckpt_path, path)
    return path


def load_checkpoint(ckpt_path, device="cpu", weights_only=True, model=None):
    """Contents of a checkpoint given as a path or URL (``resolve_checkpoint``, kept with the weights of ``model``),
    mapped to ``device``.

    ``weights_only=False`` also unpickles non-tensor objects (e.g. the original FeatUp Lightning checkpoints);
    only use it with trusted files.
    """
    return torch.load(resolve_checkpoint(ckpt_path, model), map_location=device, weights_only=weights_only)


def build_model(model_cfg, device, ckpt_path=None, strict=True, weights_only=True):
    """Instantiate a model from its config and optionally load a checkpoint (path or URL, downloaded to the folder of
    the model's ``name``) into it."""
    model = instantiate(model_cfg).to(device)
    if ckpt_path:
        state = load_checkpoint(ckpt_path, device, weights_only, model=model_cfg.get("name"))
        model.load_state_dict(state, strict=strict)
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
