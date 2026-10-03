import os
from pathlib import Path

import torch
from hydra.utils import instantiate


def build_model(model_cfg, device, ckpt_path=None, strict=True, weights_only=True):
    """Instantiate a model from its config and optionally load a checkpoint into it.

    ``weights_only=False`` also unpickles non-tensor objects (e.g. the original FeatUp Lightning checkpoints);
    only use it with trusted files.
    """
    model = instantiate(model_cfg).to(device)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=weights_only), strict=strict)
    return model


def save_checkpoint(model, ckpt_dir, step):
    """Save the model weights as ``model_<step>steps.pth`` and return the path."""
    path = os.path.join(ckpt_dir, f"model_{step}steps.pth")
    torch.save(model.state_dict(), path)
    return path


def checkpoint_run_name(ckpt_path):
    """Experiment name of a checkpoint saved by a training run (``output/<exp>/<run>/model.pth`` -> ``<exp>``).

    Empty without a checkpoint or when the path is too short to have one.
    """
    parts = Path(ckpt_path).parts if ckpt_path else ()
    return parts[-3] if len(parts) >= 3 else ""
