import os

import torch
from hydra.utils import instantiate


def build_model(model_cfg, device, ckpt_path=None, strict=True):
    """Instantiate a model from its config and optionally load a checkpoint into it."""
    model = instantiate(model_cfg).to(device)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=strict)
    return model


def save_checkpoint(model, ckpt_dir, step):
    """Save the model weights as ``model_<step>steps.pth`` and return the path."""
    path = os.path.join(ckpt_dir, f"model_{step}steps.pth")
    torch.save(model.state_dict(), path)
    return path
