import os
from pathlib import Path
from urllib.parse import urlparse

import torch
from hydra.utils import instantiate

# Where downloaded checkpoints are kept: <repo>/weights (git-ignored), or $NAF_WEIGHTS_DIR
WEIGHTS_DIR = os.environ.get("NAF_WEIGHTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights"))
NAF_RELEASE_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"


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


def build_model(model_cfg, device, ckpt_path=None, strict=True, weights_only=True):
    """Instantiate a model from its config and optionally load a checkpoint (path or URL) into it.

    ``weights_only=False`` also unpickles non-tensor objects (e.g. the original FeatUp Lightning checkpoints);
    only use it with trusted files.
    """
    model = instantiate(model_cfg).to(device)
    if ckpt_path:
        model.load_state_dict(
            torch.load(resolve_checkpoint(ckpt_path), map_location=device, weights_only=weights_only), strict=strict
        )
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
