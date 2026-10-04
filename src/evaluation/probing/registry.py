"""The probe registry: trained probes kept with the weights of their upsampler, as
``<WEIGHTS_DIR>/<model>/probes/<task>/<dataset>/<backbone>[@<checkpoint>].pth`` (the upsampler id is
``<model>[@<checkpoint>]``), with what an analysis needs to reuse them (``find_probe`` + ``load_probe``).

Probes of the older layout, ``<WEIGHTS_DIR>/probes/<task>/<dataset>/<backbone>/<upsampler>.pth``, are still found.
"""

import datetime
import glob
import os
from pathlib import Path
from urllib.parse import urlparse

import torch
from omegaconf import OmegaConf

from src.utils.checkpoint import WEIGHTS_DIR, load_checkpoint, model_weights_dir
from src.utils.run import launch_path

__all__ = [
    "find_probe",
    "load_probe",
    "probe_meta",
    "probe_path",
    "register_probe",
    "registry_path",
    "save_probe",
    "upsampler_id",
]


# ---- probe files ---------------------------------------------------------------------------------------------------


def upsampler_id(model_cfg, model_ckpt=None):
    """Name of an upsampler and its weights: ``<model name>`` or ``<model name>@<checkpoint file stem>``.

    The checkpoint is ``model_ckpt`` (loaded into the model) or, for models that load their own weights, the
    ``checkpoint`` entry of their config (e.g. ``pixelup@pixelup_convnext_s``).
    """
    name = model_cfg.get("name", "model")
    ckpt = model_ckpt or model_cfg.get("checkpoint")
    return f"{name}@{Path(urlparse(str(ckpt)).path).stem}" if ckpt else name


def probe_path(task, dataset, backbone, upsampler, root=WEIGHTS_DIR):
    """Where the probe of ``upsampler`` (``<model>[@<checkpoint>]``) on ``backbone`` features is saved for ``task`` on
    ``dataset``: ``<root>/<model>/probes/<task>/<dataset>/<backbone>[@<checkpoint>].pth``."""
    model, at, checkpoint = upsampler.partition("@")
    return os.path.join(model_weights_dir(model, root), "probes", task, dataset, f"{backbone}{at}{checkpoint}.pth")


def save_probe(classifier, path, meta):
    """Save a probe with what is needed to reuse it: the upsampler and dataset configs, metrics, settings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": classifier.state_dict(), "meta": meta}, path)
    return path


def find_probe(task, dataset, backbone, upsampler, root=WEIGHTS_DIR):
    """Path of a saved probe; ``upsampler`` may leave out the ``@<checkpoint>`` suffix when a single one matches.

    Looks in ``probe_path``'s folder, then in the older layout (``<root>/probes/...``).
    """
    model = upsampler.partition("@")[0]
    new_dir = os.path.dirname(probe_path(task, dataset, backbone, model, root))
    old_dir = os.path.join(root, "probes", task, dataset, backbone)
    # (folder, file stem of the probe of ``upsampler`` there) in both layouts
    layouts = [(new_dir, os.path.basename(probe_path(task, dataset, backbone, upsampler, root))[: -len(".pth")])]
    layouts.append((old_dir, upsampler))
    for directory, stem in layouts:
        exact = os.path.join(directory, f"{stem}.pth")
        if os.path.exists(exact):
            return exact
        matches = sorted(glob.glob(os.path.join(glob.escape(directory), f"{glob.escape(stem)}@*.pth")))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise FileNotFoundError(
                f"Probe {upsampler!r} is ambiguous in {directory}: {[Path(p).stem for p in matches]}"
            )
    available = sorted(
        Path(p).stem for p in glob.glob(os.path.join(glob.escape(new_dir), f"{glob.escape(backbone)}*.pth"))
    )
    raise FileNotFoundError(f"Probe {upsampler!r} not found in {new_dir} (nor in {old_dir}); available: {available}")


def load_probe(path, device="cpu"):
    """``(state dict, meta)`` of a saved probe; a bare state dict (older runs) comes with empty meta."""
    blob = load_checkpoint(path, device)
    if "state_dict" in blob and "meta" in blob:
        return blob["state_dict"], blob["meta"]
    return blob, {}


# ---- registering the probe of a run --------------------------------------------------------------------------------


def register_probe(cfg, task, classifier, model_ckpt, metrics, run_dir):
    """Save a trained probe and its meta to the registry; returns its path."""
    return save_probe(
        classifier, registry_path(cfg, task, model_ckpt), probe_meta(cfg, task, model_ckpt, metrics, run_dir)
    )


def registry_path(cfg, task, model_ckpt):
    """Where the probe of this run is kept for later analyses (``probe_path``; root ``eval.probe_dir`` if set)."""
    root = launch_path(cfg.eval.get("probe_dir"))
    kwargs = {"root": root} if root else {}
    return probe_path(
        task, cfg.dataset.get("tag", "dataset"), cfg.backbone.name, upsampler_id(cfg.model, model_ckpt), **kwargs
    )


def probe_meta(cfg, task, model_ckpt, metrics, run_dir):
    """Everything an analysis needs to rebuild the setting of a probe: upsampler, weights, data, settings, results."""
    return {
        "task": task,
        "dataset": cfg.dataset.get("tag", "dataset"),
        "backbone": cfg.backbone.name,
        "upsampler": upsampler_id(cfg.model, model_ckpt),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "model_ckpt": model_ckpt,
        "dataset_cfg": OmegaConf.to_container(cfg.dataset, resolve=True),
        "img_size": cfg.img_size,
        "target_size": cfg.target_size,
        "probe": {
            "num_epochs": cfg.num_epochs,
            "lr": cfg.optimizer.lr,
            "batch_size": cfg.train_dataloader.batch_size,
            "seed": cfg.get("seed", 0),
            "use_bf16": cfg.eval.get("use_bf16", False),
        },
        "metrics": metrics,
        "run_dir": run_dir,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
