"""Registering trained probes (weights/probes/<task>/<dataset>/<backbone>/<upsampler>.pth) with what an analysis
needs to reuse them."""

import datetime

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from src.utils.checkpoint import probe_path, save_probe, upsampler_id

__all__ = ["probe_meta", "register_probe", "registry_path"]


def register_probe(cfg, task, classifier, model_ckpt, metrics, run_dir):
    """Save a trained probe and its meta to the registry; returns its path."""
    return save_probe(
        classifier, registry_path(cfg, task, model_ckpt), probe_meta(cfg, task, model_ckpt, metrics, run_dir)
    )


def registry_path(cfg, task, model_ckpt):
    """Where the probe of this run is kept for later analyses (``src.utils.checkpoint.probe_path``)."""
    root = cfg.eval.get("probe_dir")
    kwargs = {"root": to_absolute_path(root)} if root else {}
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
