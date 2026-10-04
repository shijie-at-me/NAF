"""Datasets and data loaders from their configs (``cfg.dataset``, ``cfg.train_dataloader``, ``cfg.val_dataloader``)."""

import random

import numpy as np
import torch
from hydra.utils import instantiate

from .transforms import build_transforms

__all__ = ["build_dataloader", "build_dataset", "get_dataloaders", "seed_worker"]

# Old dataset modules -> their current ones (most specific first); configs saved with older probes still name them
LEGACY_DATASET_MODULES = {
    "evaluation.dataset.image_dataset.": "src.dataset.imagenet.",
    "evaluation.dataset.": "src.dataset.",
}


def seed_worker(worker_id=None):
    """``worker_init_fn`` of the loaders: seed NumPy and ``random`` in each worker from its torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def upgrade_dataset_target(dataset_cfg):
    """Copy of a dataset config whose ``_target_`` names its current module (see ``LEGACY_DATASET_MODULES``)."""
    dataset_cfg = dataset_cfg.copy()
    target = dataset_cfg.get("_target_", "")
    for old, new in LEGACY_DATASET_MODULES.items():
        if target.startswith(old):
            dataset_cfg["_target_"] = new + target.removeprefix(old)
            break
    return dataset_cfg


def build_dataset(dataset_cfg, transforms, split=None):
    """Instantiate a dataset, switched to ``split`` when given and the dataset has a split."""
    dataset_cfg = upgrade_dataset_target(dataset_cfg)
    if split is not None and "split" in dataset_cfg:
        dataset_cfg.split = split
    return instantiate(dataset_cfg, transform=transforms["image"], target_transform=transforms["label"])


def build_dataloader(dataloader_cfg, dataset, shuffle):
    """Instantiate a dataloader; deterministic order when ``shuffle`` is False, true randomness otherwise."""
    dataloader_cfg = dataloader_cfg.copy()
    extra = {}
    # Keep the workers alive across epochs instead of re-spawning them (and re-opening the dataset) every epoch
    if dataloader_cfg.get("num_workers", 0) > 0 and "persistent_workers" not in dataloader_cfg:
        extra["persistent_workers"] = True
    if shuffle:
        generator = None
        if "worker_init_fn" in dataloader_cfg:
            dataloader_cfg["worker_init_fn"] = None
    else:
        generator = torch.Generator()
        generator.manual_seed(0)

    return instantiate(dataloader_cfg, dataset=dataset, generator=generator, **extra)


def get_dataloaders(cfg, shuffle=True, val=True):
    """Build the train and val dataloaders described by ``cfg``; the val one is None unless ``val``.

    Skipping the val split avoids listing (or, for Hub datasets, downloading) data the caller never reads.
    """
    transforms = build_transforms(cfg.img_size, cfg.target_size)
    train_loader = build_dataloader(cfg.train_dataloader, build_dataset(cfg.dataset, transforms), shuffle)
    if not val:
        return train_loader, None
    val_loader = build_dataloader(cfg.val_dataloader, build_dataset(cfg.dataset, transforms, split="val"), shuffle)
    return train_loader, val_loader
