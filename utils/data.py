import random

import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from torchvision.transforms.functional import InterpolationMode

from utils.img import PILToTensor


def seed_worker(worker_id=None):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms(img_size, target_size):
    """Resize + center-crop transforms for images and their dense labels."""
    return {
        "image": T.Compose(
            [
                T.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
                T.CenterCrop((img_size, img_size)),
                T.ToTensor(),
            ]
        ),
        "label": T.Compose(
            [
                T.Resize(target_size, interpolation=InterpolationMode.NEAREST_EXACT),
                T.CenterCrop((target_size, target_size)),
                PILToTensor(),
            ]
        ),
    }


def build_datasets(dataset_cfg, transforms):
    """Instantiate the train dataset and a val copy of it (``split="val"`` when the dataset has a split)."""
    val_dataset_cfg = dataset_cfg.copy()
    if hasattr(val_dataset_cfg, "split"):
        val_dataset_cfg.split = "val"

    return tuple(
        instantiate(cfg, transform=transforms["image"], target_transform=transforms["label"])
        for cfg in (dataset_cfg, val_dataset_cfg)
    )


def build_dataloader(dataloader_cfg, dataset, shuffle):
    """Instantiate a dataloader; deterministic order when ``shuffle`` is False, true randomness otherwise."""
    dataloader_cfg = dataloader_cfg.copy()
    if shuffle:
        generator = None
        if "worker_init_fn" in dataloader_cfg:
            dataloader_cfg["worker_init_fn"] = None
    else:
        generator = torch.Generator()
        generator.manual_seed(0)

    return instantiate(dataloader_cfg, dataset=dataset, generator=generator)


def get_dataloaders(cfg, shuffle=True):
    """Build the train and val dataloaders described by ``cfg``."""
    transforms = build_transforms(cfg.img_size, cfg.target_size)
    train_dataset, val_dataset = build_datasets(cfg.dataset, transforms)
    return (
        build_dataloader(cfg.train_dataloader, train_dataset, shuffle),
        build_dataloader(cfg.val_dataloader, val_dataset, shuffle),
    )


def get_batch(batch, device):
    """Move the image tensor of a batch to ``device``."""
    batch["image"] = batch["image"].to(device)
    return batch
