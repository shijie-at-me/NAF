import random

import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from torchvision.transforms.functional import InterpolationMode


def seed_worker(worker_id=None):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resize_crop(size, interpolation):
    """Resize the short side to ``size``, then center-crop to ``size`` x ``size``; returns a uint8 tensor."""
    return T.Compose([T.Resize(size, interpolation=interpolation), T.CenterCrop((size, size)), T.PILToTensor()])


def build_transforms(img_size, target_size):
    """Transforms for images (bilinear) and their dense labels (nearest).

    Both stay uint8: images are converted to float in [0, 1] on the device by ``get_batch``, so workers,
    shared memory, pinned memory and the host-to-device copy all carry 4x less data than float32.
    """
    return {
        "image": resize_crop(img_size, InterpolationMode.BILINEAR),
        "label": resize_crop(target_size, InterpolationMode.NEAREST_EXACT),
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


def to_float_image(image):
    """uint8 image -> float32 in [0, 1], bit-identical to ``T.ToTensor()``; float images pass through.

    Divides by a 0-dim tensor rather than a Python scalar: CUDA turns scalar division into multiplication by
    the reciprocal, which is off by one ulp for about half of the values.
    """
    if image.dtype != torch.uint8:
        return image
    return image.float().div_(torch.full((), 255.0, device=image.device))


def get_batch(batch, device):
    """Move the batch images to ``device`` as float32 in [0, 1].

    The copy is asynchronous when the dataloader pins memory; the uint8 -> float conversion runs on the device.
    """
    batch["image"] = to_float_image(batch["image"].to(device, non_blocking=True))
    return batch
