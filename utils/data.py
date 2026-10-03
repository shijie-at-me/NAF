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


def build_dataset(dataset_cfg, transforms, split=None):
    """Instantiate a dataset, switched to ``split`` when given and the dataset has a split."""
    if split is not None and "split" in dataset_cfg:
        dataset_cfg = dataset_cfg.copy()
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


def to_float_image(image):
    """uint8 image -> float32 in [0, 1], bit-identical to ``T.ToTensor()``; float images pass through.

    Divides by a 0-dim tensor rather than a Python scalar: CUDA turns scalar division into multiplication by
    the reciprocal, which is off by one ulp for about half of the values.
    """
    if image.dtype != torch.uint8:
        return image
    return image.float().div_(torch.full((), 255.0, device=image.device))


IMAGE_KEYS = ("image", "clean")


def get_batch(batch, device):
    """Move every tensor of the batch to ``device``; images (``IMAGE_KEYS``) become float32 in [0, 1].

    The copies are asynchronous when the dataloader pins memory; the uint8 -> float conversion runs on the device.
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
            batch[key] = to_float_image(value) if key in IMAGE_KEYS else value
    return batch
