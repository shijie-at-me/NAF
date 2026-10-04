"""The sample transforms, and the device-side end of them.

Images and dense labels stay uint8 from the dataset to the device: workers, shared memory, pinned memory and the
host-to-device copy all carry 4x less data than float32. ``get_batch`` moves a batch to the device and only then turns
its images into float32 in [0, 1].

Also the synthetic noise of the denoising experiments (``add_noise``)."""

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

__all__ = ["IMAGE_KEYS", "add_noise", "build_transforms", "get_batch", "resize_crop", "to_float_image"]


# Batch keys holding images (converted to float by ``get_batch``); every other tensor is moved as is
IMAGE_KEYS = ("image", "clean")


def resize_crop(size, interpolation):
    """Resize the short side to ``size``, then center-crop to ``size`` x ``size``; returns a uint8 tensor."""
    return T.Compose([T.Resize(size, interpolation=interpolation), T.CenterCrop((size, size)), T.PILToTensor()])


def build_transforms(img_size, target_size):
    """Transforms for images (bilinear, to ``img_size``) and their dense labels (nearest, to ``target_size``)."""
    return {
        "image": resize_crop(img_size, InterpolationMode.BILINEAR),
        "label": resize_crop(target_size, InterpolationMode.NEAREST_EXACT),
    }


def to_float_image(image):
    """uint8 image -> float32 in [0, 1], bit-identical to ``T.ToTensor()``; float images pass through.

    Divides by a 0-dim tensor rather than a Python scalar: CUDA turns scalar division into multiplication by
    the reciprocal, which is off by one ulp for about half of the values.
    """
    if image.dtype != torch.uint8:
        return image
    return image.float().div_(torch.full((), 255.0, device=image.device))


def get_batch(batch, device):
    """Move every tensor of the batch to ``device``; images (``IMAGE_KEYS``) become float32 in [0, 1].

    The copies are asynchronous when the dataloader pins memory; the uint8 -> float conversion runs on the device.
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
            batch[key] = to_float_image(value) if key in IMAGE_KEYS else value
    return batch


# Range sampled per batch when a noise parameter is set to "range"
RANDOM_RANGE = (0.1, 0.5)
DEFAULT_PARAMS = {"gaussian": ("std", 0.1), "salt_pepper": ("prob", 0.1)}


def _rand_like(image, generator=None):
    return torch.rand(image.shape, generator=generator, device=image.device, dtype=image.dtype)


def add_gaussian_noise(image, std=0.1, generator=None):
    noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    return image + noise * std


def add_salt_pepper_noise(image, prob=0.05, generator=None):
    """Replace each value with probability ``prob`` by 0 or 1 (with equal chance)."""
    mask = _rand_like(image, generator) < prob
    salt = (_rand_like(image, generator) > 0.5).to(image.dtype)
    return torch.where(mask, salt, image)


def add_noise(image, noise_type="gaussian", noise_params=None, generator=None):
    """Noisy copy of ``image``; a parameter set to ``"range"`` is drawn uniformly in ``RANDOM_RANGE`` per call.

    Pass a seeded ``torch.Generator`` (on the image's device) for reproducible noise, e.g. for validation.
    """
    if noise_type not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown noise type: {noise_type}")
    name, default = DEFAULT_PARAMS[noise_type]
    value = (noise_params or {}).get(name, default)
    if value == "range":
        low, high = RANDOM_RANGE
        value = low + (high - low) * torch.rand((), generator=generator, device=image.device).item()

    if noise_type == "gaussian":
        return add_gaussian_noise(image, value, generator)
    return add_salt_pepper_noise(image, value, generator)
