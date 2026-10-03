"""Synthetic image noise for the denoising experiments."""

import torch

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
