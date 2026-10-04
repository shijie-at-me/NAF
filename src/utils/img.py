import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


def create_coordinate(h, w, start=0, end=1, device=None, dtype=torch.float32):
    """(row, column) coordinates of an h x w grid, each linearly spaced in [start, end]: [1, h * w, 2]."""
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).view(1, h * w, 2)


def normalize(images, mean, std):
    """Per-channel ``(images - mean) / std`` of a [..., 3, H, W] batch."""
    return TF.normalize(images, mean=mean, std=std)


def normalize_pair(images, backbone):
    """Images in [0, 1] normalized for the upsampler (ImageNet statistics) and for ``backbone`` (its own)."""
    return normalize(images, IMAGENET_MEAN, IMAGENET_STD), normalize(
        images, backbone.config["mean"], backbone.config["std"]
    )


def bilinear_resize(images, size):
    """Bilinear resize (``align_corners=False``) of [B, C, H, W] images or feature maps to ``size``."""
    return F.interpolate(images, size=size, mode="bilinear", align_corners=False)
