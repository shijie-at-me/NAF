"""Parameter-free baselines: plain interpolation of the low-res features, the guide image unused."""

import torch.nn.functional as F

from src.model.base import BaseUpsampler

__all__ = ["Bilinear", "Nearest"]


class Bilinear(BaseUpsampler):
    def forward(self, image, features, output_size, *args, **kwargs):
        return F.interpolate(features, size=output_size, mode="bilinear")


class Nearest(BaseUpsampler):
    def forward(self, image, features, output_size, *args, **kwargs):
        return F.interpolate(features, size=output_size, mode="nearest-exact")
