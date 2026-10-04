"""Interface of every model: ``forward(image, features, output_size)``."""

from abc import ABC, abstractmethod

import torch.nn as nn

__all__ = ["BaseUpsampler"]


class BaseUpsampler(nn.Module, ABC):
    """Upsamples ``features`` [B, C, h, w] to ``output_size`` guided by ``image`` (ImageNet-normalized).

    The denoisers use the same interface with the noisy image as ``features``. Constructor arguments a model
    doesn't use (``feature_dim``, ``ratio``, ``name``, ...) are swallowed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, image, features, output_size, *args, **kwargs):
        pass
