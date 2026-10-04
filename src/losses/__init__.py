"""Losses, one module per task; everything is re-exported here (e.g. ``src.losses.Loss`` in the configs).

- ``base``: ``WeightedLoss``, a weighted sum of named terms that also returns each term
- ``feature``: feature regression for training upsamplers (``Loss``, ``feature_mse``)
- ``denoising``: L1 + L2 + SSIM for the denoising experiments (``DenoisingLoss``)
- ``depth``: probe3d's depth probing losses (``DepthLoss``, ``sig_loss``, ``gradient_loss``)
"""

from .base import WeightedLoss
from .denoising import DenoisingLoss, box_ssim_loss
from .depth import DepthLoss, gradient_loss, sig_loss
from .feature import FEATURE_LOSSES, Loss, feature_mse, minmax_normalize

__all__ = [
    "WeightedLoss",
    "Loss",
    "FEATURE_LOSSES",
    "feature_mse",
    "minmax_normalize",
    "DenoisingLoss",
    "box_ssim_loss",
    "DepthLoss",
    "sig_loss",
    "gradient_loss",
]
