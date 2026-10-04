"""Image restoration loss, for the denoising experiments."""

import torch.nn.functional as F

from src.utils.metrics import ssim_map, ssim_terms

from .base import WeightedLoss

__all__ = ["DenoisingLoss", "box_ssim_loss"]


def box_ssim_loss(pred, target, c1=0.01**2, c2=0.03**2):
    """1 - SSIM with 3x3 box-filtered local statistics, averaged over the batch."""
    # One pooling call for the five local statistics instead of five
    stats = F.avg_pool2d(ssim_terms(pred, target, dim=0), 3, 1, 1)
    return 1 - ssim_map(*stats.chunk(5), c1=c1, c2=c2).mean()


class DenoisingLoss(WeightedLoss):
    """Weighted L1 + L2 + (3x3 box-window) SSIM loss; returns each weighted term and their sum as ``"total"``."""

    def __init__(self, l1_weight=1.0, l2_weight=1.0, ssim_weight=0.1):
        super().__init__({"l1": l1_weight, "l2": l2_weight, "ssim": ssim_weight})

    def terms(self):
        return {"l1": F.l1_loss, "l2": F.mse_loss, "ssim": box_ssim_loss}
