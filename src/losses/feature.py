"""Feature regression losses, for training upsamplers on backbone features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FEATURE_LOSSES", "Loss", "feature_mse", "minmax_normalize"]


def minmax_normalize(pred, target, eps=1e-6):
    """Min-max scale ``pred`` and ``target`` with the per-pixel range of the target channels."""
    min_val = torch.min(target, dim=1, keepdim=True).values
    max_val = torch.max(target, dim=1, keepdim=True).values
    scale = max_val - min_val + eps
    return (pred - min_val) / scale, (target - min_val) / scale


def feature_mse(pred, target, normalize=False):
    """MSE between feature maps, optionally after ``minmax_normalize``."""
    if normalize:
        pred, target = minmax_normalize(pred, target)
    return F.mse_loss(pred, target)


FEATURE_LOSSES = {"mse": feature_mse}


class Loss(nn.Module):
    """The feature loss named ``loss_type`` (see ``FEATURE_LOSSES``), returned as ``{"total": loss}``."""

    def __init__(self, loss_type):
        super().__init__()
        if loss_type not in FEATURE_LOSSES:
            raise NotImplementedError(f"Loss type {loss_type} not implemented, expected one of {list(FEATURE_LOSSES)}")
        self.loss_type = loss_type
        self.loss_func = FEATURE_LOSSES[loss_type]

    def forward(self, pred, target, *args, **kwargs):
        return {"total": self.loss_func(pred, target, *args, **kwargs)}
