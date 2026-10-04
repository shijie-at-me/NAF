"""Monocular depth losses, for the depth probes (as in probe3d)."""

import torch

from .base import WeightedLoss

__all__ = ["sig_loss", "gradient_loss", "DepthLoss"]


def log_difference(pred, target, eps=1e-3):
    """Difference of the log depths, ``eps`` added to both."""
    return torch.log(pred + eps) - torch.log(target + eps)


def sig_loss(pred, target, eps=1e-3):
    """Scale-invariant log loss (AdaBins / DINOv2 depth heads) over the pixels with valid ground truth."""
    mask = target > eps
    g = log_difference(pred[mask], target[mask], eps)
    return torch.sqrt(torch.var(g) + 0.15 * g.mean().square())


def gradient_loss(pred, target, eps=1e-3, num_scales=4):
    """Multi-scale gradient matching of log depths (MegaDepth) on [B, 1, H, W] maps.

    Same as probe3d's ``gradient_loss``, except that the gradients are taken along H and W (probe3d slices the
    first two dims of the [B, 1, H, W] tensors, i.e. the batch and channel dims).
    """
    loss = 0
    for scale in range(num_scales):
        step = 2 * scale if scale else 1
        pred_s, target_s = pred[..., ::step, ::step], target[..., ::step, ::step]
        mask = target_s > eps
        log_diff = log_difference(pred_s, target_s, eps) * mask

        v_grad = (log_diff[..., :-2, :] - log_diff[..., 2:, :]).abs() * (mask[..., :-2, :] & mask[..., 2:, :])
        h_grad = (log_diff[..., :, :-2] - log_diff[..., :, 2:]).abs() * (mask[..., :, :-2] & mask[..., :, 2:])
        loss = loss + (h_grad.sum() + v_grad.sum()) / mask.sum()
    return loss


class DepthLoss(WeightedLoss):
    """probe3d's depth probing loss: weighted SigLoss + gradient loss; depths above ``max_depth`` are ignored.

    Returns the total only (a scalar), as the probes train on it.
    """

    def __init__(self, weight_sig=10.0, weight_grad=0.5, max_depth=10):
        super().__init__({"sig": weight_sig, "grad": weight_grad})
        self.max_depth = max_depth

    def terms(self):
        return {"sig": sig_loss, "grad": gradient_loss}

    def forward(self, pred, target):
        # Zero out (i.e. mark invalid) depths beyond max_depth, without modifying the caller's tensor
        target = target.masked_fill(target > self.max_depth, 0)
        return super().forward(pred, target)["total"]
