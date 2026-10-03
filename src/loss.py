import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred, target, normalize=False):
        if normalize:
            # Min-max scale both with the per-pixel range of the target channels
            min_val = torch.min(target, dim=1, keepdim=True).values
            max_val = torch.max(target, dim=1, keepdim=True).values
            scale = max_val - min_val + 1e-6
            pred, target = (pred - min_val) / scale, (target - min_val) / scale

        return self.mse_loss(pred, target)


class Loss(nn.Module):
    def __init__(
        self,
        loss_type,
        dim=384,
    ):
        super().__init__()
        self.dim = dim

        if loss_type == "mse":
            loss = MSELoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        self.loss_func = loss

    def forward(self, pred, target, *args, **kwargs):
        return {"total": self.loss_func(pred, target, *args, **kwargs)}


class DenoisingLoss(nn.Module):
    """Weighted L1 + L2 + (3x3 box-window) SSIM loss; returns each weighted term and their sum as ``"total"``."""

    def __init__(self, l1_weight=1.0, l2_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.ssim_weight = ssim_weight

    @staticmethod
    def ssim_loss(pred, target, c1=0.01**2, c2=0.03**2):
        """1 - SSIM with 3x3 box-filtered local statistics."""
        # One pooling call for the five local statistics instead of five
        stats = F.avg_pool2d(torch.cat([pred, target, pred * pred, target * target, pred * target]), 3, 1, 1)
        mu1, mu2, e11, e22, e12 = stats.chunk(5)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1_sq, sigma2_sq, sigma12 = e11 - mu1_sq, e22 - mu2_sq, e12 - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1 - ssim_map.mean()

    def forward(self, pred, target):
        losses = {}
        if self.l1_weight > 0:
            losses["l1"] = F.l1_loss(pred, target) * self.l1_weight
        if self.l2_weight > 0:
            losses["l2"] = F.mse_loss(pred, target) * self.l2_weight
        if self.ssim_weight > 0:
            losses["ssim"] = self.ssim_loss(pred, target) * self.ssim_weight

        losses["total"] = sum(losses.values())
        return losses


def sig_loss(pred, target, eps=1e-3):
    """Scale-invariant log loss (AdaBins / DINOv2 depth heads) over the pixels with valid ground truth."""
    mask = target > eps
    g = torch.log(pred[mask] + eps) - torch.log(target[mask] + eps)
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
        log_diff = (torch.log(pred_s + eps) - torch.log(target_s + eps)) * mask

        v_grad = (log_diff[..., :-2, :] - log_diff[..., 2:, :]).abs() * (mask[..., :-2, :] & mask[..., 2:, :])
        h_grad = (log_diff[..., :, :-2] - log_diff[..., :, 2:]).abs() * (mask[..., :, :-2] & mask[..., :, 2:])
        loss = loss + (h_grad.sum() + v_grad.sum()) / mask.sum()
    return loss


class DepthLoss(nn.Module):
    """probe3d's depth probing loss: weighted SigLoss + gradient loss; depths above ``max_depth`` are ignored."""

    def __init__(self, weight_sig=10.0, weight_grad=0.5, max_depth=10):
        super().__init__()
        self.weight_sig = weight_sig
        self.weight_grad = weight_grad
        self.max_depth = max_depth

    def forward(self, pred, target):
        # Zero out (i.e. mark invalid) depths beyond max_depth, without modifying the caller's tensor
        target = target.masked_fill(target > self.max_depth, 0)
        return self.weight_sig * sig_loss(pred, target) + self.weight_grad * gradient_loss(pred, target)
