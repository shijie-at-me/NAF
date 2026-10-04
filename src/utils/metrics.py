"""Evaluation metrics, computed per image on the device."""

import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    """PSNR of every image of the batch: [B] (``inf`` for a perfect prediction)."""
    mse = (pred.float() - target.float()).square().flatten(1).mean(1)
    return 10 * torch.log10(max_val**2 / mse)


def gaussian_window(window_size, sigma, channels, device=None):
    """Depthwise 2D Gaussian kernel: [channels, 1, window_size, window_size]."""
    x = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-x.square() / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g).expand(channels, 1, window_size, window_size)


def ssim_terms(pred, target, dim):
    """The five maps SSIM takes local means of, concatenated along ``dim``: x, y, x^2, y^2 and xy, so that one
    filtering pass gives all the local statistics (split them with ``.chunk(5, dim)`` for ``ssim_map``)."""
    return torch.cat([pred, target, pred * pred, target * target, pred * target], dim=dim)


def ssim_map(mu1, mu2, e11, e22, e12, c1=0.01**2, c2=0.03**2):
    """SSIM at every position from the local means of x, y, x^2, y^2 and xy (``ssim_terms`` filtered)."""
    mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq, sigma2_sq, sigma12 = e11 - mu1_sq, e22 - mu2_sq, e12 - mu1_mu2
    return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))


def ssim(pred, target, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    """SSIM of every image of the batch (Gaussian window, zero padded, data range 1): [B]."""
    pred, target = pred.float(), target.float()
    channels = pred.shape[1]
    window = gaussian_window(window_size, sigma, channels, pred.device)
    # One depthwise convolution for the five local statistics
    stats = F.conv2d(
        ssim_terms(pred, target, dim=1),
        window.repeat(5, 1, 1, 1),
        padding=window_size // 2,
        groups=5 * channels,
    )
    return ssim_map(*stats.chunk(5, dim=1), c1=c1, c2=c2).flatten(1).mean(1)


DEPTH_METRICS = ("d1", "d2", "d3", "abs_rel", "sq_rel", "rmse", "rmse_log", "log_10", "silog")


def depth_metrics(gt, pred, min_depth=1e-3, max_depth=10):
    """Standard monocular depth metrics over the pixels with ``min_depth < gt < max_depth``: dict of floats."""
    mask = (gt > min_depth) & (gt < max_depth)
    gt, pred = gt[mask].double(), pred[mask].double()

    thresh = torch.maximum(gt / pred, pred / gt)
    err = torch.log(pred) - torch.log(gt)
    metrics = {
        "d1": (thresh < 1.25).double().mean(),
        "d2": (thresh < 1.25**2).double().mean(),
        "d3": (thresh < 1.25**3).double().mean(),
        "abs_rel": ((gt - pred).abs() / gt).mean(),
        "sq_rel": ((gt - pred).square() / gt).mean(),
        "rmse": (gt - pred).square().mean().sqrt(),
        "rmse_log": err.square().mean().sqrt(),
        "log_10": (torch.log10(gt) - torch.log10(pred)).abs().mean(),
        "silog": torch.sqrt(err.square().mean() - err.mean().square()) * 100,
    }
    # One device-to-host copy for all of them
    return dict(zip(metrics, torch.stack(list(metrics.values())).tolist(), strict=True))


def confusion_matrix(pred, target, num_classes):
    """Counts of (target, prediction) class pairs: [num_classes, num_classes] int64, rows indexed by the target.

    Pixels whose target is not a class index (e.g. 255, the usual ignore label) are left out. Any integer dtype
    works: the pairs are indexed in int64 (``target * num_classes`` would overflow uint8 labels).
    """
    valid = (target >= 0) & (target < num_classes)
    pairs = target[valid].long() * num_classes + pred[valid].long()
    return torch.bincount(pairs, minlength=num_classes**2).view(num_classes, num_classes)


def segmentation_scores(confusion):
    """Pixel accuracy and mean IoU of a confusion matrix, as torchmetrics' multiclass ``Accuracy`` (micro) and
    ``JaccardIndex`` (macro): the mean is over the classes found in the targets or the predictions."""
    confusion = confusion.double()
    tp = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - tp
    present = union > 0
    accuracy = tp.sum() / confusion.sum().clamp(min=1)
    iou = (tp[present] / union[present]).mean() if present.any() else torch.zeros_like(accuracy)
    # One device-to-host copy for both
    accuracy, iou = torch.stack([accuracy, iou]).tolist()
    return {"accuracy": accuracy, "iou": iou}
