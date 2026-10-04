"""The probe heads: what each task predicts from the upsampled features, its loss and its metrics."""

import torch
import torch.nn.functional as F

from src.dataset.sources import IGNORE_LABEL
from src.losses import DepthLoss
from src.utils.metrics import DEPTH_METRICS, confusion_matrix, depth_metrics, segmentation_scores

from .evaluator import ProbeEvaluator

__all__ = ["DepthProbe", "PROBES", "SegmentationProbe"]

# Depth bins of the depth probe
NUM_DEPTH_BINS = 256


class SegmentationProbe(ProbeEvaluator):
    """Linear semantic segmentation probe; pixels labeled ``IGNORE_LABEL`` count in neither the loss nor the metrics."""

    target_key = "label"

    def __init__(self, model, backbone, device, cfg, *args, **kwargs):
        self.num_classes = cfg.metrics.seg.num_classes
        self.out_channels = self.num_classes
        super().__init__(model, backbone, device, cfg, *args, **kwargs)

    def unpack(self, batch):
        images, target = super().unpack(batch)
        return images, target.long()

    def loss(self, logits, target):
        return F.cross_entropy(logits, target, ignore_index=IGNORE_LABEL)

    def reset_metrics(self):
        # Pixel accuracy and mIoU both come from one confusion matrix, accumulated on the device
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)

    def update_metrics(self, logits, target):
        self.confusion += confusion_matrix(logits.argmax(dim=1), target, self.num_classes)

    def compute_metrics(self):
        return segmentation_scores(self.confusion)


class DepthProbe(ProbeEvaluator):
    """Binned depth probe (probe3d): the depth is the expectation of ``NUM_DEPTH_BINS`` bins under the probe's
    weights."""

    out_channels = NUM_DEPTH_BINS
    target_key = "depth"

    def __init__(self, model, backbone, device, cfg, *args, **kwargs):
        super().__init__(model, backbone, device, cfg, *args, **kwargs)
        self.min_depth, self.max_depth = cfg.metrics.depth.min_depth, cfg.metrics.depth.max_depth
        self.depth_loss = DepthLoss(max_depth=self.max_depth)
        self.bins = torch.linspace(self.min_depth, self.max_depth, NUM_DEPTH_BINS, device=device)

    def head(self, logits, eps=0.1):
        # https://github.com/mbanani/probe3d/blob/c52d00b069d949b2f00c544d4991716df68d5233/evals/models/probes.py#L108
        weights = F.relu(logits) + eps
        weights = weights / weights.sum(dim=1, keepdim=True)
        return torch.einsum("bkhw,k->bhw", weights, self.bins).unsqueeze(1)

    def loss(self, pred, target):
        return self.depth_loss(pred, target)

    def reset_metrics(self):
        self.metric_sums = dict.fromkeys(DEPTH_METRICS, 0.0)
        self.num_images = 0

    def update_metrics(self, pred, target):
        # Averaged over images, whatever the batch size
        for pred_i, target_i in zip(pred, target, strict=True):
            for name, value in depth_metrics(target_i, pred_i, self.min_depth, self.max_depth).items():
                self.metric_sums[name] += value
            self.num_images += 1

    def compute_metrics(self):
        return {name: total / max(self.num_images, 1) for name, total in self.metric_sums.items()}


# eval.task -> probe; the task name also files the trained probes (weights/<model>/probes/<task>/...)
PROBES = {"seg": SegmentationProbe, "depth": DepthProbe}
