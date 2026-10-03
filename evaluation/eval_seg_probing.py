import os
import sys

import hydra
import torch
import torch.nn.functional as F

# Project root first: a site-packages "evaluation" or "utils" package must not shadow ours
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.probing import ProbeEvaluator, run_probing
from utils.metrics import confusion_matrix, segmentation_scores

IGNORE = 255


class SegmentationProbe(ProbeEvaluator):
    """Linear semantic segmentation probe; pixels labeled ``IGNORE`` count in neither the loss nor the metrics."""

    target_key = "label"

    def __init__(self, model, backbone, device, cfg, *args, **kwargs):
        self.num_classes = cfg.metrics.seg.num_classes
        self.out_channels = self.num_classes
        super().__init__(model, backbone, device, cfg, *args, **kwargs)

    def unpack(self, batch):
        images, target = super().unpack(batch)
        return images, target.long()

    def loss(self, logits, target):
        return F.cross_entropy(logits, target, ignore_index=IGNORE)

    def reset_metrics(self):
        # Pixel accuracy and mIoU both come from one confusion matrix, accumulated on the device
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long, device=self.device)

    def update_metrics(self, logits, target):
        self.confusion += confusion_matrix(logits.argmax(dim=1), target, self.num_classes)

    def compute_metrics(self):
        return segmentation_scores(self.confusion)


@hydra.main(config_path="../config", config_name="eval_probing", version_base=None)
def main(cfg):
    run_probing(cfg, SegmentationProbe, cfg.eval.task)


if __name__ == "__main__":
    main()
