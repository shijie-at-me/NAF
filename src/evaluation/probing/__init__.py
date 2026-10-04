"""Linear probing of upsampled features (tasks ``seg_probe`` and ``depth_probe``).

A 1x1 conv probe is trained on the features of a frozen backbone, upsampled by the evaluated model to the label
resolution; the upsampler is frozen too unless ``eval.supervise_model`` is set.

- ``evaluator``: ``ProbeEvaluator``, the shared training / evaluation loop
- ``probes``: the task heads (``SegmentationProbe``, ``DepthProbe``), with their loss and metrics
- ``registry``: saving trained probes with their meta for later analyses
- ``run``: the task entry point, ``run_probing``
"""

from .evaluator import ProbeEvaluator
from .probes import PROBES, DepthProbe, SegmentationProbe
from .run import run_probing

__all__ = ["PROBES", "DepthProbe", "ProbeEvaluator", "SegmentationProbe", "run_probing"]
