"""Evaluation tasks, run through ``evaluation.py task=<name>`` (configs in ``config/task/``).

- ``probing``: linear probes on upsampled features (``seg_probe``, ``depth_probe``)
- ``video_seg``: DAVIS label propagation (``video_seg``), scored by ``davis``
- ``attention``: statistics of NAF's attention around object boundaries (``attention_entropy``)
- ``guide_ratio``: output / guide size settings scored with one shared probe (``guide_ratio``)
- ``boundary``: distance to the nearest object boundary, to break metrics down by it
- ``common``: run context, upsampler loading and data loaders shared by the tasks
"""

from .attention import run_attention_entropy
from .guide_ratio import run_guide_ratio
from .probing import run_probing
from .video_seg import run_video_seg

# Task name (config/task/<name>.yaml) -> run(cfg, ctx)
TASKS = {
    "seg_probe": run_probing,
    "depth_probe": run_probing,
    "video_seg": run_video_seg,
    "attention_entropy": run_attention_entropy,
    "guide_ratio": run_guide_ratio,
}

__all__ = ["TASKS"]
