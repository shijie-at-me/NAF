"""Evaluation tasks, run through ``evaluation.py task=<name>`` (configs in ``config/task/``).

- ``probing``: linear probes on upsampled features (``seg_probe``, ``depth_probe``)
- ``video_seg``: DAVIS label propagation, scored with J&F (``video_seg``)

Every task is a ``run(cfg, run)`` function: ``run`` is the ``src.utils.run.Run`` (dir, console, device) of the run.
"""

from .probing import run_probing
from .video_seg import run_video_seg

# Task name (config/task/<name>.yaml) -> run(cfg, run)
TASKS = {
    "seg_probe": run_probing,
    "depth_probe": run_probing,
    "video_seg": run_video_seg,
}

__all__ = ["TASKS"]
