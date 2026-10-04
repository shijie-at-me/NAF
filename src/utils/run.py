"""Start of a Hydra run (training or evaluation): its output dir, a console mirrored to a log file, the run header,
the device and, optionally, a TensorBoard writer."""

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

from .config import expand_user_paths
from .log import DualConsole, print_run_header


def default_device():
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Run:
    dir: str  # Hydra's output dir: logs, checkpoints and results
    console: DualConsole
    device: torch.device
    writer: SummaryWriter | None = None


@contextmanager
def start_run(cfg, log_name, title="Starting", tensorboard=False):
    """Expand the user paths of ``cfg``, open ``<run dir>/<log_name>`` and print the run header; yields the ``Run``.

    With ``tensorboard``, the run also gets a ``SummaryWriter`` on the run dir, closed when the run ends.
    """
    expand_user_paths(cfg)
    run_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=run_dir) if tensorboard else None
    try:
        with DualConsole(os.path.join(run_dir, log_name)) as console:
            print_run_header(console, cfg, title)
            yield Run(run_dir, console, default_device(), writer)
    finally:
        if writer is not None:
            writer.close()
