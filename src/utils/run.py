"""Running an entry point (train, evaluation, analysis, denoise): config values (``~``, paths relative to the launch
directory), the Hydra output dir, a console mirrored to a log file, the run header, the device, and losses logged to
TensorBoard."""

import datetime
import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter

from .metrics import to_floats

__all__ = [
    "DualConsole",
    "Run",
    "default_device",
    "expand_user_paths",
    "launch_path",
    "log_losses",
    "print_run_header",
    "start_run",
]


def expand_user_paths(cfg):
    """Expand a leading ``~`` in every literal string of ``cfg``, in place, and return it.

    Shells like PowerShell and values written in YAML don't expand ``~``, so ``dataroot=~/data`` would reach the
    code as is. Interpolations are left untouched: once ``dataroot`` is expanded, ``${dataroot}/imagenet-1k``
    resolves to the expanded path, and unrelated interpolations are never resolved here.
    """
    keys = cfg.keys() if isinstance(cfg, DictConfig) else range(len(cfg))
    for key in keys:
        if OmegaConf.is_interpolation(cfg, key) or OmegaConf.is_missing(cfg, key):
            continue
        value = cfg[key]
        if isinstance(value, (DictConfig, ListConfig)):
            expand_user_paths(value)
        elif isinstance(value, str) and value.startswith("~"):
            cfg[key] = os.path.expanduser(value)
    return cfg


def launch_path(path):
    """A path given in the config, made absolute against the launch directory (Hydra may run from its output
    directory); URLs pass through, and an empty value gives None."""
    if not path:
        return None
    path = str(path)
    return path if path.startswith(("http://", "https://")) else to_absolute_path(path)


class DualConsole:
    """Print rich output to the terminal and mirror it to a log file."""

    def __init__(self, file_path):
        self.terminal = Console()
        # Explicit UTF-8: the platform default (e.g. cp1252 on Windows) can't encode every character rich prints
        self.file = Console(file=open(file_path, "w", encoding="utf-8"))

    def print(self, *args, **kwargs):
        self.terminal.print(*args, **kwargs)
        self.file.print(*args, **kwargs)
        self.file.file.flush()

    def close(self):
        self.file.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def print_run_header(console: DualConsole, cfg: DictConfig, title: str = "Starting"):
    """Print the start timestamp and the resolved config."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold blue]{'=' * 50}[/bold blue]")
    console.print(f"[bold blue]{title} at {timestamp}[/bold blue]")
    console.print("[bold green]Configuration:[/bold green]")
    console.print(OmegaConf.to_yaml(cfg))


def log_losses(writer, console, losses, lr, step, prefix):
    """Write losses and learning rate to TensorBoard and print a one-line summary."""
    values = to_floats(losses)
    for name, value in values.items():
        writer.add_scalar(f"Loss/{name}", value, step)
    writer.add_scalar("Learning Rate", lr, step)

    loss_str = " | ".join(f"{name}: {value:.4f}" for name, value in values.items())
    console.print(f"{prefix} | {loss_str} | ")


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
