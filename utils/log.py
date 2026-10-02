import datetime
import os

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter


class DualConsole:
    """Print rich output to the terminal and mirror it to a log file."""

    def __init__(self, file_path):
        self.terminal = Console()
        self.file = Console(file=open(file_path, "w"))

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


def print_run_header(console: DualConsole, cfg: DictConfig):
    """Print the start timestamp and the resolved config."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold blue]{'=' * 50}[/bold blue]")
    console.print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    console.print("[bold green]Configuration:[/bold green]")
    console.print(OmegaConf.to_yaml(cfg))


def create_writer(base_log_dir):
    """Create a TensorBoard writer in the next free ``version_<n>`` subdirectory of ``base_log_dir``."""
    os.makedirs(base_log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1])
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")
    ]
    log_dir = os.path.join(base_log_dir, f"version_{max(existing_versions, default=-1) + 1}")
    return SummaryWriter(log_dir=log_dir), log_dir
