"""Run logs: a console mirrored to a file, the run header, and losses to TensorBoard."""

import datetime

from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from .tensors import to_floats


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
