"""What the evaluation tasks share: their run context, the upsampler under evaluation and fixed-order data loaders."""

from dataclasses import dataclass

import torch
from hydra.utils import to_absolute_path

from src.utils.checkpoint import build_model
from src.utils.data import build_dataloader, build_dataset, build_transforms
from src.utils.log import DualConsole

__all__ = ["RunContext", "checkpoint_arg", "load_upsampler", "split_loader"]


@dataclass
class RunContext:
    """Where a task writes (``run_dir``, ``console``) and runs (``device``); set up by ``evaluation.py``."""

    run_dir: str
    console: DualConsole
    device: torch.device


def checkpoint_arg(path):
    """A checkpoint given in the config: a URL as is, a path relative to the launch directory made absolute."""
    if not path:
        return None
    path = str(path)
    return path if path.startswith(("http://", "https://")) else to_absolute_path(path)


def load_upsampler(cfg, ctx: RunContext, trainable=False):
    """The upsampler of ``cfg.model`` with the weights of ``cfg.eval.model_ckpt`` (path or URL; null: as built),
    frozen unless ``trainable``; returns ``(model, checkpoint path or URL)``."""
    model_ckpt = checkpoint_arg(cfg.eval.get("model_ckpt"))
    model = build_model(cfg.model, ctx.device, model_ckpt, weights_only=False)
    if model_ckpt:
        ctx.console.print(f"[green]Loaded model from checkpoint: {model_ckpt}[/green]")
    else:
        ctx.console.print("[yellow]No model checkpoint provided, using the model as built[/yellow]")
    return model.requires_grad_(trainable).eval(), model_ckpt


def split_loader(cfg, split, loader_cfg, size=None):
    """Loader of a split of ``cfg.dataset`` in a fixed order; images and labels resized and center-cropped to
    ``size`` (default: ``cfg.target_size``)."""
    size = size or cfg.target_size
    dataset = build_dataset(cfg.dataset, build_transforms(size, size), split=split)
    return build_dataloader(loader_cfg, dataset, shuffle=False)
