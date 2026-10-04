"""What the evaluation tasks share: their run context, the upsampler under evaluation and fixed-order data loaders."""

from dataclasses import dataclass

import torch

from src.dataset.loading import build_dataloader, build_dataset
from src.dataset.transforms import build_transforms
from src.utils.checkpoint import build_model
from src.utils.config import launch_path
from src.utils.log import DualConsole

__all__ = ["RunContext", "load_upsampler", "split_loader"]


@dataclass
class RunContext:
    """Where a task writes (``run_dir``, ``console``) and runs (``device``); set up by ``evaluation.py``."""

    run_dir: str
    console: DualConsole
    device: torch.device


def load_upsampler(cfg, ctx: RunContext, trainable=False):
    """The upsampler of ``cfg.model`` with the weights of ``cfg.eval.model_ckpt`` (path or URL; null: as built),
    frozen unless ``trainable``; returns ``(model, checkpoint path or URL)``."""
    model_ckpt = launch_path(cfg.eval.get("model_ckpt"))
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
