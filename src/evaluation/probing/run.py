"""Tasks ``seg_probe`` and ``depth_probe``: train a probe on the upsampled features and evaluate it."""

import os

from torch.utils.tensorboard import SummaryWriter

from src.backbone import load_backbone
from src.dataset.loading import get_dataloaders
from src.utils.checkpoint import load_upsampler
from src.utils.run import launch_path

from .probes import PROBES
from .registry import register_probe

__all__ = ["run_probing"]


def run_probing(cfg, run):
    """Train a probe of ``cfg.eval.task`` on the upsampled features of ``cfg.dataset`` and evaluate it."""
    task = cfg.eval.task
    run.console.print(f"\n[bold cyan]Processing {task} task, image size {cfg.img_size}[/bold cyan]")
    backbone = load_backbone(cfg.backbone, run.device)
    train_model = cfg.eval.get("supervise_model", False)
    model_ckpt = launch_path(cfg.eval.get("model_ckpt"))
    model = load_upsampler(cfg.model, model_ckpt, run.device, run.console, trainable=train_model)

    train_loader, val_loader = get_dataloaders(cfg, shuffle=False)
    run.console.print(f"[bold cyan]Train Dataset size: {len(train_loader.dataset)}[/bold cyan]")
    run.console.print(f"[bold cyan]Val Dataset size: {len(val_loader.dataset)}[/bold cyan]")

    writer = SummaryWriter(log_dir=os.path.join(run.dir, "tb"))
    evaluator = PROBES[task](model, backbone, run.device, cfg, writer, run.console, train_model=train_model)
    metrics = evaluator.fit(train_loader, val_loader)
    evaluator.save_checkpoint(os.path.join(run.dir, "linear_probe.pth"))
    if not cfg.sanity:
        path = register_probe(cfg, task, evaluator.classifier, model_ckpt, metrics, run.dir)
        run.console.print(f"[bold green]Probe registered at: {path}[/bold green]")
    writer.close()
    return metrics
