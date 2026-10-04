"""The shared training / evaluation loop of the linear probes."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm import tqdm

from src.backbone.features import upsample_features
from src.dataset.transforms import get_batch
from src.utils.training import autocast

__all__ = ["ProbeEvaluator"]

LOG_INTERVAL = 100


class ProbeEvaluator:
    """Subclasses set ``out_channels`` and ``target_key`` and implement ``head``, ``loss`` and the metric hooks."""

    out_channels = None
    target_key = None

    def __init__(self, model, backbone, device, cfg, writer, console, train_model=False):
        self.model, self.backbone, self.device = model, backbone, device
        self.cfg, self.writer, self.console = cfg, writer, console
        self.train_model = train_model
        self.use_bf16 = cfg.eval.get("use_bf16", False)
        self.classifier = nn.Conv2d(backbone.embed_dim, self.out_channels, 1).to(device)

    # ---- task-specific hooks ----

    def head(self, logits):
        """Probe output [B, out_channels, H, W] -> prediction."""
        return logits

    def loss(self, pred, target):
        raise NotImplementedError

    def reset_metrics(self):
        raise NotImplementedError

    def update_metrics(self, pred, target):
        raise NotImplementedError

    def compute_metrics(self):
        raise NotImplementedError

    # ---- shared loop ----

    def predict(self, images, output_size):
        """Prediction at ``output_size`` for images in [0, 1]."""
        # The probe runs under the same autocast as the upsampler: with bf16, casting the full-resolution features
        # to float32 first would allocate a second, twice larger copy of them (and keep it for the backward pass)
        with autocast(self.device, self.use_bf16):
            # The upsampler only needs a graph when it is trained along with the probe
            with torch.set_grad_enabled(self.train_model and torch.is_grad_enabled()):
                feats, _ = upsample_features(self.backbone, self.model, images, output_size)
            logits = self.classifier(feats).float()
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear")
        return self.head(logits)

    def set_optimizer(self, num_steps):
        params = list(self.classifier.parameters())
        if self.train_model:
            params += list(self.model.parameters())
        self.optimizer = instantiate(self.cfg.optimizer, params=params)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_steps)
        num_params = sum(p.numel() for p in params if p.requires_grad)
        self.console.print(f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]")

    def unpack(self, batch):
        batch = get_batch(batch, self.device)
        return batch["image"], batch[self.target_key]

    def train_step(self, images, target):
        """One optimizer step on a batch, horizontally flipped with probability 1/2; returns the detached loss."""
        if random.random() < 0.5:
            images, target = images.flip(-1), target.flip(-1)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(self.predict(images, target.shape[-2:]), target)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def log_loss(self, total_loss, num_steps, global_step):
        """Print and write the average loss of the epoch so far."""
        avg_loss = total_loss.item() / num_steps
        lr = self.optimizer.param_groups[0]["lr"]
        self.console.print(f"[cyan]Iteration {num_steps}[/cyan] - Loss: {avg_loss:.6f} - LR: {lr:.5e}")
        self.writer.add_scalar("Loss/Step", avg_loss, global_step)

    def train_epoch(self, loader, epoch):
        self.console.print(f"[yellow]Training model epoch {epoch + 1}...[/yellow]")
        self.backbone.eval()
        self.model.train(self.train_model)
        self.classifier.train()

        # Accumulated on the device: no host sync per step
        total_loss = torch.zeros((), device=self.device)
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}")):
            total_loss += self.train_step(*self.unpack(batch))

            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(loader) - 1 or self.cfg.sanity:
                self.log_loss(total_loss, batch_idx + 1, epoch * len(loader) + batch_idx)
            if self.cfg.sanity:
                break
            self.scheduler.step()

    @torch.inference_mode()
    def evaluate(self, loader, epoch):
        self.console.print("[yellow]Evaluating model...[/yellow]")
        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        self.reset_metrics()
        for batch in tqdm(loader, desc="Evaluating"):
            images, target = self.unpack(batch)
            self.update_metrics(self.predict(images, target.shape[-2:]), target)
            if self.cfg.sanity:
                break

        metrics = self.compute_metrics()
        for name, value in metrics.items():
            self.writer.add_scalar(f"Metrics/{name}", value, epoch)
        self.console.print(f"[bold green]Results: {metrics}[/bold green]")
        return metrics

    def fit(self, train_loader, val_loader):
        """Train for ``cfg.num_epochs`` epochs and return the metrics after the last one.

        Validation runs after the last epoch, and also every ``cfg.eval.val_every`` epochs when that is set (each
        pass upsamples the whole val split, so validating every epoch costs about as much as training).
        """
        num_epochs, val_every = self.cfg.num_epochs, self.cfg.eval.get("val_every")
        self.set_optimizer(num_epochs * len(train_loader))
        self.console.print(f"[yellow]Training for {num_epochs} epochs[/yellow]\n")
        metrics = None
        for epoch in range(num_epochs):
            self.train_epoch(train_loader, epoch)
            if epoch == num_epochs - 1 or (val_every and (epoch + 1) % val_every == 0):
                metrics = self.evaluate(val_loader, epoch)
        return metrics

    def save_checkpoint(self, path):
        torch.save(self.classifier.state_dict(), path)
        self.console.print(f"[bold green]Training completed. Model saved at: {path}[/bold green]")
