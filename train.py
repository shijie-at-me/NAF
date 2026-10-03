import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.backbone import load_multiple_backbones
from utils.checkpoint import build_model, save_checkpoint
from utils.config import expand_user_paths
from utils.data import get_batch, get_dataloaders
from utils.img import normalize_pair, round_to_nearest_multiple
from utils.log import DualConsole, log_losses, print_run_header
from utils.training import autocast, setup_training_optimizations

LOG_FREQ = 100
# Side of the guide image given to the upsampler: 4x the target features, at most GUIDE_MAX_SIZE
GUIDE_MAX_SIZE = 224


def get_lr_size(cfg, height, width, patch_size, min_rescale=0.25, max_rescale=0.60):
    """Size of the low-res input: ``cfg.lr_img_size`` if set, else the image downscaled by ``cfg.down_factor``."""
    if cfg.get("lr_img_size", None) is not None:
        return (cfg.lr_img_size, cfg.lr_img_size)

    if cfg.down_factor == "random":
        downscale_factor = np.random.uniform(min_rescale, max_rescale)
    elif cfg.down_factor == "fixed":
        downscale_factor = 0.5
    else:
        raise ValueError(f"Unknown down_factor: {cfg.down_factor!r}")

    return (
        round_to_nearest_multiple(height * downscale_factor, patch_size),
        round_to_nearest_multiple(width * downscale_factor, patch_size),
    )


@torch.no_grad()
def compute_feats(cfg, backbone, image_batch):
    """Backbone features of the full-res images (target) and of a downscaled copy (input)."""
    hr_feats = backbone(image_batch)

    lr_size = get_lr_size(cfg, *image_batch.shape[-2:], backbone.patch_size)
    low_res_batch = F.interpolate(image_batch, size=lr_size, mode="bilinear")
    lr_feats = backbone(low_res_batch)

    return hr_feats, lr_feats


def prepare_images(images, img_size, backbone):
    """Resize the batch (if needed) and normalize it once for the upsampler and once for the backbone."""
    if images.shape[-2:] != img_size:
        images = F.interpolate(images, size=img_size, mode="bilinear", align_corners=False)
    return normalize_pair(images, backbone)


def compute_loss(cfg, model, backbone, criterion, img_ups, img_back):
    """Upsample low-res backbone features and compare them with the high-res ones."""
    hr_feats, lr_feats = compute_feats(cfg, backbone, img_back)

    guide_size = [min(GUIDE_MAX_SIZE, v * 4) for v in hr_feats.shape[-2:]]
    img_guide = F.interpolate(img_ups, size=guide_size, mode="bilinear")
    pred_feats = model(img_guide, lr_feats, hr_feats.shape[-2:])

    return criterion["mse"](pred_feats.float(), hr_feats.float(), normalize=False)["total"]


def train_step(cfg, model, backbone, criterion, optimizer, img_ups, img_back, use_bf16):
    optimizer.zero_grad(set_to_none=True)
    with autocast(img_ups.device, use_bf16):
        loss = compute_loss(cfg, model, backbone, criterion, img_ups, img_back)
    loss.backward()
    optimizer.step()
    return loss.detach()


def train(cfg, writer, ckpt_dir, console):
    """Train for ``cfg.train_steps`` optimizer steps, or until ``cfg.epochs`` passes over the data if that is fewer."""
    # ============ Backbone ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones, names, _ = load_multiple_backbones(cfg, cfg.backbone, device)
    backbone, backbone_name = backbones[0], names[0]
    console.print(f"[bold cyan]Loaded {len(backbones)} backbones: {names}[/bold cyan]")
    console.print(f"[bold yellow]Using device: {device}[/bold yellow]")
    console.print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    img_size = (cfg.img_size, cfg.img_size)

    # ============ Model, optimizer, losses ============ #
    model = build_model(cfg.model, device, cfg.model_ckpt).train()
    if cfg.model_ckpt is not None:
        console.print(f"[bold green]Loaded model checkpoint from {cfg.model_ckpt}[/bold green]")
    console.print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = instantiate(cfg.optimizer, params=list(model.parameters()))
    criterion = {name: instantiate(loss_cfg) for name, loss_cfg in cfg.loss.items()}

    use_bf16, _ = setup_training_optimizations(model, cfg)
    console.print(f"[bold yellow]Training optimizations: bf16={use_bf16}[/bold yellow]")

    # ============ Data ============ #
    train_dataloader, _ = get_dataloaders(cfg, val=False)
    console.print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Training loop ============ #
    num_batches = len(train_dataloader)
    total_steps = min(cfg.train_steps, cfg.epochs * num_batches)
    checkpoint_interval = max(total_steps // 4, 1)
    step = 0  # optimizer steps done

    for epoch in range(cfg.epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            images = get_batch(batch, device)["image"]
            img_ups, img_back = prepare_images(images, img_size, backbone)
            loss = train_step(cfg, model, backbone, criterion, optimizer, img_ups, img_back, use_bf16)

            if step % LOG_FREQ == 0:
                prefix = (
                    f"Epoch={epoch}/{cfg.epochs} | "
                    f"Batch={batch_idx}/{num_batches} | "
                    f"Progress: {step / total_steps * 100:.1f}% | "
                    f"Image Size={img_size}"
                )
                log_losses(writer, console, {backbone_name: loss}, optimizer.param_groups[0]["lr"], step, prefix)

            step += 1
            if step >= total_steps or cfg.sanity:
                break
            if step % checkpoint_interval == 0:
                console.print(f"Saved checkpoint: {save_checkpoint(model, ckpt_dir, step)}")

        writer.flush()
        if step >= total_steps or cfg.sanity:
            break

    # Always keep the final weights, whether the step budget or the data ran out first
    console.print(f"Saved checkpoint: {save_checkpoint(model, ckpt_dir, step)}")


@hydra.main(config_path="config", config_name="base", version_base=None)
def trainer(cfg: DictConfig):
    expand_user_paths(cfg)
    log_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)

    with DualConsole(os.path.join(log_dir, "train.log")) as console:
        print_run_header(console, cfg)
        train(cfg, writer, log_dir, console)

    writer.close()


if __name__ == "__main__":
    trainer()
