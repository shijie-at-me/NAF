import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.backbone import load_multiple_backbones
from utils.checkpoint import build_model, save_checkpoint
from utils.data import get_batch, get_dataloaders
from utils.img import IMAGENET_MEAN, IMAGENET_STD, round_to_nearest_multiple
from utils.log import DualConsole, log_losses, print_run_header
from utils.training import setup_training_optimizations

LOG_FREQ = 100


def get_lr_size(cfg, height, width, patch_size, min_rescale=0.60, max_rescale=0.25):
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


def prepare_images(images, img_size, ups_norm, back_norm):
    """Resize the batch and normalize it once for the upsampler and once for the backbone."""
    images = F.interpolate(images, size=img_size, mode="bilinear", align_corners=False)
    return ups_norm(images), back_norm(images)


def compute_loss(cfg, model, backbone, criterion, img_ups, img_back):
    """Upsample low-res backbone features and compare them with the high-res ones."""
    hr_feats, lr_feats = compute_feats(cfg, backbone, img_back)

    guide_size = [min(224, v * 4) for v in hr_feats.shape[-2:]]
    img_guide = F.interpolate(img_ups, size=guide_size, mode="bilinear")
    pred_feats = model(img_guide, lr_feats, hr_feats.shape[-2:])

    return criterion["mse"](pred_feats.float(), hr_feats.float(), normalize=False)["total"]


def train_step(cfg, model, backbone, criterion, optimizer, img_ups, img_back, use_bf16):
    optimizer.zero_grad()
    with torch.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
        loss = compute_loss(cfg, model, backbone, criterion, img_ups, img_back)
    loss.backward()
    optimizer.step()
    return loss


def train(cfg, writer, ckpt_dir, console):
    # ============ Backbone ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbones, names, _ = load_multiple_backbones(cfg, cfg.backbone, device)
    backbone, backbone_name = backbones[0], names[0]
    console.print(f"[bold cyan]Loaded {len(backbones)} backbones: {names}[/bold cyan]")
    console.print(f"[bold yellow]Using device: {device}[/bold yellow]")
    console.print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    img_size = (cfg.img_size, cfg.img_size)
    ups_norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    back_norm = T.Normalize(mean=backbone.config["mean"], std=backbone.config["std"])

    # ============ Model, optimizer, losses ============ #
    model = build_model(cfg.model, device, cfg.model_ckpt).train()
    if cfg.model_ckpt is not None:
        console.print(f"[bold green]Loaded model checkpoint from {cfg.model_ckpt}[/bold green]")
    console.print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = instantiate(cfg.optimizer, params=list(model.parameters()))
    criterion = {name: instantiate(loss_cfg) for name, loss_cfg in cfg.loss.items()}

    _, use_bf16, _ = setup_training_optimizations(model, cfg)
    console.print(f"[bold yellow]Training optimizations: bf16={use_bf16}[/bold yellow]")

    # ============ Data ============ #
    train_dataloader, _ = get_dataloaders(cfg)
    console.print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Training loop ============ #
    num_batches = len(train_dataloader)
    total_steps = cfg.epochs * cfg.train_steps
    checkpoint_interval = max(cfg.train_steps // 4, 1)
    done = False

    for epoch in range(cfg.epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            step = epoch * num_batches + batch_idx

            images = get_batch(batch, device)["image"]
            img_ups, img_back = prepare_images(images, img_size, ups_norm, back_norm)
            loss = train_step(cfg, model, backbone, criterion, optimizer, img_ups, img_back, use_bf16)

            if batch_idx % LOG_FREQ == 0:
                prefix = (
                    f"Epoch={epoch}/{cfg.epochs} | "
                    f"Batch={batch_idx}/{num_batches} | "
                    f"Progress: {step / total_steps * 100:.1f}% | "
                    f"Image Size={img_size}"
                )
                log_losses(writer, console, {backbone_name: loss}, optimizer.param_groups[0]["lr"], step, prefix)

            done = step >= cfg.train_steps
            if (batch_idx % checkpoint_interval == 0 and batch_idx != 0) or done:
                console.print(f"Saved checkpoint: {save_checkpoint(model, ckpt_dir, step)}")

            if done or cfg.sanity:
                break

        writer.flush()
        if done:
            break


@hydra.main(config_path="config", config_name="base", version_base=None)
def trainer(cfg: DictConfig):
    log_dir = HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)

    with DualConsole(os.path.join(log_dir, "train.log")) as console:
        print_run_header(console, cfg)
        train(cfg, writer, log_dir, console)

    writer.close()


if __name__ == "__main__":
    trainer()
