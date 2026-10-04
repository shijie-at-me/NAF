"""Feature regression, the task upsamplers are trained on: upsample the backbone features of a downscaled image to
match the features of the full-size image.

Every step: the backbone encodes the image (target features) and a downscaled copy (input features), the upsampler
upsamples the input features to the target size guided by the image (at most GUIDE_MAX_SIZE per side), and the loss
compares both.
"""

import numpy as np
import torch
from hydra.utils import instantiate

from src.backbone import load_multiple_backbones
from src.dataset.loading import get_dataloaders
from src.dataset.transforms import get_batch
from src.utils.checkpoint import build_model, save_checkpoint
from src.utils.img import bilinear_resize, normalize_pair, round_to_nearest_multiple
from src.utils.log import log_losses
from src.utils.training import autocast, count_parameters, setup_training_optimizations

from .loop import is_checkpoint_step, step_budget, training_batches

LOG_FREQ = 100
# Side of the guide image given to the upsampler: 4x the target features, at most GUIDE_MAX_SIZE
GUIDE_MAX_SIZE = 224


# ---- one step ------------------------------------------------------------------------------------------------------


def lr_size(height, width, patch_size, down_factor="fixed", lr_img_size=None, min_rescale=0.25, max_rescale=0.60):
    """Size of the low-res input: ``lr_img_size`` if set, else the image downscaled by half (``down_factor="fixed"``)
    or by a random factor in [min_rescale, max_rescale] (``"random"``), rounded to multiples of the patch size."""
    if lr_img_size is not None:
        return (lr_img_size, lr_img_size)

    if down_factor == "random":
        downscale_factor = np.random.uniform(min_rescale, max_rescale)
    elif down_factor == "fixed":
        downscale_factor = 0.5
    else:
        raise ValueError(f"Unknown down_factor: {down_factor!r}")

    return (
        round_to_nearest_multiple(height * downscale_factor, patch_size),
        round_to_nearest_multiple(width * downscale_factor, patch_size),
    )


@torch.no_grad()
def regression_features(backbone, images, down_factor="fixed", lr_img_size=None):
    """Backbone features of the images (target) and of a downscaled copy (input): ``(hr_feats, lr_feats)``."""
    hr_feats = backbone(images)
    size = lr_size(*images.shape[-2:], backbone.patch_size, down_factor, lr_img_size)
    lr_feats = backbone(bilinear_resize(images, size))
    return hr_feats, lr_feats


def guide_size(feature_size, max_size=GUIDE_MAX_SIZE):
    """Side of the guide image for features of ``feature_size``: 4x, at most ``max_size``."""
    return [min(max_size, 4 * side) for side in feature_size]


def prepare_images(images, img_size, backbone):
    """Resize the batch (if needed) and normalize it once for the upsampler and once for the backbone."""
    if images.shape[-2:] != img_size:
        images = bilinear_resize(images, img_size)
    return normalize_pair(images, backbone)


def regression_loss(model, backbone, loss_fn, img_ups, img_back, down_factor="fixed", lr_img_size=None):
    """Upsample the features of the downscaled images and compare them with those of the images."""
    hr_feats, lr_feats = regression_features(backbone, img_back, down_factor, lr_img_size)
    img_guide = bilinear_resize(img_ups, guide_size(hr_feats.shape[-2:]))
    pred_feats = model(img_guide, lr_feats, hr_feats.shape[-2:])
    return loss_fn(pred_feats.float(), hr_feats.float(), normalize=False)["total"]


def train_step(model, backbone, loss_fn, optimizer, img_ups, img_back, use_bf16, **lr_kwargs):
    """One optimizer step on a batch (bf16 autocast when ``use_bf16``); returns the detached loss."""
    optimizer.zero_grad(set_to_none=True)
    with autocast(img_ups.device, use_bf16):
        loss = regression_loss(model, backbone, loss_fn, img_ups, img_back, **lr_kwargs)
    loss.backward()
    optimizer.step()
    return loss.detach()


# ---- the run -------------------------------------------------------------------------------------------------------


def train(cfg, run):
    """Train for ``cfg.train_steps`` optimizer steps, or until ``cfg.epochs`` passes over the data if that is fewer.

    ``run`` is the ``src.utils.run.Run`` (with a TensorBoard writer): checkpoints go to its dir.
    """
    console, device, writer = run.console, run.device, run.writer

    # ============ Backbone ============ #
    backbones, names, _ = load_multiple_backbones(cfg.backbone, device)
    backbone, backbone_name = backbones[0], names[0]
    console.print(f"[bold cyan]Loaded {len(backbones)} backbones: {names}[/bold cyan]")
    console.print(f"[bold yellow]Using device: {device}[/bold yellow]")
    console.print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")
    img_size = (cfg.img_size, cfg.img_size)

    # ============ Model, optimizer, losses ============ #
    model = build_model(cfg.model, device, cfg.model_ckpt).train()
    if cfg.model_ckpt is not None:
        console.print(f"[bold green]Loaded model checkpoint from {cfg.model_ckpt}[/bold green]")
    console.print(f"Number of parameters: {count_parameters(model)}")
    optimizer = instantiate(cfg.optimizer, params=list(model.parameters()))
    criterion = {name: instantiate(loss_cfg) for name, loss_cfg in cfg.loss.items()}
    use_bf16, _ = setup_training_optimizations(model, cfg)
    console.print(f"[bold yellow]Training optimizations: bf16={use_bf16}[/bold yellow]")
    lr_kwargs = {"down_factor": cfg.down_factor, "lr_img_size": cfg.get("lr_img_size", None)}

    # ============ Data ============ #
    train_dataloader, _ = get_dataloaders(cfg, val=False)
    console.print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Training loop ============ #
    num_batches = len(train_dataloader)
    total_steps = step_budget(cfg.train_steps, cfg.epochs, num_batches)
    steps_done = 0
    batches = training_batches(train_dataloader, cfg.epochs, total_steps, cfg.sanity, on_epoch_end=writer.flush)
    for step, epoch, batch_idx, batch in batches:
        img_ups, img_back = prepare_images(get_batch(batch, device)["image"], img_size, backbone)
        loss = train_step(model, backbone, criterion["mse"], optimizer, img_ups, img_back, use_bf16, **lr_kwargs)

        if step % LOG_FREQ == 0:
            prefix = (
                f"Epoch={epoch}/{cfg.epochs} | "
                f"Batch={batch_idx}/{num_batches} | "
                f"Progress: {step / total_steps * 100:.1f}% | "
                f"Image Size={img_size}"
            )
            log_losses(writer, console, {backbone_name: loss}, optimizer.param_groups[0]["lr"], step, prefix)

        steps_done = step + 1
        if not cfg.sanity and is_checkpoint_step(steps_done, total_steps):
            console.print(f"Saved checkpoint: {save_checkpoint(model, run.dir, steps_done)}")

    # Always keep the final weights, whether the step budget or the data ran out first
    console.print(f"Saved checkpoint: {save_checkpoint(model, run.dir, steps_done)}")
