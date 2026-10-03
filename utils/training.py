import random

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

CHECKPOINTED_MODULE_KEYS = ("cross_decode", "encoder", "sft")


def wrap_forward_with_checkpoint(module):
    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        # Non-reentrant: with the reentrant variant, a module whose inputs don't require grad (e.g. an encoder
        # fed the input image) returns an output without grad, and its parameters silently get no gradient
        return checkpoint.checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module.forward = checkpointed_forward


def enable_gradient_checkpointing(model, keys=CHECKPOINTED_MODULE_KEYS):
    """Use the model's built-in gradient checkpointing, or wrap every submodule whose name contains one of ``keys``.

    Only the outermost matching modules are wrapped: checkpointing a module nested in a checkpointed one saves no
    memory and recomputes it once more per nesting level in the backward pass.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("   [ok] Using built-in gradient checkpointing")
        return

    checkpointed_modules = []
    for name, module in model.named_modules():  # parents come before their children
        if any(key in name for key in keys) and not any(name.startswith(f"{p}.") for p in checkpointed_modules):
            wrap_forward_with_checkpoint(module)
            checkpointed_modules.append(name)

    if checkpointed_modules:
        print(f"   [ok] Applied custom gradient checkpointing to: {checkpointed_modules}")
    else:
        print("   [warning] No modules found for gradient checkpointing")


def setup_training_optimizations(model, cfg):
    """
    Setup training optimizations based on configuration

    Args:
        model: The model to apply optimizations to
        cfg: Configuration object with use_bf16, use_checkpointing and cudnn_benchmark flags

    Returns:
        tuple: (use_bf16, use_checkpointing) for use in training loop. bf16 autocast needs no gradient scaler.
    """
    use_bf16 = cfg.get("use_bf16", False)
    use_checkpointing = cfg.get("use_checkpointing", False)
    # Input sizes are fixed (or take a few values), so cuDNN can pick the fastest conv algorithms once
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", True)

    if use_checkpointing:
        enable_gradient_checkpointing(model)

    print("Training optimizations:")
    print(f"  Mixed precision (bfloat16): {use_bf16}")
    print(f"  Gradient checkpointing: {use_checkpointing}")

    return use_bf16, use_checkpointing


def autocast(device, enabled):
    """bf16 autocast on ``device`` (a no-op context when not ``enabled``)."""
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=enabled)


def seed_everything(seed):
    """Seed the Python, NumPy and PyTorch (CPU and CUDA) random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
