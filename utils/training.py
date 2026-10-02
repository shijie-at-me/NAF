import torch
import torch.utils.checkpoint as checkpoint

CHECKPOINTED_MODULE_KEYS = ("cross_decode", "encoder", "sft")


def wrap_forward_with_checkpoint(module):
    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        return checkpoint.checkpoint(original_forward, *args, **kwargs)

    module.forward = checkpointed_forward


def enable_gradient_checkpointing(model, keys=CHECKPOINTED_MODULE_KEYS):
    """Use the model's built-in gradient checkpointing, or wrap every submodule whose name contains one of ``keys``."""
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("   ✓ Using built-in gradient checkpointing")
        return

    checkpointed_modules = []
    for name, module in model.named_modules():
        if any(key in name for key in keys):
            wrap_forward_with_checkpoint(module)
            checkpointed_modules.append(name)

    if checkpointed_modules:
        print(f"   ✓ Applied custom gradient checkpointing to: {checkpointed_modules}")
    else:
        print("   ⚠ No modules found for gradient checkpointing")


def setup_training_optimizations(model, cfg):
    """
    Setup training optimizations based on configuration

    Args:
        model: The model to apply optimizations to
        cfg: Configuration object with use_bf16 and use_checkpointing flags

    Returns:
        tuple: (scaler, use_bf16, use_checkpointing) for use in training loop
    """
    use_bf16 = getattr(cfg, "use_bf16", False)
    use_checkpointing = getattr(cfg, "use_checkpointing", False)

    scaler = torch.amp.GradScaler("cuda", enabled=use_bf16)

    if use_checkpointing:
        enable_gradient_checkpointing(model)

    print("Training optimizations:")
    print(f"  Mixed precision (bfloat16): {use_bf16}")
    print(f"  Gradient checkpointing: {use_checkpointing}")

    return scaler, use_bf16, use_checkpointing
