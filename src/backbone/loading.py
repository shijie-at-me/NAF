"""Frozen backbones from their configs (``{"name": ...}``), ready for feature extraction."""

from hydra.utils import instantiate
from omegaconf import ListConfig

from .wrapper import PretrainedViTWrapper

__all__ = ["load_backbone", "load_multiple_backbones"]


def load_backbone(backbone_cfg, device):
    """Load one frozen backbone in eval mode; ``name: rgb`` is instantiated from the config itself.

    ``layer`` (optional) is the block whose features it returns, 1-based; unset or null for the last one.
    """
    name = backbone_cfg["name"]
    if name == "rgb":
        backbone = instantiate(backbone_cfg)
    else:
        backbone = PretrainedViTWrapper(name=name, layer=backbone_cfg.get("layer"))
    return backbone.to(device).eval().requires_grad_(False)


def load_multiple_backbones(backbone_configs, device):
    """Load one backbone config or a list of them.

    Returns:
        (backbones, names, input sizes (H, W) from their data configs)
    """
    if not isinstance(backbone_configs, (list, tuple, ListConfig)):
        backbone_configs = [backbone_configs]
    print(f"Loading {len(backbone_configs)} backbone(s)...")

    backbones, names, img_sizes = [], [], []
    for i, backbone_cfg in enumerate(backbone_configs):
        backbone = load_backbone(backbone_cfg, device)
        print(f"  [{i}] Loaded {backbone_cfg['name']}")
        backbones.append(backbone)
        names.append(backbone_cfg["name"])
        img_sizes.append(backbone.config["input_size"][1:])
    return backbones, names, img_sizes
