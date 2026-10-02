from hydra.utils import instantiate
from omegaconf import ListConfig

from src.backbone import PretrainedViTWrapper


def load_backbone(backbone_cfg, device):
    """Load one frozen backbone in eval mode; ``name: rgb`` is instantiated from the config itself."""
    name = backbone_cfg["name"]
    backbone = instantiate(backbone_cfg) if name == "rgb" else PretrainedViTWrapper(name=name)
    return backbone.to(device).eval()


def load_multiple_backbones(cfg, backbone_configs, device):
    """
    Load one or several backbones.

    Returns:
        tuple: (backbones, backbone_names, backbone_img_sizes)
    """
    if not isinstance(backbone_configs, (list, ListConfig)):
        backbone_configs = [backbone_configs]
    print(f"Loading {len(backbone_configs)} backbone(s)...")

    backbones, backbone_names, backbone_img_sizes = [], [], []
    for i, backbone_cfg in enumerate(backbone_configs):
        backbone = load_backbone(backbone_cfg, device)
        print(f"  [{i}] Loaded {backbone_cfg['name']}")

        backbones.append(backbone)
        backbone_names.append(backbone_cfg["name"])
        backbone_img_sizes.append(backbone.config["input_size"][1:])

    return backbones, backbone_names, backbone_img_sizes
