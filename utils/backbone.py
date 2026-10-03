import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import ListConfig

from src.backbone import PretrainedViTWrapper
from utils.img import normalize_pair


def load_backbone(backbone_cfg, device):
    """Load one frozen backbone in eval mode; ``name: rgb`` is instantiated from the config itself."""
    name = backbone_cfg["name"]
    backbone = instantiate(backbone_cfg) if name == "rgb" else PretrainedViTWrapper(name=name)
    return backbone.to(device).eval().requires_grad_(False)


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


def upsample_features(backbone, upsampler, images, output_size, guide_size=None):
    """Features of the frozen ``backbone`` for ``images`` (in [0, 1]) upsampled to ``output_size``.

    The upsampler is guided by the ImageNet-normalized images, resized to ``guide_size`` if given. Gradients flow
    into the upsampler (wrap the call in ``torch.no_grad()`` when it is frozen) but never into the backbone.

    Returns:
        (high-res features, low-res backbone features)
    """
    img_ups, img_bck = normalize_pair(images, backbone)
    with torch.no_grad():
        lr_feats = backbone(img_bck)
    if guide_size is not None:
        img_ups = F.interpolate(img_ups, size=guide_size, mode="bicubic", align_corners=False)
    return upsampler(img_ups, lr_feats, output_size), lr_feats
