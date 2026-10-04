"""Config resolvers for backbone properties, registered when Hydra starts (it imports every module of this package).

The knowledge itself lives in ``src/backbone/registry.py``; it is imported when a resolver first runs, since importing
the backbone package pulls in timm, which every Hydra startup would otherwise pay for.
"""

from omegaconf import OmegaConf


def get_feature(target: str) -> int:
    """Feature dimension of a backbone (``src.backbone.registry.feature_dim``); raises for an unknown name."""
    from src.backbone.registry import feature_dim

    return feature_dim(target)


def get_patch_size(target: str) -> int:
    """Patch size of a backbone, as the backbone wrapper infers it (``patchXX`` in the name, or a known family)."""
    from src.backbone.registry import infer_patch_size

    return infer_patch_size(target.lower())


# Hydra imports every module of this package at startup, even one a script already imported itself: re-registering
# must replace, not raise
OmegaConf.register_new_resolver("get_feature", get_feature, replace=True)
OmegaConf.register_new_resolver("get_patch_size", get_patch_size, replace=True)
