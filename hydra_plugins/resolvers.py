from omegaconf import OmegaConf

# (name patterns, feature dim), checked in order
FEATURE_DIMS = (
    (("7b",), 4096),
    (("vits", "small"), 384),
    (("vitb", "base", "radio_v2.5-b"), 768),
    (("vitl", "large", "radio_v2.5-l"), 1024),
    (("so400m",), 1152),
    (("vith", "huge"), 1280),
    (("vitg", "giant"), 1536),
    (("tiny",), 192),
)


def get_feature(target: str) -> int:
    """Feature dimension of a backbone, from the first ``FEATURE_DIMS`` pattern found in its name.

    Raises for a name that matches none: a model built with a made-up dimension would only fail much later.
    """
    model_name = target.lower()
    for patterns, dim in FEATURE_DIMS:
        if any(pattern in model_name for pattern in patterns):
            return dim
    known = ", ".join(f"{'/'.join(patterns)} -> {dim}" for patterns, dim in FEATURE_DIMS)
    raise ValueError(
        f"get_feature: no known feature dimension for backbone {target!r} (name patterns: {known}). "
        "Add its pattern to FEATURE_DIMS in hydra_plugins/resolvers.py, or set the model's feature dim explicitly."
    )


def get_patch_size(target: str) -> int:
    """Patch size of a backbone, as the backbone wrapper infers it (``patchXX`` in the name, or a known family)."""
    # Imported here: the backbone package pulls in timm, which every Hydra startup would otherwise pay for
    from src.backbone.factory import infer_patch_size

    return infer_patch_size(target.lower())


# Hydra imports every module of this package at startup, even one a script already imported itself (e.g.
# tools/failure_analysis.py): re-registering must replace, not raise
OmegaConf.register_new_resolver("get_feature", get_feature, replace=True)
OmegaConf.register_new_resolver("get_patch_size", get_patch_size, replace=True)
