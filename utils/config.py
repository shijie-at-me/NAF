import os

from omegaconf import DictConfig, ListConfig, OmegaConf


def expand_user_paths(cfg):
    """Expand a leading ``~`` in every literal string of ``cfg``, in place, and return it.

    Shells like PowerShell and values written in YAML don't expand ``~``, so ``dataroot=~/data`` would reach the
    code as is. Interpolations are left untouched: once ``dataroot`` is expanded, ``${dataroot}/imagenet-1k``
    resolves to the expanded path, and unrelated interpolations are never resolved here.
    """
    keys = cfg.keys() if isinstance(cfg, DictConfig) else range(len(cfg))
    for key in keys:
        if OmegaConf.is_interpolation(cfg, key) or OmegaConf.is_missing(cfg, key):
            continue
        value = cfg[key]
        if isinstance(value, (DictConfig, ListConfig)):
            expand_user_paths(value)
        elif isinstance(value, str) and value.startswith("~"):
            cfg[key] = os.path.expanduser(value)
    return cfg
