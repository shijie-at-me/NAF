"""Upsampler models, all with the ``BaseUpsampler`` interface ``forward(image, features, output_size)``.

- ``interpolation`` (Bilinear, Nearest), ``naf``, ``jafar``, ``featup``, ``anyup``: feature upsamplers;
- ``pixelup``: PixelUp, a package of its blocks, checkpoint handling and model;
- ``denoisers``: image restoration baselines for the denoising experiments (IRCNN, REDNet, Restormer, JBF, JBU).

Every submodule and subpackage of this package is imported automatically, and the names listed in its ``__all__``
are re-exported here (e.g. ``src.model.NAF``). To add a model, create a new file (or package) that defines
``__all__``; this file does not need to change.

Use ``get_model(name, **kwargs)`` to build a model from its name.
"""

import importlib
import pkgutil

__all__ = []

for _info in pkgutil.iter_modules(__path__):
    _module = importlib.import_module(f"{__name__}.{_info.name}")
    for _name in getattr(_module, "__all__", []):
        if _name in __all__:
            raise ImportError(f"{__name__}: duplicate export {_name!r} from {_module.__name__}")
        globals()[_name] = getattr(_module, _name)
        __all__.append(_name)

del _info, _module, _name


def get_model(name, **kwargs):
    """Instantiate an exported model by its class name or one of its ``aliases`` (case-insensitive).

    By convention every upsampler accepts ``feature_dim`` (channels of the low-res features) and
    ``ratio`` (upsampling factor), and ignores them through ``**kwargs`` when it does not need them.
    """
    models = {}
    for export in __all__:
        cls = globals()[export]
        for key in (export, *getattr(cls, "aliases", ())):
            models[key.lower()] = cls

    if name.lower() not in models:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(__all__)}")
    return models[name.lower()](**kwargs)
