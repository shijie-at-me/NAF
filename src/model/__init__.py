"""Upsampler models.

Every submodule in this package is imported automatically, and the names listed in
its ``__all__`` are re-exported here (e.g. ``src.model.NAF``). To add a model, create
a new file that defines ``__all__``; this file does not need to change.
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
