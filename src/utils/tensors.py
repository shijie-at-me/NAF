"""Small tensor helpers."""

import torch

__all__ = ["to_floats"]


def to_floats(values):
    """``{name: 0-dim tensor}`` -> ``{name: float}`` with a single device-to-host copy (one sync, not one per value).

    Gathered in float64, which holds bf16, float32 and float64 values exactly.
    """
    if not values:
        return {}
    return dict(zip(values, torch.stack([v.detach().double() for v in values.values()]).tolist(), strict=True))
