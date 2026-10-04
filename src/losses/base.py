"""Weighted sums of named loss terms."""

from collections.abc import Callable

import torch.nn as nn

__all__ = ["WeightedLoss"]


class WeightedLoss(nn.Module):
    """Weighted sum of named loss terms, each a function ``(pred, target) -> scalar``.

    ``forward`` returns every weighted term and their sum as ``"total"``; terms weighted 0 are not computed.
    Subclasses pass ``{name: weight}`` to ``__init__`` and give the matching functions in ``terms``.
    """

    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = dict(weights)

    def terms(self) -> dict[str, Callable]:
        raise NotImplementedError

    def forward(self, pred, target):
        terms = self.terms()
        losses = {name: terms[name](pred, target) * weight for name, weight in self.weights.items() if weight > 0}
        losses["total"] = sum(losses.values())
        return losses
