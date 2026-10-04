"""Upsampler training (``python train.py``): the generic step loop and the feature-regression task.

- ``loop``: the step budget over epochs (``training_batches``) and when to save checkpoints;
- ``feature_regression``: what a training step computes and the training run itself (``train``).
"""

from .feature_regression import train

__all__ = ["train"]
