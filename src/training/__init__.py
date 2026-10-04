"""Training: the generic step loop, the feature-regression task upsamplers are trained on (``python train.py``) and
the denoising side experiment (``python denoise.py``).

- ``loop``: the step budget over epochs (``training_batches``) and when to save checkpoints;
- ``feature_regression``: what a training step computes and the training run itself (``train``);
- ``denoising``: noisy / clean pairs, the training and validation of a denoiser (``train_denoising``).
"""

from .feature_regression import train

__all__ = ["train"]
