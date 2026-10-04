"""Project-wide helpers that belong to no single package.

- ``paths``: where weights and fine-tuned backbones live (overridable by environment variables)
- ``config``: ``~`` expansion in configs, paths relative to the launch directory
- ``checkpoint``: checkpoint files (path or URL), and models built from a config with one
- ``img``: image normalization (ImageNet / backbone statistics), resizing, coordinate grids
- ``upsampling``: the shared feature pipeline, frozen backbone features then an upsampler
- ``training``: bf16 autocast, gradient checkpointing, seeding, parameter counts
- ``log``: console mirrored to a log file, run header, losses to TensorBoard
- ``run``: run setup for the entry points (output dir, console, TensorBoard writer, device)
- ``metrics``: PSNR / SSIM, depth metrics, confusion matrix, accuracy and IoU
- ``tensors``: dicts of 0-dim tensors to floats in one host copy
- ``pca``, ``visualization``: feature maps as RGB images, figures, the PASCAL palette

Elsewhere: building datasets and loaders, transforms and synthetic noise are in ``src.dataset`` (``loading``,
``transforms``, ``noise``); the probe registry is ``src.evaluation.probing.registry``.
"""
