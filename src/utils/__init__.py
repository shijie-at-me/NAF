"""Project-wide helpers that belong to no single package.

- ``run``: running an entry point: config paths (``~``, launch directory), the console mirrored to a log file, the
  run header, the output dir, TensorBoard and the device (``start_run``)
- ``checkpoint``: where weights live (overridable by environment variables), checkpoint files (path or URL), and
  models built from a config with one (``build_model``, ``load_upsampler``)
- ``image``: image normalization (ImageNet / backbone statistics), resizing, coordinate grids
- ``metrics``: PSNR / SSIM, depth metrics, confusion matrix, accuracy and IoU, losses to floats
- ``training``: bf16 autocast, gradient checkpointing, seeding, parameter counts
- ``visualization``: feature maps as RGB images (PCA), figures, the PASCAL palette

Elsewhere: datasets, loaders, transforms and synthetic noise are in ``src.dataset``; the backbone -> upsampler feature
pipeline is ``src.backbone.features``; the probe registry is ``src.evaluation.probing.registry``.
"""
