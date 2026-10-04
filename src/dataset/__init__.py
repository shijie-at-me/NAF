"""Datasets, instantiated from ``config/dataset/*.yaml`` (``_target_: src.dataset.<module>.<class>``).

- ``imagenet``: ImageNet-style class-labeled images for upsampler training (local class folders or the Hub);
- ``segmentation``: base of the dense-label datasets (batch, label-id -> train-id table, class names), subclassed by
  ``voc``, ``coco``, ``ade20k``, ``cityscapes``, ``kitti360`` and ``davis``;
- ``cityscapes_labels``: the Cityscapes / KITTI-360 label table;
- ``common``: split names and sizes, file listings; ``hub``: Hugging Face Hub datasets.
- ``loading``: datasets and data loaders from their configs; ``transforms``: uint8 sample transforms and batches
  to the device; ``noise``: synthetic noise for the denoising experiments.
"""
