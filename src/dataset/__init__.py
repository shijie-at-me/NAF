"""Datasets, instantiated from ``config/dataset/*.yaml`` (``_target_: src.dataset.<module>.<class>``).

- ``imagenet``: ImageNet-style class-labeled images for upsampler training (local class folders or the Hub);
- ``segmentation``: base of the dense-label datasets (batch, label-id -> train-id table, class names), subclassed by
  ``voc``, ``coco``, ``ade20k``, ``cityscapes`` (with the Cityscapes label table, also used by ``kitti360``),
  ``kitti360`` and ``davis``;
- ``sources``: split names and sizes, local file listings, Hugging Face Hub datasets;
- ``loading``: datasets and data loaders from their configs, fixed-order split loaders;
- ``transforms``: uint8 sample transforms, batches to the device, synthetic noise for the denoising experiments.
"""
