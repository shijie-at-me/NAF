"""Base of the dense-label (segmentation) datasets.

Every one returns the batch ``{"image", "img_path"}`` (+ ``"label"`` with ``include_labels``): the transformed image,
its path (or Hub id), and a uint8 [H, W] tensor of train ids with IGNORE_LABEL on the pixels left out.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import IGNORE_LABEL


def train_id_lut(mapping: dict[int, int]) -> torch.Tensor:
    """uint8 table label id (0-255) -> train id from ``{label id: train id}``; the other ids are ignored."""
    lut = torch.full((256,), IGNORE_LABEL, dtype=torch.uint8)
    for label_id, train_id in mapping.items():
        lut[label_id] = train_id
    return lut


def to_label(mask) -> torch.Tensor:
    """[H, W] tensor of a mask: a PIL image, or the [1, H, W] tensor a target transform makes of it."""
    label = torch.as_tensor(np.asarray(mask))
    return label[0] if label.ndim == 3 else label


class SegmentationDataset(Dataset):
    """Applies the transforms, maps label ids to train ids and builds the batch.

    Subclasses fill ``samples`` with ``(image path, label path)`` pairs, or override ``load`` and ``__len__``.
    ``label_lut`` maps the label ids of the files to train ids (None: the files hold train ids). ``num_classes``
    defaults to the number of ``class_names``. ``kwargs`` swallow config entries that don't apply to the dataset.
    """

    class_names: tuple[str, ...] = ()

    def __init__(
        self,
        transform=None,
        target_transform=None,
        include_labels: bool = True,
        num_classes: int | None = None,
        tag: str | None = None,
        **kwargs,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        self.tag = tag
        self.samples: list[tuple[str, str]] = []
        self.label_lut: torch.Tensor | None = None

    def load(self, index: int):
        """``(RGB image, mask or None without labels, path or id)`` of sample ``index``."""
        image_path, label_path = self.samples[index]
        mask = Image.open(label_path) if self.include_labels else None
        return Image.open(image_path).convert("RGB"), mask, image_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, mask, path = self.load(index)
        if self.transform is not None:
            image = self.transform(image)

        batch = {"image": image, "img_path": path}
        if self.include_labels:
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            label = to_label(mask)
            batch["label"] = self.label_lut[label.long()] if self.label_lut is not None else label.to(torch.uint8)
        return batch

    def get_class_names(self):
        return list(self.class_names)

    @property
    def pred_offset(self):
        return 0
