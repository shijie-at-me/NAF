"""PASCAL VOC 2012 semantic segmentation: a local VOCdevkit, or a dataset on the Hugging Face Hub.

Both return the same batch: ``{"image", "img_path"}`` (+ ``"label"`` with ``include_labels``), where labels are uint8
class ids 0-20 with 255 on object boundaries (ignored). ``datasets`` is imported only by the Hub code, and
``torchvision``'s VOC reader only by the local one.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from evaluation.dataset.image_dataset import resolve_split, to_repo_id

DEFAULT_HF_REPO = "shijli/voc2012"

CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# Images in the standard segmentation splits (the SBD-augmented train split has 10582)
SPLIT_SIZES = {"train": 1464, "val": 1449}


def check_split_size(dataset, split: str):
    """Fail early on an incomplete download or a wrong root: the standard splits have a known size."""
    expected = SPLIT_SIZES.get(split)
    if expected is not None and len(dataset) != expected:
        raise RuntimeError(f"VOC {split} split has {len(dataset)} images, expected {expected}")


# --- datasets -------------------------------------------------------------------------------------------------------


class BaseVOCDataset(Dataset):
    """Applies the transforms and builds the batch; subclasses only say how to load sample ``index``.

    ``kwargs`` swallow config entries that don't apply to the data source.
    """

    def __init__(
        self,
        transform=None,
        target_transform=None,
        include_labels: bool = True,
        num_classes: int = len(CLASS_NAMES),
        tag: str | None = None,
        **kwargs,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.include_labels = include_labels
        self.num_classes = num_classes
        self.tag = tag

    def load(self, index: int) -> tuple[Image.Image, Image.Image | None, str]:
        """Return ``(RGB image, mask or None without labels, path or id)`` of sample ``index``."""
        raise NotImplementedError

    def __getitem__(self, index):
        image, mask, path = self.load(index)
        if self.transform is not None:
            image = self.transform(image)

        batch = {"image": image, "img_path": path}
        if self.include_labels:
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            batch["label"] = torch.as_tensor(np.asarray(mask), dtype=torch.uint8).squeeze()
        return batch

    @property
    def pred_offset(self):
        return 0

    def get_class_names(self):
        return list(CLASS_NAMES)


class VOCDataset(BaseVOCDataset):
    """VOC from ``root/VOCdevkit/VOC<year>``, through torchvision's ``VOCSegmentation`` (which can download it)."""

    def __init__(self, root: str, split: str = "train", year: str = "2012", download: bool = False, **kwargs):
        from torchvision.datasets import VOCSegmentation

        super().__init__(**kwargs)
        self.root = root
        self.split = split
        self.voc = VOCSegmentation(
            root=root, year=year, image_set="train" if split == "train" else "val", download=download
        )
        check_split_size(self, split)

    def load(self, index):
        image_path, mask_path = self.voc.images[index], self.voc.masks[index]
        mask = Image.open(mask_path) if self.include_labels else None
        return Image.open(image_path).convert("RGB"), mask, image_path

    def __len__(self):
        return len(self.voc.images)


class HFVOCDataset(BaseVOCDataset):
    """VOC from a Hub dataset with image and (palette PNG) mask columns, given as a repo id or URL.

    ``name`` is the config: "segmentation" (1464 / 1449 images) or "segmentation_aug" (SBD-augmented train split).
    ``split="val"`` also matches a split named "validation". Downloaded on first use and cached under ``cache_dir``
    (default: ``$HF_HOME/datasets``).
    """

    def __init__(
        self,
        repo: str = DEFAULT_HF_REPO,
        split: str = "train",
        name: str | None = "segmentation",
        cache_dir: str | None = None,
        image_key: str = "image",
        mask_key: str = "mask",
        id_key: str = "id",
        num_proc: int | None = None,
        **kwargs,
    ):
        from datasets import load_dataset

        super().__init__(**kwargs)
        self.repo_id = to_repo_id(repo)
        self.split = resolve_split(self.repo_id, split, name)
        self.data = load_dataset(self.repo_id, name, split=self.split, cache_dir=cache_dir, num_proc=num_proc)
        if not self.include_labels:
            # Rows decode every image column they hold: drop the masks rather than decode them for nothing
            self.data = self.data.remove_columns(mask_key)
        self.image_key, self.mask_key, self.id_key = image_key, mask_key, id_key
        if name == "segmentation":
            check_split_size(self, split)

    def load(self, index):
        item = self.data[index]
        mask = item.get(self.mask_key)
        return item[self.image_key].convert("RGB"), mask, f"{self.repo_id}/{self.split}/{item.get(self.id_key, index)}"

    def __len__(self):
        return len(self.data)
