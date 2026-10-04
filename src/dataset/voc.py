"""PASCAL VOC 2012 semantic segmentation: a local VOCdevkit, or a dataset on the Hugging Face Hub.

Labels are class ids 0-20 with 255 on object boundaries (ignored). ``datasets`` is imported only by the Hub code, and
``torchvision``'s VOC reader only by the local one.
"""

from .common import check_split_size
from .hub import load_hub_split
from .segmentation import SegmentationDataset

DEFAULT_HF_REPO = "shijli/voc2012"

# fmt: off
CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
# fmt: on

# Images in the standard segmentation splits (the SBD-augmented train split has 10582)
SPLIT_SIZES = {"train": 1464, "val": 1449}


class VOCDataset(SegmentationDataset):
    """VOC from ``root/VOCdevkit/VOC<year>``, through torchvision's ``VOCSegmentation`` (which can download it)."""

    class_names = CLASS_NAMES

    def __init__(self, root: str, split: str = "train", year: str = "2012", download: bool = False, **kwargs):
        from torchvision.datasets import VOCSegmentation

        super().__init__(**kwargs)
        self.root = root
        self.split = split
        voc = VOCSegmentation(root=root, year=year, image_set="train" if split == "train" else "val", download=download)
        self.samples = list(zip(voc.images, voc.masks, strict=True))
        check_split_size("VOC", split, len(self), SPLIT_SIZES)


class HFVOCDataset(SegmentationDataset):
    """VOC from a Hub dataset with image and (palette PNG) mask columns, given as a repo id or URL.

    ``name`` is the config: "segmentation" (1464 / 1449 images) or "segmentation_aug" (SBD-augmented train split).
    ``split="val"`` also matches a split named "validation". Downloaded on first use and cached under ``cache_dir``
    (default: ``$HF_HOME/datasets``).
    """

    class_names = CLASS_NAMES

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
        super().__init__(**kwargs)
        self.repo_id, self.split, self.data = load_hub_split(repo, split, name, cache_dir, num_proc)
        if not self.include_labels:
            # Rows decode every image column they hold: drop the masks rather than decode them for nothing
            self.data = self.data.remove_columns(mask_key)
        self.image_key, self.mask_key, self.id_key = image_key, mask_key, id_key
        if name == "segmentation":
            check_split_size("VOC", split, len(self), SPLIT_SIZES)

    def load(self, index):
        item = self.data[index]
        mask = item.get(self.mask_key)
        return item[self.image_key].convert("RGB"), mask, f"{self.repo_id}/{self.split}/{item.get(self.id_key, index)}"

    def __len__(self):
        return len(self.data)
