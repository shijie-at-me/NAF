"""Cityscapes semantic segmentation (fine annotations) on its 19 train classes.

Cityscapes-style label definitions, shared by Cityscapes (ids 0-33) and KITTI-360 (which adds ids 34-44).

From https://github.com/autonomousvision/kitti360Scripts/blob/32f6d64eef27c32b52c542b4e95be8af9c9c6444/kitti360scripts/helpers/labels.py

From a local Cityscapes folder, or a private dataset on the Hugging Face Hub (the Cityscapes license forbids
distributing the data). ``datasets`` is imported only by the Hub code.
"""

from collections import namedtuple

from .segmentation import SegmentationDataset, train_id_lut
from .sources import IGNORE_LABEL, check_split_size, load_hub_split

DEFAULT_HF_REPO = "shijli/cityscapes"

Label = namedtuple("Label", "name id kittiId trainId category categoryId hasInstances ignoreInEval ignoreInInst color")

# fmt: off
LABELS = (
    #     name                     id  kittiId trainId  category       catId  hasInstances ignoreInEval ignoreInInst color
    Label("unlabeled",              0, -1, 255, "void",         0, False, True,  True,  (0, 0, 0)),
    Label("ego vehicle",            1, -1, 255, "void",         0, False, True,  True,  (0, 0, 0)),
    Label("rectification border",   2, -1, 255, "void",         0, False, True,  True,  (0, 0, 0)),
    Label("out of roi",             3, -1, 255, "void",         0, False, True,  True,  (0, 0, 0)),
    Label("static",                 4, -1, 255, "void",         0, False, True,  True,  (0, 0, 0)),
    Label("dynamic",                5, -1, 255, "void",         0, False, True,  True,  (111, 74, 0)),
    Label("ground",                 6, -1, 255, "void",         0, False, True,  True,  (81, 0, 81)),
    Label("road",                   7,  1,   0, "flat",         1, False, False, False, (128, 64, 128)),
    Label("sidewalk",               8,  3,   1, "flat",         1, False, False, False, (244, 35, 232)),
    Label("parking",                9,  2, 255, "flat",         1, False, True,  True,  (250, 170, 160)),
    Label("rail track",            10, 10, 255, "flat",         1, False, True,  True,  (230, 150, 140)),
    Label("building",              11, 11,   2, "construction", 2, True,  False, False, (70, 70, 70)),
    Label("wall",                  12,  7,   3, "construction", 2, False, False, False, (102, 102, 156)),
    Label("fence",                 13,  8,   4, "construction", 2, False, False, False, (190, 153, 153)),
    Label("guard rail",            14, 30, 255, "construction", 2, False, True,  True,  (180, 165, 180)),
    Label("bridge",                15, 31, 255, "construction", 2, False, True,  True,  (150, 100, 100)),
    Label("tunnel",                16, 32, 255, "construction", 2, False, True,  True,  (150, 120, 90)),
    Label("pole",                  17, 21,   5, "object",       3, True,  False, True,  (153, 153, 153)),
    Label("polegroup",             18, -1, 255, "object",       3, False, True,  True,  (153, 153, 153)),
    Label("traffic light",         19, 23,   6, "object",       3, True,  False, True,  (250, 170, 30)),
    Label("traffic sign",          20, 24,   7, "object",       3, True,  False, True,  (220, 220, 0)),
    Label("vegetation",            21,  5,   8, "nature",       4, False, False, False, (107, 142, 35)),
    Label("terrain",               22,  4,   9, "nature",       4, False, False, False, (152, 251, 152)),
    Label("sky",                   23,  9,  10, "sky",          5, False, False, False, (70, 130, 180)),
    Label("person",                24, 19,  11, "human",        6, True,  False, False, (220, 20, 60)),
    Label("rider",                 25, 20,  12, "human",        6, True,  False, False, (255, 0, 0)),
    Label("car",                   26, 13,  13, "vehicle",      7, True,  False, False, (0, 0, 142)),
    Label("truck",                 27, 14,  14, "vehicle",      7, True,  False, False, (0, 0, 70)),
    Label("bus",                   28, 34,  15, "vehicle",      7, True,  False, False, (0, 60, 100)),
    Label("caravan",               29, 16, 255, "vehicle",      7, True,  True,  True,  (0, 0, 90)),
    Label("trailer",               30, 15, 255, "vehicle",      7, True,  True,  True,  (0, 0, 110)),
    Label("train",                 31, 33,  16, "vehicle",      7, True,  False, False, (0, 80, 100)),
    Label("motorcycle",            32, 17,  17, "vehicle",      7, True,  False, False, (0, 0, 230)),
    Label("bicycle",               33, 18,  18, "vehicle",      7, True,  False, False, (119, 11, 32)),
    Label("garage",                34, 12,   2, "construction", 2, True,  True,  True,  (64, 128, 128)),
    Label("gate",                  35,  6,   4, "construction", 2, False, True,  True,  (190, 153, 153)),
    Label("stop",                  36, 29, 255, "construction", 2, True,  True,  True,  (150, 120, 90)),
    Label("smallpole",             37, 22,   5, "object",       3, True,  True,  True,  (153, 153, 153)),
    Label("lamp",                  38, 25, 255, "object",       3, True,  True,  True,  (0, 64, 64)),
    Label("trash bin",             39, 26, 255, "object",       3, True,  True,  True,  (0, 128, 192)),
    Label("vending machine",       40, 27, 255, "object",       3, True,  True,  True,  (128, 64, 0)),
    Label("box",                   41, 28, 255, "object",       3, True,  True,  True,  (64, 64, 128)),
    Label("unknown construction",  42, 35, 255, "void",         0, False, True,  True,  (102, 0, 0)),
    Label("unknown vehicle",       43, 36, 255, "void",         0, False, True,  True,  (51, 0, 51)),
    Label("unknown object",        44, 37, 255, "void",         0, False, True,  True,  (32, 32, 32)),
    Label("license plate",         -1, -1,  -1, "vehicle",      7, False, True,  True,  (0, 0, 142)),
)
# fmt: on

CITYSCAPES_MAX_ID = 33  # Cityscapes has the ids up to bicycle; KITTI-360 adds the rest


def label_lut(max_id: int | None = None):
    """uint8 table label id -> train id of the labels with an id up to ``max_id`` (None: all); others are ignored."""
    return train_id_lut(
        {
            label.id: label.trainId
            for label in LABELS
            if label.id >= 0 and label.trainId not in (-1, IGNORE_LABEL) and (max_id is None or label.id <= max_id)
        }
    )


def train_class_names() -> tuple[str, ...]:
    """Name of every train id: the first label (by id) that has it."""
    names = {}
    for label in LABELS:
        if label.trainId not in (-1, IGNORE_LABEL):
            names.setdefault(label.trainId, label.name)
    return tuple(names[i] for i in range(len(names)))


SPLIT_SIZES = {"train": 2975, "val": 500}


class CityscapesDataset(SegmentationDataset):
    """Images and label-id maps listed by torchvision's ``Cityscapes`` (``root/leftImg8bit``, ``root/gtFine``)."""

    class_names = train_class_names()

    def __init__(self, root: str, split: str = "train", **kwargs):
        from torchvision.datasets import Cityscapes

        super().__init__(**kwargs)
        self.root = root
        self.split = split
        self.label_lut = label_lut(CITYSCAPES_MAX_ID)
        cityscapes = Cityscapes(root=root, split=split, mode="fine", target_type="semantic")
        self.samples = [
            (image, targets[0]) for image, targets in zip(cityscapes.images, cityscapes.targets, strict=True)
        ]
        check_split_size("Cityscapes", split, len(self), SPLIT_SIZES)


class HFCityscapesDataset(SegmentationDataset):
    """Cityscapes from a Hub dataset with ``id``, ``image`` and (label-id, mode L PNG) ``mask`` columns.

    The default repo is private: run ``huggingface-cli login`` with an account that can read it. ``split="val"`` also
    matches a split named "validation". Downloaded on first use and cached under ``cache_dir`` (default:
    ``$HF_HOME/datasets``).
    """

    class_names = train_class_names()

    def __init__(
        self,
        repo: str = DEFAULT_HF_REPO,
        split: str = "train",
        name: str | None = "semantic",
        cache_dir: str | None = None,
        image_key: str = "image",
        mask_key: str = "mask",
        id_key: str = "id",
        num_proc: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_lut = label_lut(CITYSCAPES_MAX_ID)
        self.repo_id, self.split, self.data = load_hub_split(repo, split, name, cache_dir, num_proc)
        if not self.include_labels:
            # Rows decode every image column they hold: drop the masks rather than decode them for nothing
            self.data = self.data.remove_columns(mask_key)
        self.image_key, self.mask_key, self.id_key = image_key, mask_key, id_key
        if name == "semantic":
            check_split_size("Cityscapes", split, len(self), SPLIT_SIZES)

    def load(self, index):
        item = self.data[index]
        mask = item.get(self.mask_key)
        return item[self.image_key].convert("RGB"), mask, f"{self.repo_id}/{self.split}/{item.get(self.id_key, index)}"

    def __len__(self):
        return len(self.data)
