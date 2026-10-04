"""ADE20K (SceneParsing, ``ADEChallengeData2016``): 150 classes, plus class 0 for "other" pixels.

From a local ``ADEChallengeData2016`` folder, or a dataset on the Hugging Face Hub. ``datasets`` is imported only by
the Hub code.
"""

import os

from .segmentation import SegmentationDataset, train_id_lut
from .sources import check_split_size, load_hub_split

DEFAULT_HF_REPO = "shijli/ade20k"

# fmt: off
CLASS_NAMES = (
    "background", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence",
    "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path",
    "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge", "bookcase",
    "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm",
    "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light",
    "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver", "airplane",
    "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
    "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
)
# fmt: on
SPLIT_DIRS = {"train": "training", "val": "validation"}
SPLIT_SIZES = {"train": 20210, "val": 2000}


class ADE20KDataset(SegmentationDataset):
    """``root/images/<split dir>/*.jpg`` with the labels in ``root/annotations/<split dir>/*.png``.

    ``file_set`` restricts the dataset to these image ids; ``skip_other_class`` makes class 0 ("other") ignored.
    """

    class_names = CLASS_NAMES

    def __init__(self, root: str, split: str = "train", skip_other_class: bool = False, file_set=None, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.split = split
        if skip_other_class:
            self.label_lut = train_id_lut({i: i for i in range(1, 256)})
        image_dir = os.path.join(root, "images", SPLIT_DIRS[split])
        label_dir = os.path.join(root, "annotations", SPLIT_DIRS[split])
        if file_set is None:
            images, labels = sorted(os.listdir(image_dir)), sorted(os.listdir(label_dir))
        else:
            images, labels = [f"{f}.jpg" for f in sorted(file_set)], [f"{f}.png" for f in sorted(file_set)]
        self.samples = [
            (os.path.join(image_dir, i), os.path.join(label_dir, a)) for i, a in zip(images, labels, strict=True)
        ]
        if file_set is None:
            check_split_size("ADE20K", split, len(self), SPLIT_SIZES)


class HFADE20KDataset(SegmentationDataset):
    """ADE20K from a Hub dataset with ``id``, ``image`` and (mode L PNG) ``mask`` columns, given as a repo id or URL.

    ``name`` is the config ("segmentation": 20210 / 2000 images); ``file_set`` and ``skip_other_class`` as in
    ``ADE20KDataset``. ``split="val"`` also matches a split named "validation". Downloaded on first use and cached
    under ``cache_dir`` (default: ``$HF_HOME/datasets``).
    """

    class_names = CLASS_NAMES

    def __init__(
        self,
        repo: str = DEFAULT_HF_REPO,
        split: str = "train",
        name: str | None = "segmentation",
        skip_other_class: bool = False,
        file_set=None,
        cache_dir: str | None = None,
        image_key: str = "image",
        mask_key: str = "mask",
        id_key: str = "id",
        num_proc: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if skip_other_class:
            self.label_lut = train_id_lut({i: i for i in range(1, 256)})
        self.repo_id, self.split, self.data = load_hub_split(repo, split, name, cache_dir, num_proc)
        if file_set is not None:
            # select by the id column alone, so the filter decodes no image
            wanted = set(file_set)
            self.data = self.data.select([i for i, f in enumerate(self.data[id_key]) if f in wanted])
        if not self.include_labels:
            # Rows decode every image column they hold: drop the masks rather than decode them for nothing
            self.data = self.data.remove_columns(mask_key)
        self.image_key, self.mask_key, self.id_key = image_key, mask_key, id_key
        if file_set is None and name == "segmentation":
            check_split_size("ADE20K", split, len(self), SPLIT_SIZES)

    def load(self, index):
        item = self.data[index]
        mask = item.get(self.mask_key)
        return item[self.image_key].convert("RGB"), mask, f"{self.repo_id}/{self.split}/{item.get(self.id_key, index)}"

    def __len__(self):
        return len(self.data)
