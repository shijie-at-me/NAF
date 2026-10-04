"""Image datasets for upsampler training: a local class-per-folder tree, or a dataset on the Hugging Face Hub.

Both return the same batch: ``{"index", "image", "target", "path"}`` (+ ``"label"`` with ``include_labels``).
``datasets`` is imported only by the Hub code, so local folders work without it installed.
"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import folder

from .common import relative_posix, resolve_local_split
from .hub import load_hub_split

DEFAULT_HF_REPO = "ILSVRC/imagenet-1k"


# --- local folder ---------------------------------------------------------------------------------------------------


def read_sample_list(cache_path: str) -> list[tuple[str, int]]:
    """Read ``(relative path, class index)`` pairs from a ``path;idx`` per line listing."""
    with open(cache_path) as f:
        return [(path, int(idx)) for path, idx in (line.strip().split(";") for line in f)]


def write_sample_list(cache_path: str, samples: list[tuple[str, int]]):
    with open(cache_path, "w") as f:
        f.writelines(f"{path};{idx}\n" for path, idx in samples)


def list_samples(root: str, class_to_idx: dict, extensions, is_valid_file, cache_path: str) -> list[tuple[str, int]]:
    """``(path relative to root, class index)`` for every image, from ``cache_path`` if it exists, else by walking
    ``root`` once and caching the result there (listing ImageNet takes minutes on a network file system)."""
    if os.path.isfile(cache_path):
        print(f"Using directory list at: {cache_path}")
        return read_sample_list(cache_path)

    print(f"Walking directory: {root}")
    samples = folder.make_dataset(root, class_to_idx, extensions, is_valid_file)
    samples = [(relative_posix(path, root), idx) for path, idx in samples]
    write_sample_list(cache_path, samples)
    return samples


# --- datasets -------------------------------------------------------------------------------------------------------


class BaseImageDataset(Dataset):
    """Applies the transform and builds the batch; subclasses only say how to load sample ``index``.

    ``kwargs`` swallow what the training code passes to every dataset but doesn't apply to class labels,
    such as the dense-label ``target_transform``.
    """

    def __init__(self, transform=None, include_labels: bool = False, **kwargs):
        self.transform = transform
        self.include_labels = include_labels

    def load(self, index: int) -> tuple[Image.Image, int, str]:
        """Return ``(RGB image, class index, path or id)`` of sample ``index``."""
        raise NotImplementedError

    def __getitem__(self, index):
        image, target, path = self.load(index)
        if self.transform is not None:
            image = self.transform(image)

        batch = {"index": index, "image": image, "target": target, "path": path}
        if self.include_labels:
            batch["label"] = target
        return batch


class ImageDataset(BaseImageDataset):
    """Class-per-folder image tree (``<image dir>/<class>/<image>``) as in torchvision's ``ImageFolder``.

    The image dir is ``root`` itself, or ``root/<split>`` when ``split`` is given (aliases such as ``val`` ->
    ``validation`` are resolved). Its file list is cached as ``<image dir name>.txt`` in ``root_cache``
    (default: next to the image dir) so later runs skip the directory walk.
    """

    def __init__(
        self,
        root: str,
        split: str | None = None,
        root_cache: str | None = None,
        loader=folder.default_loader,
        extensions=folder.IMG_EXTENSIONS,
        is_valid_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if split is not None:
            root = resolve_local_split(root, split)
        self.root = root
        self.split = split
        self.loader = loader
        self.extensions = extensions
        self.classes, self.class_to_idx = folder.find_classes(root)

        image_dir = os.path.normpath(root)
        cache_path = os.path.join(root_cache or os.path.dirname(image_dir), os.path.basename(image_dir) + ".txt")
        self.samples = list_samples(root, self.class_to_idx, extensions, is_valid_file, cache_path)
        if not self.samples:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {root}\nSupported extensions are: {','.join(extensions)}"
            )
        self.targets = [idx for _, idx in self.samples]

    def load(self, index):
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        return self.loader(path), target, path

    def __len__(self):
        return len(self.samples)


class HFImageDataset(BaseImageDataset):
    """Image dataset from the Hugging Face Hub, given as a repo id or URL (default: ImageNet-1k).

    Downloaded on first use and cached under ``cache_dir`` (default: ``$HF_HOME/datasets``).
    Gated datasets such as ImageNet need ``huggingface-cli login`` and the terms accepted on the Hub page.
    """

    def __init__(
        self,
        repo: str = DEFAULT_HF_REPO,
        split: str = "train",
        name: str | None = None,
        cache_dir: str | None = None,
        image_key: str = "image",
        label_key: str = "label",
        num_proc: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repo_id, self.split, self.data = load_hub_split(repo, split, name, cache_dir, num_proc)
        self.image_key = image_key
        self.label_key = label_key

    def load(self, index):
        item = self.data[index]
        return item[self.image_key].convert("RGB"), item.get(self.label_key, -1), f"{self.repo_id}/{self.split}/{index}"

    def __len__(self):
        return len(self.data)
