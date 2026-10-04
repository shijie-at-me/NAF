"""Where dataset files come from: split names and sizes, local file listings, and the Hugging Face Hub.

``datasets`` is imported inside the Hub functions, so local data works without it installed."""

import os

IGNORE_LABEL = 255  # train id of the pixels left out of training and evaluation

# Split names that the training code asks for but some datasets name differently
SPLIT_ALIASES = {"val": ("validation", "valid"), "validation": ("val", "valid")}


def split_candidates(split: str) -> tuple[str, ...]:
    """``split`` followed by its aliases (``val`` -> ``validation``, ``valid``)."""
    return (split, *SPLIT_ALIASES.get(split, ()))


def resolve_local_split(root: str, split: str) -> str:
    """``root/split``, or ``root/<alias>`` if only an alias exists (``val`` -> ``validation`` / ``valid``)."""
    for candidate in split_candidates(split):
        path = os.path.join(root, candidate)
        if os.path.isdir(path):
            return path
    available = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    raise ValueError(f"Split '{split}' not found in {root}, available: {available}")


def check_split_size(name: str, split: str, size: int, expected: dict[str, int]):
    """Fail early on an incomplete download or a wrong root: the standard splits have a known size."""
    if split in expected and size != expected[split]:
        raise RuntimeError(f"{name} {split} split has {size} images, expected {expected[split]}")


def read_lines(path: str) -> list[str]:
    """Non-empty lines of a text file (image ids, video names), stripped."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def relative_posix(path: str, root: str) -> str:
    """``path`` relative to ``root`` with "/" separators, so listings written on Windows also work on Linux."""
    return os.path.relpath(path, root).replace(os.sep, "/")


HF_DATASET_URL = "https://huggingface.co/datasets/"


def to_repo_id(repo: str) -> str:
    """Accept a Hub repo id (``ILSVRC/imagenet-1k``) or its URL (``https://huggingface.co/datasets/ILSVRC/imagenet-1k``)."""
    return repo.removeprefix(HF_DATASET_URL).split("/tree/")[0].strip("/")


def resolve_split(repo_id: str, split: str, name: str | None = None) -> str:
    """Return ``split`` if the dataset has it, else the first alias it has (``val`` -> ``validation`` / ``valid``)."""
    from datasets import get_dataset_split_names

    available = get_dataset_split_names(repo_id, name)
    for candidate in split_candidates(split):
        if candidate in available:
            return candidate
    raise ValueError(f"Split '{split}' not found in {repo_id}, available: {available}")


def load_hub_split(repo: str, split: str, name: str | None = None, cache_dir=None, num_proc=None):
    """``(repo id, resolved split, datasets.Dataset)`` of a split of a Hub dataset given as a repo id or URL.

    Downloaded on first use and cached under ``cache_dir`` (default: ``$HF_HOME/datasets``).
    """
    from datasets import load_dataset

    repo_id = to_repo_id(repo)
    split = resolve_split(repo_id, split, name)
    return repo_id, split, load_dataset(repo_id, name, split=split, cache_dir=cache_dir, num_proc=num_proc)
