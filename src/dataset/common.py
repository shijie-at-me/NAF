"""Helpers shared by the datasets: split names, split sizes and listing files."""

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
