"""Datasets on the Hugging Face Hub. ``datasets`` is imported inside the functions, so local data works without it."""

from .common import split_candidates

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
