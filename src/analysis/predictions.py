"""Predictions of the probed upsamplers on the val split, computed once (``analysis=predictions``) and read back by the
analyses that only need predictions (``error_breakdown``) or from a notebook (``load_predictions``).

A cache folder, ``output/predictions/<task>_<dataset>_<backbone short name>/`` (named like the probes, see
``src.evaluation.probing.registry.probe_name``), holds for the first N images of the val split, at the probes' size S:

- ``labels.npy``: ground truth, [N, S, S] uint8 (255: void);
- ``<model>.npy``: probe classes of the features upsampled to S x S, [N, S, S] uint8;
- ``<model>_lowres.npy``: probe classes of the low-res backbone features, [N, h, w] uint8;
- ``meta.json``: what the predictions depend on (probe files with their size and modification time, dataset config,
  S, N), written last. Reading checks it against the registered probes, so a cache made with other probes raises
  instead of giving stale numbers; a folder without it (an interrupted run) is no cache.

The arrays are memory-mapped when read: an analysis reads the images it needs, not the whole cache.
"""

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from src.dataset.transforms import get_batch, to_float_image
from src.evaluation.probing.registry import find_probe, probe_name
from src.utils.checkpoint import ROOT, WEIGHTS_DIR
from src.utils.run import launch_path

from .pixels import BUCKETS, boundary_distance, own_class_fraction, region_width_area
from .shared import ValSplit, load_sample, load_setup, load_val_split, model_predictions, predict, window_oracle

__all__ = [
    "PREDICTIONS_DIR",
    "PredictionSource",
    "Predictions",
    "cache_dir",
    "load_predictions",
    "oracle_prediction",
    "PIXEL_SUBSETS",
    "per_image_accuracy",
    "pixel_subsets",
    "prediction_source",
    "run_predictions",
    "write_predictions",
]

PREDICTIONS_DIR = os.path.join(ROOT, "output", "predictions")
VOID = 255  # label of the void pixels, so at most 255 classes fit in uint8
# Pixels "near an edge": closer than this to a boundary, error_breakdown's bands [0,2) and [2,4)
EDGE_DISTANCE = 4

# Pixel subsets of the per-image accuracies (name -> what they are), the first bucket of error_breakdown's groupings
PIXEL_SUBSETS = {
    "acc": "all labeled pixels",
    "edge_acc": "near edges (< 4 px from a boundary)",
    "thin_acc": "thin structures (regions < 8 px wide)",
    "minority_acc": "minority pixels (class < 25% of their patch)",
    "small_acc": "small regions (< 32x32 px)",
}


def cache_dir(probes, root=PREDICTIONS_DIR):
    """Cache folder of the predictions of the probes ``probes`` (``task``, ``dataset``, ``backbone``)."""
    return os.path.join(root, probe_name(probes.task, probes.dataset, probes.backbone))


def probe_record(probes, model):
    """What identifies the probe registered for ``model``: its file (relative to ``WEIGHTS_DIR``), size and mtime."""
    path = find_probe(probes.task, probes.dataset, probes.backbone, model)
    stat = os.stat(path)
    return {
        "file": os.path.relpath(path, WEIGHTS_DIR).replace(os.sep, "/"),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


# ---- writing -------------------------------------------------------------------------------------------------------


def write_predictions(setup, probes, out, device):
    """Predict the first ``setup.num_images`` val images with every model of ``setup`` and write them to ``out``."""
    if setup.num_classes > VOID:
        raise ValueError(f"{setup.num_classes} classes don't fit in uint8 next to the void label {VOID}")
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)  # the folder is no valid cache until the new predictions are complete
    n, size = setup.num_images, setup.size
    arrays = {"labels": np.lib.format.open_memmap(os.path.join(out, "labels.npy"), "w+", np.uint8, (n, size, size))}
    for i, _, label, preds, lr_preds in model_predictions(setup, device):
        if i == 0:
            for name in preds:
                path = os.path.join(out, f"{name}.npy")
                arrays[name] = np.lib.format.open_memmap(path, "w+", np.uint8, (n, size, size))
                path = os.path.join(out, f"{name}_lowres.npy")
                arrays[f"{name}_lowres"] = np.lib.format.open_memmap(path, "w+", np.uint8, (n, *lr_preds[name].shape))
        arrays["labels"][i] = label.cpu().numpy()
        for name in preds:
            arrays[name][i] = preds[name].to(torch.uint8).cpu().numpy()
            arrays[f"{name}_lowres"][i] = lr_preds[name].to(torch.uint8).cpu().numpy()
    for array in arrays.values():
        array.flush()
    del arrays

    first = next(iter(setup.runs.values())).meta
    meta = {
        "task": probes.task,
        "dataset": probes.dataset,
        "backbone": first["backbone"],
        "models": list(setup.runs),
        "probes": {name: probe_record(probes, name) for name in setup.runs},
        "dataset_cfg": first["dataset_cfg"],
        "size": size,
        "lr_size": list(lr_preds[name].shape),
        "num_images": n,
        "split_size": len(setup.dataset),
        "num_classes": setup.num_classes,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return meta


@torch.inference_mode()
def run_predictions(cfg, run):
    """Predict the val split with the upsamplers ``cfg.analysis.models`` probed as ``cfg.analysis.probes`` and cache
    the predictions in ``cfg.analysis.out`` (default: ``cache_dir``)."""
    acfg, device = cfg.analysis, run.device
    out = launch_path(acfg.get("out")) or cache_dir(acfg.probes)
    setup = load_setup(acfg.probes, list(acfg.models), device, acfg.get("max_images"))
    meta = write_predictions(setup, acfg.probes, out, device)
    run.console.print(
        f"[bold green]Predictions of {meta['models']} on {meta['num_images']} images in {out}[/bold green]"
    )
    return meta


# ---- reading -------------------------------------------------------------------------------------------------------


@dataclass
class Predictions:
    """A prediction cache, its arrays memory-mapped (see the module docstring)."""

    dir: str
    meta: dict
    labels: np.ndarray  # [N, S, S] uint8
    preds: dict[str, np.ndarray]  # model -> [N, S, S] uint8
    lowres: dict[str, np.ndarray]  # model -> [N, h, w] uint8

    @property
    def num_images(self):
        return self.meta["num_images"]


def check_predictions(meta, probes, models, max_images):
    """Raise unless the cache ``meta`` holds the predictions of the probes currently registered for ``models`` on the
    first ``max_images`` images (None: the whole split)."""
    regenerate = "regenerate it with `python analysis.py analysis=predictions`"
    missing = [m for m in models if m not in meta["models"]]
    if missing:
        raise ValueError(f"Prediction cache has no {missing} (has {meta['models']}); {regenerate}")
    wanted = meta["split_size"] if max_images is None else min(max_images, meta["split_size"])
    if meta["num_images"] < wanted:
        raise ValueError(f"Prediction cache holds {meta['num_images']} images, {wanted} wanted; {regenerate}")
    for model in models:
        if probe_record(probes, model) != meta["probes"][model]:
            raise ValueError(f"The probe of {model!r} changed since the prediction cache was made; {regenerate}")


def load_predictions(directory, models=None, probes=None, max_images=None):
    """The prediction cache in ``directory``, for ``models`` (None: all of them).

    With ``probes`` (``task``, ``dataset``, ``backbone``), it is first checked against the registered probes and must
    hold the first ``max_images`` images (None: the whole split); raises otherwise.
    """
    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)
    models = list(meta["models"] if models is None else models)
    if probes is not None:
        check_predictions(meta, probes, models, max_images)

    def array(name):
        return np.load(os.path.join(directory, f"{name}.npy"), mmap_mode="r")

    return Predictions(
        directory,
        meta,
        array("labels"),
        {m: array(m) for m in models},
        {m: array(f"{m}_lowres") for m in models},
    )


def oracle_prediction(predictions, kernel, index, reference="bilinear"):
    """The window oracle ``oracle_<kernel>x<kernel>`` of image ``index`` (``shared.window_oracle``, as
    ``error_breakdown`` computes it), from the cached labels and low-res / upsampled predictions of ``reference``:
    [S, S] uint8, on the CPU."""

    def read(array):
        return torch.from_numpy(np.array(array[index]))

    label = read(predictions.labels)
    lr_pred, fallback = read(predictions.lowres[reference]).long(), read(predictions.preds[reference]).long()
    return window_oracle(lr_pred, label, fallback, kernel).to(torch.uint8).numpy()


def pixel_subsets(label, num_classes, lr_size):
    """Masks of the labeled pixels of a label map ([H, W] tensor, on the CPU) in every subset of ``PIXEL_SUBSETS``,
    defined as error_breakdown's groups (``pixels``): dict name -> [H, W] bool."""
    valid = label != VOID
    width, area = (torch.from_numpy(a) for a in region_width_area(label.numpy()))
    own_frac = own_class_fraction(label[None], num_classes, lr_size)[0]
    return {
        "acc": valid,
        "edge_acc": valid & (boundary_distance(label[None])[0] < EDGE_DISTANCE),
        "thin_acc": valid & (width < BUCKETS["width"][0][0]),
        "minority_acc": valid & (own_frac < BUCKETS["own_frac"][0][0]),
        "small_acc": valid & (area < BUCKETS["area"][0][0]),
    }


def per_image_accuracy(predictions, models=None):
    """Accuracy of every cached image for every model (None: all) over every pixel subset of ``PIXEL_SUBSETS``:
    ``<subset>_<model>`` (``acc_naf``, ``thin_acc_naf``, ...; NaN without such pixels), and the number of pixels of
    each subset ``n_<subset>``; [N] arrays. ``acc`` and ``edge_acc`` are error_breakdown's per_image.npz."""
    models = list(predictions.preds if models is None else models)
    n, meta = predictions.num_images, predictions.meta
    out = {f"{subset}_{m}": np.empty(n) for subset in PIXEL_SUBSETS for m in models}
    out |= {f"n_{subset}": np.empty(n, dtype=np.int64) for subset in PIXEL_SUBSETS}
    for i in range(n):
        label = torch.from_numpy(np.array(predictions.labels[i]))
        subsets = pixel_subsets(label, meta["num_classes"], tuple(meta["lr_size"]))
        lab = label.long()
        for subset, mask in subsets.items():
            out[f"n_{subset}"][i] = int(mask.sum())
        for m in models:
            correct = torch.from_numpy(np.array(predictions.preds[m][i])).long() == lab
            for subset, mask in subsets.items():
                out[f"{subset}_{m}"][i] = correct[mask].float().mean().item()
    return out


# ---- predictions for an analysis: cached or computed ---------------------------------------------------------------


@dataclass
class PredictionSource:
    """The predictions an analysis reads, whether cached or computed by the models.

    ``batches()`` yields (index, image [1, 3, H, W] in [0, 1], label [H, W], predictions, low-res predictions) of
    every analyzed image, predictions as dicts model -> [H, W] / [h, w] long on the device; ``sample(index)`` gives
    (image, label, predictions as uint8 / int64 arrays) of one image.
    """

    models: list[str]
    split: ValSplit
    num_classes: int
    cache_dir: str | None  # the prediction cache read, None when the models run
    batches: Callable[[], Iterator[tuple]]
    sample: Callable[[int], tuple]


def cached_source(cache, split, device):
    """``PredictionSource`` reading the predictions from ``cache``; images and labels come from the dataset."""

    def read(arrays, index):
        return {m: torch.from_numpy(np.array(a[index])).to(device).long() for m, a in arrays.items()}

    def batches():
        for i, batch in enumerate(split.loader):
            if i == split.num_images:
                break
            batch = get_batch(batch, device)
            label = batch["label"][0]
            if i == 0 and not np.array_equal(label.cpu().numpy(), cache.labels[0]):
                raise ValueError(f"The labels of {cache.dir} don't match the dataset; regenerate the cache")
            yield i, batch["image"], label, read(cache.preds, i), read(cache.lowres, i)

    def sample(index):
        item = split.dataset[index]
        image = to_float_image(item["image"][None].to(device))
        return image, item["label"].numpy(), {m: np.asarray(a[index]) for m, a in cache.preds.items()}

    return PredictionSource(list(cache.preds), split, cache.meta["num_classes"], cache.dir, batches, sample)


def model_source(setup, device):
    """``PredictionSource`` computing the predictions with the probed models of ``setup``."""
    split = ValSplit(setup.dataset, setup.loader, setup.num_images, setup.size)

    def batches():
        yield from model_predictions(setup, device)

    def sample(index):
        image, label, img_ups, lr = load_sample(setup, index, device)
        preds = {m: predict(run, img_ups, lr, setup.size).cpu().numpy() for m, run in setup.runs.items()}
        return image, label.cpu().numpy(), preds

    return PredictionSource(list(setup.runs), split, setup.num_classes, None, batches, sample)


def prediction_source(probes, models, device, max_images=None, cache="auto"):
    """Predictions of ``models`` probed as ``probes`` on the first ``max_images`` val images (None: all).

    ``cache``: "auto" reads ``cache_dir(probes)`` when it holds a cache (which must then match, see
    ``check_predictions``) and runs the models otherwise; a folder reads that cache; False always runs the models.
    """
    directory = cache_dir(probes) if cache == "auto" else (launch_path(cache) if cache else None)
    if directory and (cache != "auto" or os.path.exists(os.path.join(directory, "meta.json"))):
        predictions = load_predictions(directory, models, probes, max_images)
        split = load_val_split(predictions.meta["dataset_cfg"], predictions.meta["size"], max_images)
        return cached_source(predictions, split, device)
    return model_source(load_setup(probes, models, device, max_images), device)
