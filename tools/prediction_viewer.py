"""Web UI to browse the cached predictions of the probed upsamplers (``python analysis.py analysis=predictions``).

    streamlit run tools/prediction_viewer.py [-- --predictions output/predictions]

Lists the prediction caches under ``--predictions`` and, for the chosen one, every val image with each model's
accuracy over a pixel subset (all labeled pixels, near edges, thin structures, minority pixels or small regions, as the
error breakdown groups them), sorted by a model or by the difference between two models. The images of a page are drawn with the ground truth and each model's prediction or errors; window oracles
(``oracle_<k>x<k>``, as the error breakdown defines them) can be drawn too, computed from the cached bilinear
predictions. A cache made with probes that changed since is flagged.

"Show features" adds under every image its features (PCA colors of the backbone's low-res features and of each
model's upsampled ones, one PCA per image): these are not cached, the models compute them, so the first time loads
them on the GPU.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

VOID = 255
ORACLES = ("oracle_1x1", "oracle_3x3", "oracle_5x5", "oracle_9x9")
ORACLE_NAME = re.compile(r"oracle_(?P<k>\d+)x(?P=k)")
# Short tag of each pixel subset (src.analysis.predictions.PIXEL_SUBSETS) in the image captions
SUBSET_TAGS = {"acc": "", "edge_acc": "edge", "thin_acc": "thin", "minority_acc": "minority", "small_acc": "small"}


def parse_args():
    parser = argparse.ArgumentParser(description="Browse cached predictions")
    parser.add_argument(
        "--predictions", default=os.path.join(ROOT, "output", "predictions"), help="folder of the prediction caches"
    )
    return parser.parse_args(sys.argv[1:])


# ---- data ----------------------------------------------------------------------------------------------------------


@st.cache_data
def find_caches(root):
    """meta.json of every prediction cache under ``root``: dict folder -> meta, newest first."""
    paths = sorted(glob.glob(os.path.join(root, "*", "meta.json")), key=os.path.getmtime, reverse=True)
    caches = {}
    for path in paths:
        with open(path) as f:
            caches[os.path.dirname(path)] = json.load(f)
    return caches


@st.cache_resource
def open_cache(cache_dir):
    """Memory-mapped labels and predictions of a prediction cache, and its meta."""
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    labels = np.load(os.path.join(cache_dir, "labels.npy"), mmap_mode="r")
    preds = {m: np.load(os.path.join(cache_dir, f"{m}.npy"), mmap_mode="r") for m in meta["models"]}
    return meta, labels, preds


def load_predictions(cache_dir):
    from src.analysis.predictions import load_predictions as load

    return load(cache_dir)


@st.cache_resource
def accuracy_code_version():
    """Hash of the code ``per_image_table`` runs: its disk cache is keyed by its own source only, so a change of the
    accuracies' definition must change one of its arguments."""
    import hashlib
    import inspect

    from src.analysis import pixels, predictions

    sources = [inspect.getsource(predictions.per_image_accuracy), inspect.getsource(predictions.pixel_subsets)]
    sources += [inspect.getsource(pixels), repr(predictions.PIXEL_SUBSETS), repr(predictions.EDGE_DISTANCE)]
    return hashlib.sha1("".join(sources).encode()).hexdigest()


@st.cache_data(persist="disk", show_spinner="Computing the per-image accuracies (once per cache and code version)...")
def per_image_table(cache_dir, meta_mtime, code_version):
    """``<subset>_<model>`` accuracies and ``n_<subset>`` pixel counts of every cached image
    (``src.analysis.predictions.per_image_accuracy``), indexed by the val image index; ``meta_mtime`` invalidates it
    when the cache is regenerated, ``code_version`` (``accuracy_code_version``) when the accuracies' code changes."""
    from src.analysis.predictions import per_image_accuracy

    table = pd.DataFrame(per_image_accuracy(load_predictions(cache_dir)))
    table.index.name = "image"
    return table


@st.cache_data
def cache_problem(cache_dir, meta_mtime):
    """Why the cache doesn't hold the current predictions of its probes (``check_predictions``), or None."""
    from types import SimpleNamespace

    from src.analysis.predictions import check_predictions

    meta, _, _ = open_cache(cache_dir)
    probes = SimpleNamespace(task=meta["task"], dataset=meta["dataset"], backbone=meta["backbone"])
    try:
        check_predictions(meta, probes, meta["models"], meta["num_images"])
    except (ValueError, FileNotFoundError) as e:
        return str(e)
    return None


@st.cache_data(max_entries=512)
def oracle_prediction(cache_dir, model, index):
    """The window oracle ``model`` (``oracle_<k>x<k>``) of image ``index``, computed from the cached bilinear
    predictions (``src.analysis.predictions.oracle_prediction``)."""
    from src.analysis.predictions import oracle_prediction as compute

    return compute(load_predictions(cache_dir), int(ORACLE_NAME.fullmatch(model)["k"]), index)


def has_oracles(cache_dir, meta):
    return "bilinear" in meta["models"] and os.path.exists(os.path.join(cache_dir, "bilinear_lowres.npy"))


# The probe whose classes of the low-res backbone features are shown: bilinear upsampling only averages those
# features, so its probe reads them in the space it was trained in
LOWRES_PROBE = "bilinear"


@st.cache_resource
def open_lowres(cache_dir):
    """Memory-mapped classes of the low-res backbone features under ``LOWRES_PROBE``'s probe, or None."""
    path = os.path.join(cache_dir, f"{LOWRES_PROBE}_lowres.npy")
    return np.load(path, mmap_mode="r") if os.path.exists(path) else None


def lowres_prediction(cache_dir, index, size):
    """The low-res classes of image ``index`` enlarged to size x size with nearest neighbors (each token covers its
    patch), with the low-res grid size; None without a cached low-res prediction."""
    lowres = open_lowres(cache_dir)
    if lowres is None:
        return None, None
    grid = np.asarray(lowres[index])
    enlarged = np.asarray(Image.fromarray(grid).resize((size, size), Image.NEAREST))
    return enlarged, grid.shape


@st.cache_resource
def open_dataset(dataset_cfg_json, size):
    """The val split the predictions were made on, images and labels at ``size``, and its class names."""
    from omegaconf import OmegaConf

    import hydra_plugins.resolvers  # noqa: F401  (config resolvers)
    from src.dataset.loading import build_dataset
    from src.dataset.transforms import build_transforms
    from src.utils.run import expand_user_paths

    cfg = expand_user_paths(OmegaConf.create(json.loads(dataset_cfg_json)))
    dataset = build_dataset(cfg, build_transforms(size, size), split="val")
    return dataset, dataset.get_class_names()


@st.cache_resource
def palette():
    from src.utils.visualization import pascal_palette

    return pascal_palette()


@st.cache_resource
def open_raw_dataset(dataset_cfg_json):
    """The same val split without transforms: the images and labels at their original size."""
    from omegaconf import OmegaConf

    import hydra_plugins.resolvers  # noqa: F401  (config resolvers)
    from src.dataset.loading import build_dataset
    from src.utils.run import expand_user_paths

    cfg = expand_user_paths(OmegaConf.create(json.loads(dataset_cfg_json)))
    return build_dataset(cfg, {"image": None, "label": None}, split="val")


def crop_box(width, height, size):
    """The square the predictions cover, in original pixels: torchvision's ``Resize(size)`` (short side to ``size``)
    then ``CenterCrop(size)``, as ``src.dataset.transforms.resize_crop``; (left, top, right, bottom)."""
    short, long = min(width, height), max(width, height)
    resized_long = int(size * long / short)
    resized_w, resized_h = (size, resized_long) if width <= height else (resized_long, size)
    left, top = int(round((resized_w - size) / 2.0)), int(round((resized_h - size) / 2.0))
    scale = short / size
    return left * scale, top * scale, (left + size) * scale, (top + size) * scale


@st.cache_data(max_entries=256)
def load_full_image(dataset_cfg_json, size, index):
    """The original image ``index`` with what the center crop leaves out dimmed and the crop framed in red ([h, w, 3]
    uint8), and the share of the original pixels of each class outside the crop: dict class id -> share."""
    from PIL import ImageDraw

    item = open_raw_dataset(dataset_cfg_json)[index]
    image = item["image"]
    width, height = image.size
    left, top, right, bottom = crop_box(width, height, size)
    array = np.asarray(image).astype(np.float32)
    outside = np.ones((height, width), bool)
    outside[int(round(top)) : int(round(bottom)), int(round(left)) : int(round(right))] = False
    array[outside] *= 0.35
    framed = Image.fromarray(array.astype(np.uint8))
    line = max(2, round(min(width, height) / 120))
    ImageDraw.Draw(framed).rectangle((left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=line)
    label = np.asarray(item["label"])
    classes, counts = np.unique(label[outside], return_counts=True)
    lost = {int(c): n / label.size for c, n in zip(classes, counts, strict=True) if c != VOID}
    return np.asarray(framed), lost


@st.cache_data(max_entries=256)
def load_image(dataset_cfg_json, size, index):
    """The val image ``index`` as an [S, S, 3] uint8 array."""
    dataset, _ = open_dataset(dataset_cfg_json, size)
    return dataset[index]["image"].permute(1, 2, 0).numpy()


# ---- features (not cached: computed by the models for one image) --------------------------------------------------


@st.cache_resource(show_spinner="Loading the backbone and the probed upsamplers (once)...")
def load_models(cache_dir):
    """(backbone, {model: upsampler}, device) of the cache's probes, frozen, on the GPU if there is one."""
    import torch

    from src.analysis.shared import load_probed_model
    from src.backbone import load_backbone

    meta, _, _ = open_cache(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Built as normal tensors: modules created under inference mode get inference-tensor buffers (RoPE's cache key
    # reads their version counter, which inference tensors don't have)
    with torch.inference_mode(False):
        backbone = load_backbone({"name": meta["backbone"]}, device)
        upsamplers = {
            m: load_probed_model(meta["task"], meta["dataset"], meta["backbone"], m, device).model
            for m in meta["models"]
        }
    return backbone, upsamplers, device


@st.cache_data(max_entries=32, show_spinner="Computing the features...")
def feature_images(cache_dir, index, models):
    """The backbone's low-res features of image ``index`` and each model's upsampled ones, as RGB images of their first
    3 principal components (one PCA fit on all of them, so a color means the same features in every panel): dict
    name -> [S, S, 3] uint8, the low-res map enlarged with nearest neighbors."""
    import torch
    import torch.nn.functional as F

    from src.backbone import backbone_features
    from src.dataset.transforms import to_float_image
    from src.utils.visualization import pca

    meta, _, _ = open_cache(cache_dir)
    size = meta["size"]
    dataset, _ = open_dataset(json.dumps(meta["dataset_cfg"], sort_keys=True), size)
    backbone, upsamplers, device = load_models(cache_dir)
    with torch.inference_mode():
        image = to_float_image(dataset[index]["image"][None].to(device))
        img_ups, lr = backbone_features(backbone, image)
        feats = {f"backbone {lr.shape[-2]}x{lr.shape[-1]}": lr.float()}
        feats |= {m: upsamplers[m](img_ups, lr, (size, size)).float() for m in models if m in upsamplers}
        reduced, _ = pca(list(feats.values()))
    images = {}
    for name, rgb in zip(feats, reduced, strict=True):
        rgb = F.interpolate(rgb, size=(size, size), mode="nearest") if rgb.shape[-1] != size else rgb
        images[name] = (rgb[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return images


def draw_features(cache_dir, index, shown):
    """A row aligned with ``draw_image``'s: the low-res backbone features under the low-res prediction (under the
    image without one), then each shown model's upsampled features under its prediction (PCA colors); window
    oracles have no features."""
    meta, _, _ = open_cache(cache_dir)
    images = feature_images(cache_dir, int(index), tuple(m for m in shown if m in meta["models"]))
    has_lowres = open_lowres(cache_dir) is not None
    columns = st.columns(2 + int(has_lowres) + len(shown))
    backbone = next(iter(images))
    columns[2 if has_lowres else 0].image(images[backbone], caption=f"{backbone} features", width="stretch")
    for col, m in zip(columns[2 + int(has_lowres) :], shown, strict=True):
        if m in images:
            col.image(images[m], caption=f"{m} features", width="stretch")


# ---- drawing -------------------------------------------------------------------------------------------------------


def colorize(label):
    """RGB image of a label map, void pixels in white."""
    out = palette()[np.where(label == VOID, 0, label)]
    out[label == VOID] = 255
    return out


def error_overlay(image, label, pred):
    """The image dimmed, wrong labeled pixels in red."""
    wrong = np.zeros_like(image)
    wrong[(label != VOID) & (pred != label)] = (255, 0, 0)
    return (0.4 * image + 0.6 * wrong).astype(np.uint8)


@st.cache_resource
def metrics():
    """Metric name -> description: the accuracy over each pixel subset of ``per_image_accuracy``."""
    from src.analysis.predictions import PIXEL_SUBSETS

    return {name: f"Accuracy, {pixels}" for name, pixels in PIXEL_SUBSETS.items()}


@st.cache_data(max_entries=512)
def scored_pixels(cache_dir, index, metric):
    """The pixels ``metric`` scores in image ``index`` (``src.analysis.predictions.pixel_subsets``): [S, S] bool."""
    import torch

    from src.analysis.predictions import pixel_subsets

    meta, labels, _ = open_cache(cache_dir)
    label = torch.from_numpy(np.array(labels[index]))
    return pixel_subsets(label, meta["num_classes"], tuple(meta["lr_size"]))[metric].numpy()


def accuracy(label, pred, pixels):
    """Share of ``pixels`` where ``pred`` is the label; NaN without such pixels."""
    return float((pred[pixels] == label[pixels]).mean()) if pixels.any() else float("nan")


def score_caption(model, value, metric):
    """``naf (57.4)``, ``naf (thin 41.3)``, or a dash when the image has no such pixels."""
    score = "–" if np.isnan(value) else f"{value * 100:.1f}"
    tag = SUBSET_TAGS.get(metric, metric)
    return f"{model} ({tag + ' ' if tag else ''}{score})"


def swatch(color, text):
    r, g, b = color
    box = "padding:0 8px;margin-right:4px;border:1px solid #888"
    return f"<span style='background:rgb({r},{g},{b});{box}'>&nbsp;</span>{text}"


def class_legend(label, preds, class_names):
    """Markdown swatches of the classes of the ground truth (with their share of the pixels, void included), then of
    the classes only some prediction has."""

    def name(c):
        return class_names[c] if c < len(class_names) else str(c)

    def percent(share):
        return "&lt;1%" if 0 < share < 0.005 else f"{share * 100:.0f}%"

    classes, counts = np.unique(label, return_counts=True)
    shares = dict(zip(classes.tolist(), (counts / label.size).tolist(), strict=True))
    truth = [
        swatch((255, 255, 255), f"void {percent(shares[c])}")
        if c == VOID
        else swatch(palette()[c], f"{name(c)} {percent(shares[c])}")
        for c in sorted(shares, key=shares.get, reverse=True)
    ]
    predicted = sorted(set(np.unique(np.concatenate([np.unique(p) for p in preds.values()])).tolist()) - set(shares))
    line = " &nbsp; ".join(truth)
    if predicted:
        line += " &nbsp;&nbsp; | &nbsp;&nbsp; **predicted only:** " + " &nbsp; ".join(
            swatch(palette()[c], name(c)) for c in predicted
        )
    return line


def draw_image(cache_dir, index, models, errors, metric="acc", show_legend=False, full=False):
    """One row: the image index (and the class legend) above the image, ground truth, the classes of the low-res
    backbone features (``LOWRES_PROBE``'s probe, before any upsampling), then every model's prediction (or errors),
    each with its accuracy under ``metric`` (``metrics()``). With ``full``, the image is the original one, the
    evaluated crop framed, and the header names the classes the crop leaves out."""
    meta, labels, preds = open_cache(cache_dir)
    cfg_json = json.dumps(meta["dataset_cfg"], sort_keys=True)
    image = load_image(cfg_json, meta["size"], int(index))
    label = np.asarray(labels[index])
    model_preds = {
        m: np.asarray(preds[m][index]) if m in preds else oracle_prediction(cache_dir, m, int(index)) for m in models
    }
    _, class_names = open_dataset(cfg_json, meta["size"])
    header = f"**val #{index}**"
    if show_legend:
        header += " &nbsp;&nbsp; " + class_legend(label, model_preds, class_names)
    if full:
        full_image, lost = load_full_image(cfg_json, meta["size"], int(index))
        # Objects only: the background the crop leaves out is no loss worth flagging
        lost = sorted(((share, c) for c, share in lost.items() if c != 0 and share >= 0.001), reverse=True)
        if lost:
            cut = ", ".join(f"{class_names[c]} {share * 100:.1f}%" for share, c in lost)
            header += f" &nbsp;&nbsp; | &nbsp;&nbsp; **cut by the crop:** {cut}"
    st.markdown(header, unsafe_allow_html=True)
    lowres, grid = lowres_prediction(cache_dir, int(index), meta["size"])
    columns = st.columns(2 + int(lowres is not None) + len(model_preds))
    if full:
        columns[0].image(full_image, caption=f"val #{index} (evaluated crop in red)", width="stretch")
    else:
        columns[0].image(image, caption=f"val #{index}", width="stretch")
    columns[1].image(colorize(label), caption="ground truth", width="stretch")
    pixels = scored_pixels(cache_dir, int(index), metric)
    if lowres is not None:
        shown = error_overlay(image, label, lowres) if errors else colorize(lowres)
        name = f"{grid[0]}x{grid[1]} ({LOWRES_PROBE} probe)"
        columns[2].image(shown, caption=score_caption(name, accuracy(label, lowres, pixels), metric), width="stretch")
        columns = columns[1:]
    for col, (m, pred) in zip(columns[2:], model_preds.items(), strict=True):
        shown = error_overlay(image, label, pred) if errors else colorize(pred)
        col.image(shown, caption=score_caption(m, accuracy(label, pred, pixels), metric), width="stretch")


# ---- page ----------------------------------------------------------------------------------------------------------


def percent_columns(df):
    return {c: st.column_config.NumberColumn(c, format="%.2f") for c in df.columns if df[c].dtype.kind == "f"}


def images_page(cache_dir, meta, per_image):
    models = list(meta["models"])
    drawable = models + (list(ORACLES) if has_oracles(cache_dir, meta) else [])

    names = metrics()
    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
    metric = c1.selectbox("Metric", list(names), format_func=names.get)
    sort_by = c2.selectbox("Sort by", models, index=models.index("naf") if "naf" in models else 0)
    versus = c3.selectbox("minus (optional)", ["—", *[m for m in models if m != sort_by]])
    min_pixels = c4.number_input(
        "Min pixels", min_value=0, value=0, step=50, help="Only images with at least this many scored pixels"
    )
    worst_first = c5.toggle("Worst first", value=True)
    shown = st.multiselect(
        "Models", drawable, default=models, key="shown_models", help="Window oracles are drawn, not sorted by"
    )

    # Sort key first, then the accuracy (%) of every shown model that has one
    key = per_image[f"{metric}_{sort_by}"] * 100
    key_name = f"sort: {sort_by}"
    if versus != "—":
        key = key - per_image[f"{metric}_{versus}"] * 100
        key_name = f"sort: {sort_by} − {versus}"
    table = pd.DataFrame(
        {key_name: key, "pixels": per_image[f"n_{metric}"]}
        | {m: per_image[f"{metric}_{m}"] * 100 for m in shown if m in models}
    )
    table = table.loc[key.sort_values(ascending=worst_first, na_position="last").index]
    table = table[table["pixels"] >= max(min_pixels, 1)].reset_index()

    st.caption(
        f"{len(table)} images, sorted by {key_name.removeprefix('sort: ')} ({'lowest' if worst_first else 'highest'} first)"
    )
    event = st.dataframe(
        table,
        hide_index=True,
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        column_config=percent_columns(table),
        key="image_table",
    )

    d0, d1, d2, d3, d4 = st.columns([1, 1, 1, 1, 2])
    full = d0.toggle(
        "Original image",
        value=True,
        help="The whole image before the center crop, the evaluated square framed in red: the predictions, labels "
        "and scores only cover that square",
    )
    errors = d1.toggle("Show errors", value=False)
    legend = d2.toggle("Class legend", value=True)
    features = d3.toggle(
        "Show features",
        key="show_features",
        help="A second row under every image: PCA colors of the backbone's low-res features and of each model's "
        "upsampled features, computed by the models (the first time loads them on the GPU, ~15 s)",
    )
    page_size = d4.select_slider("Images per page", options=[4, 8, 16, 32], value=32)

    selected = event.selection.rows
    if selected:
        index = int(table.loc[selected[0], "image"])
        st.subheader("Selected image")
        draw_image(cache_dir, index, shown, errors, metric=metric, show_legend=True, full=full)
        if features:
            draw_features(cache_dir, index, shown)
        st.divider()

    pages = max(1, (len(table) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1) - 1
    for _, row in table.iloc[page * page_size : (page + 1) * page_size].iterrows():
        draw_image(cache_dir, int(row["image"]), shown, errors, metric=metric, show_legend=legend, full=full)
        if features:
            draw_features(cache_dir, int(row["image"]), shown)


def main():
    args = parse_args()
    st.set_page_config(page_title="Predictions", layout="wide")
    caches = find_caches(args.predictions)
    if not caches:
        st.error(f"No prediction cache under {args.predictions}: run `python analysis.py analysis=predictions`.")
        return

    with st.sidebar:
        st.header("Predictions")
        cache_dir = st.selectbox("Cache", list(caches), format_func=os.path.basename)
        if st.button("Rescan"):
            st.cache_data.clear()
            st.rerun()
        meta_mtime = os.path.getmtime(os.path.join(cache_dir, "meta.json"))
        problem = cache_problem(cache_dir, meta_mtime)
        if problem:
            st.warning(f"This cache may be stale: {problem}")

    meta = caches[cache_dir]
    st.header(f"{meta['task']} · {meta['dataset']} val · {meta['backbone']}")
    st.caption(f"{meta['num_images']} images at {meta['size']} px, models: {', '.join(meta['models'])}")
    images_page(cache_dir, meta, per_image_table(cache_dir, meta_mtime, accuracy_code_version()))


if __name__ == "__main__":
    main()
