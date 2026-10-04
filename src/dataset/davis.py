"""DAVIS 2017 (480p) for video object segmentation: a local DAVIS folder, or a dataset on the Hugging Face Hub.

Both give the same two views of a split:
- frames, as a segmentation dataset: one sample per frame, labeled with the object ids of its video (0 background,
  1..N objects, 255 void; ids are per video, not classes);
- videos, for label propagation: ``videos``, then ``frame_ids(video)``, ``image(video, i)`` and ``mask(video, i)``.

``datasets`` is imported only by the Hub code.
"""

import glob
import os

from PIL import Image

from .common import check_split_size, read_lines
from .hub import load_hub_split
from .segmentation import SegmentationDataset

DEFAULT_HF_REPO = "shijli/davis2017"

# Frames of the official 2017 trainval split lists (60 / 30 videos)
SPLIT_SIZES = {"train": 4209, "val": 1999}


def annotation_path(frame_path: str) -> str:
    """Annotation of a frame: ``JPEGImages/480p/<video>/<frame>.jpg`` -> ``Annotations/480p/<video>/<frame>.png``."""
    return os.path.splitext(frame_path.replace("JPEGImages", "Annotations"))[0] + ".png"


def video_spans(videos: list[str]) -> dict[str, tuple[int, int]]:
    """``{video: (first, end)}`` sample range of every video, from the video name of each sample in order.

    Fails when a video's samples are not contiguous: the frames of a video must be one run, in frame order.
    """
    spans = {}
    for index, video in enumerate(videos):
        if video in spans:
            first, end = spans[video]
            if end != index:
                raise ValueError(f"the frames of video {video!r} are not contiguous")
            spans[video] = (first, index + 1)
        else:
            spans[video] = (index, index + 1)
    return spans


class BaseDAVIS(SegmentationDataset):
    """Frames in video order, plus access by video. Subclasses set ``spans`` (``video_spans``), ``frame_names``
    (one per sample, in order) and implement ``load`` (frame sample) and ``image`` / ``mask`` (by video)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spans: dict[str, tuple[int, int]] = {}
        self.frame_names: list[str] = []

    @property
    def videos(self) -> list[str]:
        """Video names of the split, in the order of the official list."""
        return list(self.spans)

    def num_frames(self, video: str) -> int:
        first, end = self.spans[video]
        return end - first

    def frame_ids(self, video: str) -> list[str]:
        """Frame file stems of a video (``00000``, ``00001``, ...), in frame order."""
        first, end = self.spans[video]
        return self.frame_names[first:end]

    def index(self, video: str, frame: int) -> int:
        """Sample index of frame ``frame`` of ``video``."""
        first, end = self.spans[video]
        if not 0 <= frame < end - first:
            raise IndexError(f"video {video!r} has {end - first} frames, no frame {frame}")
        return first + frame

    def image(self, video: str, frame: int):
        """RGB PIL image of a frame."""
        raise NotImplementedError

    def mask(self, video: str, frame: int):
        """Annotation of a frame: palette PIL image of object ids."""
        raise NotImplementedError


class DAVIS(BaseDAVIS):
    """Every frame of the videos listed in ``root/ImageSets/2017/<split>.txt`` (``root``: the unpacked DAVIS folder)."""

    def __init__(self, root: str, split: str = "train", resolution: str = "480p", **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.split = split
        names = []
        for video in read_lines(os.path.join(root, "ImageSets", "2017", f"{split}.txt")):
            frames = sorted(glob.glob(os.path.join(root, "JPEGImages", resolution, video, "*.jpg")))
            if not frames:
                raise FileNotFoundError(f"no frames of video {video!r} under {root}")
            self.samples += [(frame, annotation_path(frame)) for frame in frames]
            names += [video] * len(frames)
            self.frame_names += [os.path.splitext(os.path.basename(frame))[0] for frame in frames]
        self.spans = video_spans(names)
        check_split_size("DAVIS", split, len(self), SPLIT_SIZES)

    def image(self, video, frame):
        return Image.open(self.samples[self.index(video, frame)][0]).convert("RGB")

    def mask(self, video, frame):
        return Image.open(self.samples[self.index(video, frame)][1])


class HFDAVIS(BaseDAVIS):
    """DAVIS from a Hub dataset with one row per frame (``id`` = ``<video>/<frame>``, ``video``, ``image``, ``mask``),
    the rows of a video contiguous and in frame order, given as a repo id or URL (default ``shijli/davis2017``).

    ``split="val"`` also matches a split named "validation". Downloaded on first use and cached under ``cache_dir``
    (default: ``$HF_HOME/datasets``).
    """

    def __init__(
        self,
        repo: str = DEFAULT_HF_REPO,
        split: str = "train",
        name: str | None = "semi-supervised",
        cache_dir: str | None = None,
        num_proc: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repo_id, self.split, self.data = load_hub_split(repo, split, name, cache_dir, num_proc)
        # Plain columns only: reading them doesn't decode any image
        self.spans = video_spans(self.data["video"])
        self.ids = self.data["id"]
        self.frame_names = [frame_id.split("/", 1)[1] for frame_id in self.ids]
        # One-column views: a row of the full table decodes both its images
        self.images = self.data.select_columns(["image"])
        self.masks = self.data.select_columns(["mask"])
        check_split_size("DAVIS", split, len(self), SPLIT_SIZES)

    def load(self, index):
        image = self.images[index]["image"].convert("RGB")
        mask = self.masks[index]["mask"] if self.include_labels else None
        return image, mask, f"{self.repo_id}/{self.split}/{self.ids[index]}"

    def __len__(self):
        return len(self.data)

    def image(self, video, frame):
        return self.images[self.index(video, frame)]["image"].convert("RGB")

    def mask(self, video, frame):
        return self.masks[self.index(video, frame)]["mask"]
