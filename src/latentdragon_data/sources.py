"""Sources: file-backed map datasets, resolved through the `SOURCE` registry."""

import os
import random
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from latentdragon_common.log import get_logger
from latentdragon_common.registry import Registry

try:
    import av
except ImportError:
    av = None

SOURCE = Registry("source")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov")
log = get_logger(__name__)


def scan_files(root: str, suffixes: tuple[str, ...]) -> np.ndarray:
    """Relative paths of every matching file under `root`, sorted."""
    found = []
    for folder, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(suffixes):
                found.append(os.path.relpath(os.path.join(folder, name), root))
    return np.array(sorted(found))


def cached_files(root: str, suffixes: tuple[str, ...], meta: str) -> np.ndarray:
    """The file list from `<root>/<meta>`, scanning and caching when absent or empty."""
    path = os.path.join(root, meta)
    if os.path.isfile(path):
        files = np.load(path)
        if len(files):
            return files
    files = scan_files(root, suffixes)
    if len(files) == 0:
        raise FileNotFoundError(f"no files under {root}")
    temp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(temp, files)
    os.replace(temp, path)
    return files


class FileSource(Dataset):
    """A file list with retry on a bad item; `_load` reads one path."""

    suffixes: tuple[str, ...] = ()

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        meta: str = "imgonly_metadata_filtered.npy",
    ):
        self.root = root
        self.transform = transform
        self.files = cached_files(root, self.suffixes, meta)

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: str):
        raise NotImplementedError

    def __getitem__(self, index: int):
        """The sample at `index`; an unreadable file is replaced by a random one."""
        while True:
            path = os.path.join(self.root, str(self.files[index]))
            try:
                return self._load(path)
            except (OSError, ValueError) as err:
                log.warning("bad sample", path=path, error=err)
                index = random.randint(0, len(self.files) - 1)


@SOURCE.register("image_folder")
class ImageFolder(FileSource):
    """Images read with PIL and passed through `transform`."""

    suffixes = IMAGE_SUFFIXES

    def _load(self, path: str):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


@SOURCE.register("video_folder")
class VideoFolder(FileSource):
    """Videos decoded to `frames` consecutive frames, `(T, C, H, W)`."""

    suffixes = VIDEO_SUFFIXES

    def __init__(self, root, transform=None, frames: int = 17, **kwargs):
        if av is None:
            raise ImportError("video_folder needs the `av` package")
        super().__init__(root, transform, **kwargs)
        self.frames = frames

    def _load(self, path: str) -> torch.Tensor:
        # Decode every frame of the first video stream
        with av.open(path) as container:
            images = [f.to_image().convert("RGB") for f in container.decode(video=0)]
        if len(images) < self.frames:
            raise ValueError(f"{path} has fewer than {self.frames} frames")

        # Random window of consecutive frames -> (T, C, H, W)
        start = random.randint(0, len(images) - self.frames)
        chosen = images[start : start + self.frames]
        clip = np.stack([np.asarray(image) for image in chosen])
        return self.transform(torch.from_numpy(clip).permute(0, 3, 1, 2))


@SOURCE.register("synthetic")
class Synthetic(Dataset):
    """Fixed random `shape` samples, for measuring a step without file reads."""

    def __init__(self, transform=None, shape=(3, 256, 256), count: int = 4096):
        self.count = count
        self.sample = torch.rand(tuple(shape)) * 2 - 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.sample


class Concat(Dataset):
    """Several sources read as one index space."""

    def __init__(self, sources: list[Dataset]):
        self.sources = sources
        self.ends = np.cumsum([len(s) for s in sources])

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int):
        # Locate the source holding `index`, then the offset within it
        which = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[which - 1]) if which else 0
        return self.sources[which][index - start]
