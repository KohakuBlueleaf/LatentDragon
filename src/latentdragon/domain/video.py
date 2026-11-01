"""The `video` domain: `(B, T, C, H, W)`, first frame is the keyframe."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as trns
from torch import Tensor

from latentdragon.domain.record import DOMAIN, Domain, sample_grid, save_image_grid


def video_transforms(size: int, train: bool) -> Callable:
    """`(T, C, H, W)` clip to a normalized tensor with one crop for all frames."""
    crop = trns.RandomCrop(size) if train else trns.CenterCrop(size)
    return trns.Compose(
        [
            trns.Resize(size, antialias=True),
            crop,
            trns.ConvertImageDtype(torch.float32),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def fit_video_latents(latents: Tensor, real: Tensor) -> Tensor:
    """Scale rendered latents to the sample size; frames repeat to the clip length."""
    batch, latent_t = latents.shape[:2]
    clip_t = real.size(1)
    frames = F.interpolate(
        latents.flatten(0, 1), size=real.shape[-2:], mode="nearest"
    ).unflatten(0, (batch, latent_t))
    # Frame 0 is the keyframe; the rest split evenly over the later latents.
    later = 1 + torch.arange(clip_t - 1) * (latent_t - 1) // max(clip_t - 1, 1)
    index = torch.cat([torch.zeros(1, dtype=torch.long), later])
    return frames[:, index.clamp(max=latent_t - 1).to(frames.device)]


def video_to_frames(x: Tensor) -> Tensor:
    """`(B, T, C, H, W)` to a frame batch."""
    return x.flatten(0, 1)


def save_video_grid(grid: Tensor, path: Path, frames: int = 4) -> None:
    """Write the first `frames` frames of a `(T, H, W, 3)` grid side by side."""
    save_image_grid(torch.cat(list(grid[:frames]), dim=1), path)


VIDEO = Domain(
    name="video",
    rank=5,
    transforms=video_transforms,
    metrics=("psnr", "ssim", "rfid"),
    to_grid=partial(sample_grid, fit_video_latents),
    to_frames=video_to_frames,
    save_grid=save_video_grid,
)


@DOMAIN.register("video")
def video_domain() -> Domain:
    """The video record."""
    return VIDEO
