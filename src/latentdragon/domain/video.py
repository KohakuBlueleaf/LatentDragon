"""The `video` domain: one slot of `(B, T, C, H, W)`, the first frame is the keyframe."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as trns
from torch import Tensor

from latentdragon.domain.record import (
    DOMAIN,
    Domain,
    ToSlot,
    sample_grid,
    save_image_grid,
)


def video_transforms(size: int, train: bool, slot: str = "x") -> Callable:
    """`(T, C, H, W)` clip to `{slot: tensor}` with one crop for all frames."""
    crop = trns.RandomCrop(size) if train else trns.CenterCrop(size)
    pipeline = trns.Compose(
        [
            trns.Resize(size, antialias=True),
            crop,
            trns.ConvertImageDtype(torch.float32),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return ToSlot(slot, pipeline)


def fit_video_latents(latents: Tensor, real: Tensor) -> Tensor:
    """Scale a channels-first latent preview to the sample size; frames repeat to the clip.

    A `(B, 3, T, H, W)` preview becomes frames; a `(B, 3, H, W)` one is a single frame.
    """
    latents = latents[:, :, None] if latents.ndim == 4 else latents
    latents = latents.movedim(1, 2)
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


@DOMAIN.register("video")
def video_domain(slot: str = "x") -> Domain:
    """Clips under one slot, `x` by default."""
    return Domain(
        name="video",
        slots=(slot,),
        transforms=partial(video_transforms, slot=slot),
        loss_dims={slot: 3},
        metrics={slot: ("psnr", "ssim", "rfid")},
        to_frames={slot: video_to_frames},
        to_grid=partial(sample_grid, fit_video_latents, slot),
        save_grid=save_video_grid,
    )
