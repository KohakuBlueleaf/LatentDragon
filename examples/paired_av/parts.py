"""Paired video and audio: two data slots, one unified latent slot, two outputs.

    {"video", "audio"} -> encoder -> {"h"} -> kl -> {"z"} -> decoder -> {"frames", "wave"}
                                                                  -> head -> {"video", "audio"}

Every part here is registered from this file, the way an outside library adds its own;
the framework has no code for any of them. The data is synthetic: a blob that moves across
the clip, and a tone whose pitch follows the blob's height.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from latentdragon.domain import DOMAIN, Domain
from latentdragon.domain.video import video_to_frames
from latentdragon.model import DECODER, ENCODER, HEAD, Part
from latentdragon_common.calls import identity
from latentdragon_data import SOURCE

FRAMES, SIZE, WAVE = 4, 16, 256


@SOURCE.register("toy_av")
class ToyAV(Dataset):
    """`{"video": (T, 3, S, S), "audio": (1, L)}` samples, both in [-1, 1]."""

    def __init__(self, transform=None, count: int = 64):
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict:
        generator = torch.Generator().manual_seed(index)
        start, speed = torch.rand(2, generator=generator) * 0.5 + 0.25
        grid = torch.linspace(0, 1, SIZE)

        # A blob drifting right, one frame per step
        frames = []
        for step in range(FRAMES):
            centre = (start + speed * step / FRAMES) % 1.0
            blob = torch.exp(-((grid[None, :] - centre) ** 2) / 0.01)
            blob = blob * torch.exp(-((grid[:, None] - start) ** 2) / 0.01)
            frames.append(blob.expand(3, SIZE, SIZE))
        video = torch.stack(frames) * 2 - 1

        # A tone whose pitch follows the blob's height
        time = torch.arange(WAVE) / WAVE
        audio = torch.sin(2 * math.pi * (4 + 12 * start) * time)[None]
        return {"video": video, "audio": audio}


def passthrough(size: int, train: bool):
    """The toy source already yields finished samples."""
    return identity


@DOMAIN.register("toy_av")
def toy_av() -> Domain:
    """Two slots: clips compared as video, waves compared as plain tensors."""
    return Domain(
        name="toy_av",
        slots=("video", "audio"),
        transforms=passthrough,
        loss_dims={"video": 3, "audio": 2},
        metrics={"video": ("psnr", "ssim")},
        to_frames={"video": video_to_frames},
    )


@ENCODER.register("toy_av")
class AVEncoder(Part):
    """Frames stacked on channels, plus a pooled audio code broadcast over the grid."""

    inputs = ("video", "audio")
    outputs = ("h",)

    def __init__(self, out_channels: int, width: int = 32):
        super().__init__()
        self.video = nn.Conv2d(FRAMES * 3, width, 4, 2, 1)
        self.audio = nn.Conv1d(1, width, 8, 4, 2)
        self.out = nn.Conv2d(width, out_channels, 3, 1, 1)

    @property
    def last_layer(self) -> nn.Module:
        """The output projection, which a kl latent zero-initializes half of."""
        return self.out

    def forward(self, slots: dict) -> dict:
        video = self.video(slots["video"].flatten(1, 2))
        audio = self.audio(slots["audio"]).mean(dim=-1)
        return {"h": self.out(F.silu(video + audio[:, :, None, None]))}


@DECODER.register("toy_av")
class AVDecoder(Part):
    """One latent to clip features and wave features."""

    inputs = ("z",)
    outputs = ("frames", "wave")

    def __init__(self, in_channels: int, width: int = 32):
        super().__init__()
        self.width = width
        self.video = nn.ConvTranspose2d(in_channels, FRAMES * width, 4, 2, 1)
        self.audio = nn.Linear(in_channels, width * WAVE // 4)

    def forward(self, slots: dict) -> dict:
        z = slots["z"]
        frames = self.video(z).unflatten(1, (FRAMES, self.width))
        wave = self.audio(z.mean(dim=(-2, -1))).unflatten(1, (self.width, WAVE // 4))
        return {"frames": F.silu(frames), "wave": F.silu(wave)}


@HEAD.register("toy_av")
class AVHead(Part):
    """Per-frame RGB projection and a wave upsampler."""

    inputs = ("frames", "wave")
    outputs = ("video", "audio")
    objective = "reconstruction"

    def __init__(self, width: int = 32):
        super().__init__()
        self.video = nn.Conv2d(width, 3, 1)
        self.audio = nn.ConvTranspose1d(width, 1, 4, 4)

    @property
    def last_layer(self) -> nn.Module:
        """The video projection, where an adaptive GAN weight would read gradients."""
        return self.video

    def forward(self, slots: dict) -> dict:
        frames = slots["frames"]
        video = self.video(frames.flatten(0, 1)).unflatten(0, frames.shape[:2])
        return {"video": video, "audio": self.audio(slots["wave"])}
