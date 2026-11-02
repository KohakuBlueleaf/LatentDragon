"""A clip packed into a keyframe latent and a motion latent, each with its own latent def.

    {"x"} -> encoder -> {"h_key", "h_motion"} -> slots(kl, vq) -> {"key", "motion"}
          -> decoder -> {"y"} -> conv head (per frame) -> {"x"}

The latent def is `slots`: a kl def on the keyframe slot next to a vq def on the motion
slot. The vq def stores indices and reads them back through its own codebook, so a packed
latent is `{"key": tensor, "motion": {"indices": tensor}}`. Encoder and decoder are
registered from this file; the domain, latent defs and head are the built-in ones.
"""

import torch
import torch.nn.functional as F
from torch import nn

from latentdragon.model import DECODER, ENCODER, Part

FRAMES = 5


@ENCODER.register("key_motion")
class KeyMotionEncoder(Part):
    """The first frame to a keyframe slot, frame differences to a motion slot."""

    inputs = ("x",)
    outputs = ("h_key", "h_motion")

    def __init__(self, key_channels: int, motion_channels: int, width: int = 32):
        super().__init__()
        self.key = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1), nn.SiLU(), nn.Conv2d(width, key_channels, 1)
        )
        self.motion = nn.Sequential(
            nn.Conv2d((FRAMES - 1) * 3, width, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(width, motion_channels, 1),
        )

    @property
    def last_layer(self) -> nn.Module:
        """The keyframe projection; the kl def zero-initializes its logvar half."""
        return self.key[-1]

    def forward(self, slots: dict) -> dict:
        clip = slots["x"]
        motion = (clip[:, 1:] - clip[:, :-1]).flatten(1, 2)
        return {"h_key": self.key(clip[:, 0]), "h_motion": self.motion(motion)}


@DECODER.register("key_motion")
class KeyMotionDecoder(Part):
    """Keyframe and motion latents to per-frame features `(B, T, width, H, W)`."""

    inputs = ("key", "motion")
    outputs = ("y",)

    def __init__(self, key_channels: int, motion_channels: int, width: int = 32):
        super().__init__()
        self.width = width
        self.mix = nn.Conv2d(key_channels + motion_channels, width, 3, 1, 1)
        self.up = nn.ConvTranspose2d(width, FRAMES * width, 4, 2, 1)

    def forward(self, slots: dict) -> dict:
        joined = torch.cat([slots["key"], slots["motion"]], dim=1)
        frames = self.up(F.silu(self.mix(joined)))
        return {"y": F.silu(frames).unflatten(1, (FRAMES, self.width))}
