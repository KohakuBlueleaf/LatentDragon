"""What latent views and terms can read."""

from dataclasses import dataclass, field

import torch


@dataclass
class LatentCtx:
    """One micro-batch's signal and latents, plus requested teacher features."""

    x: torch.Tensor
    z_raw: torch.Tensor
    z: torch.Tensor
    step: int
    teacher: dict[str, torch.Tensor] = field(default_factory=dict)
    fake: torch.Tensor | None = None
    seed: torch.Tensor | None = None
