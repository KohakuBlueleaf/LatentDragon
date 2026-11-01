"""Bottleneck output record."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class BottleneckOut:
    """What a bottleneck hands to the trunk and the task."""

    z: Tensor  # latent handed to the trunk, channels at dim 1
    aux: dict[str, Tensor]  # live scalars added to the composed backward
    metrics: dict[str, Tensor]  # detached scalars for logging
    indices: Tensor | None = None  # code indices, (..., P) or (..., P, R)
    raw: Tensor | None = None  # latent before quantization or sampling
