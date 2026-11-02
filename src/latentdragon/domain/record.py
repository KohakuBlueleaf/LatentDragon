"""The `Domain` record, a data def, and the sample grid shared by the built-in ones.

A domain names the slots of a sample and owns everything that reads them: how a source item
becomes a sample, which loss layout each slot takes, which metrics apply to which slot, how a
slot flattens to frames for image metrics, and how a batch and its reconstructions become a
sample picture.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from latentdragon_common.registry import Registry

DOMAIN = Registry("domain")

GRID_PAD = 4


@dataclass(frozen=True)
class Domain:
    """One data def.

    `transforms(size, train)` returns a callable from a source item to a sample dict.
    `loss_dims` gives each compared slot its sensecraft layout (2 for frames, 3 for clips).
    `metrics` lists, per slot, the names among `psnr`, `ssim` and `rfid`; `to_frames` flattens
    such a slot to `(N, C, H, W)` in [-1, 1]. `to_grid(real, fakes, latents)` builds the sample
    picture from a batch, its reconstructions and optional latent previews, or is None when
    the domain draws none.
    """

    name: str
    slots: tuple[str, ...]
    transforms: Callable[..., Callable]
    loss_dims: dict[str, int]
    metrics: dict[str, tuple[str, ...]] = field(default_factory=dict)
    to_frames: dict[str, Callable[[Tensor], Tensor]] = field(default_factory=dict)
    to_grid: Callable[..., Tensor] | None = None
    save_grid: Callable[[Tensor, Path], None] | None = None


class ToSlot:
    """A source item through `pipeline`, placed under `slot`; picklable for data workers."""

    def __init__(self, slot: str, pipeline: Callable):
        self.slot = slot
        self.pipeline = pipeline

    def __call__(self, item) -> dict:
        return {self.slot: self.pipeline(item)}


def _strip(x: Tensor) -> Tensor:
    """`(B, ..., C, H, W)` to a padded row `(..., H, B*W, C)`."""
    x = F.pad(x, (GRID_PAD,) * 4, mode="constant", value=0)
    return x.movedim(-3, -1).movedim(0, -3).flatten(-3, -2)


def sample_grid(
    fit_latents: Callable[[Tensor, Tensor], Tensor],
    slot: str,
    real: dict,
    fakes: Sequence[dict],
    latents: Tensor | None = None,
    limit: int = 8,
) -> Tensor:
    """Rows real, latents, fakes of one slot; signals in [-1, 1], `latents` in [0, 1]."""
    real = real[slot][:limit].float()
    rows = [real * 0.5 + 0.5]
    if latents is not None:
        rows.append(fit_latents(latents[:limit].float(), real))
    rows.extend(fake[slot][:limit].float() * 0.5 + 0.5 for fake in fakes)
    grid = torch.cat([_strip(row) for row in rows], dim=-3)
    return grid.clamp(0, 1).mul(255).byte().cpu()


def save_image_grid(grid: Tensor, path: Path) -> None:
    """Write an `(H, W, 3)` uint8 grid as webp."""
    Image.fromarray(grid.numpy()).save(path, quality=95)
