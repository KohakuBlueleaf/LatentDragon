"""The frozen `Domain` record and the sample grid shared by its entries."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
    """Everything tasks and eval may need to know about a signal type."""

    name: str
    rank: int
    transforms: Callable[..., Callable]
    metrics: tuple[str, ...]
    to_grid: Callable[..., Tensor]
    to_frames: Callable[[Tensor], Tensor]
    save_grid: Callable[[Tensor, Path], None]


def _strip(x: Tensor) -> Tensor:
    """`(B, ..., C, H, W)` to a padded row `(..., H, B*W, C)`."""
    x = F.pad(x, (GRID_PAD,) * 4, mode="constant", value=0)
    return x.movedim(-3, -1).movedim(0, -3).flatten(-3, -2)


def sample_grid(
    fit_latents: Callable[[Tensor, Tensor], Tensor],
    real: Tensor,
    fakes: Sequence[Tensor],
    latents: Tensor | None = None,
    limit: int = 8,
) -> Tensor:
    """Rows real, latents, fakes; `real` and `fakes` in [-1, 1], `latents` in [0, 1]."""
    real = real[:limit].float()
    rows = [real * 0.5 + 0.5]
    if latents is not None:
        rows.append(fit_latents(latents[:limit].float(), real))
    rows.extend(fake[:limit].float() * 0.5 + 0.5 for fake in fakes)
    grid = torch.cat([_strip(row) for row in rows], dim=-3)
    return grid.clamp(0, 1).mul(255).byte().cpu()


def save_image_grid(grid: Tensor, path: Path) -> None:
    """Write an `(H, W, 3)` uint8 grid as webp."""
    Image.fromarray(grid.numpy()).save(path, quality=95)
