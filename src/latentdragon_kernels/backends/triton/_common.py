"""Shared by Triton ops: the candidate table, launching and compile errors."""

import torch
import triton
from triton.errors import TritonError

ERRORS = (TritonError,)
MIN_PAD = 16
MIN_SM = 80
# row, col, depth, warps, stages; rank order, the first row is the untimed default
TILES = (
    (32, 32, 16, 4, 1),
    (32, 64, 16, 4, 1),
    (64, 32, 16, 4, 1),
    (64, 64, 16, 4, 1),
    (64, 128, 16, 4, 1),
    (128, 64, 16, 4, 1),
    (128, 128, 16, 8, 1),
    (32, 32, 32, 2, 1),
    (64, 64, 32, 4, 1),
    (128, 128, 32, 8, 1),
    (64, 64, 16, 4, 2),
    (128, 64, 16, 8, 2),
)


def tiles(row: str, col: str, depth: str | None = None, top: int = 128) -> list[dict]:
    """`TILES` as configs under a kernel's block names, without blocks over `top`."""
    configs: list[dict] = []
    for tile_row, tile_col, tile_depth, warps, stages in TILES:
        config = {
            row: tile_row,
            col: tile_col,
            "num_warps": warps,
            "num_stages": stages,
        }
        if depth is not None:
            config[depth] = tile_depth
        if max(tile_row, tile_col) <= top and config not in configs:
            configs.append(config)
    return configs


def pad(size: int) -> int:
    """`size` rounded up to a `tl.dot` block size."""
    return max(triton.next_power_of_2(size), MIN_PAD)


def launch(kernel, grid: tuple, device: torch.device, *args, **config) -> None:
    """Run `kernel` on `device`."""
    with torch.cuda.device(device):
        kernel[grid](*args, **config)
