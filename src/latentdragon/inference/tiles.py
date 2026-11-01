"""Tile geometry: which input range feeds each tile and which part of it is kept."""

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil

from torch import Tensor


@dataclass(frozen=True)
class Span:
    """One tile on one axis: feed `[start, end)`, keep `[keep_start, keep_end)`."""

    start: int
    end: int
    keep_start: int
    keep_end: int


def symmetric_spans(length: int, tile: int, overlap: int) -> list[Span]:
    """The fewest tiles of at most `tile` cells, with `overlap` kept out on inner edges."""
    if length <= tile:
        return [Span(0, length, 0, length)]
    if tile <= 2 * overlap:
        raise ValueError(f"tile {tile} must exceed twice the overlap {overlap}")

    # Edge tiles keep their outer overlap, so only the middle has to be divided
    middle = length - 2 * overlap
    count = max(2, ceil(middle / (tile - 2 * overlap)))
    cuts = [
        0,
        *(overlap + index * middle // count for index in range(1, count)),
        length,
    ]

    return [
        Span(max(0, low - overlap), min(length, high + overlap), low, high)
        for low, high in zip(cuts, cuts[1:])
    ]


def causal_spans(length: int, tile: int, overlap: int) -> list[Span]:
    """The fewest tiles of at most `tile` frames, each fed `overlap` past frames."""
    if length <= tile:
        return [Span(0, length, 0, length)]
    if tile <= overlap:
        raise ValueError(f"temporal tile {tile} must exceed the overlap {overlap}")

    # The first tile has no past, so it keeps all of its frames
    fresh = tile - overlap
    cuts = [0, *range(tile, length, fresh), length]
    return [
        Span(max(0, low - overlap), high, low, high)
        for low, high in zip(cuts, cuts[1:])
    ]


def stitch(
    x: Tensor,
    tile: tuple[int, int],
    overlap: int,
    scale: int,
    decode: Callable[[Tensor, int], Tensor],
) -> Tensor:
    """Decode `x` tile by tile over its last two dims and paste the kept regions.

    `tile` is `(height, width)` in output pixels and `overlap` is in cells of `x`.
    `decode(part, index)` gets each tile with its position in the tile order.
    """
    rows = symmetric_spans(x.shape[-2], tile[0] // scale, overlap)
    cols = symmetric_spans(x.shape[-1], tile[1] // scale, overlap)
    grid = [(row, col) for row in rows for col in cols]

    output = None
    for index, (row, col) in enumerate(grid):
        decoded = decode(x[..., row.start : row.end, col.start : col.end], index)
        if output is None:
            height, width = x.shape[-2] * scale, x.shape[-1] * scale
            output = decoded.new_empty((*decoded.shape[:-2], height, width))

        # Drop the overlap from the decoded tile and paste what is left
        top = (row.keep_start - row.start) * scale
        left = (col.keep_start - col.start) * scale
        rows_kept = (row.keep_end - row.keep_start) * scale
        cols_kept = (col.keep_end - col.keep_start) * scale
        output[
            ...,
            row.keep_start * scale : row.keep_end * scale,
            col.keep_start * scale : col.keep_end * scale,
        ] = decoded[..., top : top + rows_kept, left : left + cols_kept]
    return output
