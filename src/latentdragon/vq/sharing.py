"""Partition sharing: how tables and projector groups map onto the `(R, P, N, D)` grid."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from torch import Tensor

MODES = ("shared", "indep", "hyper")


def _shared_grid(codes: Tensor, partitions: int) -> Tensor:
    """`(R, N, D)` -> `(R, P, N, D)` expanded views."""
    return codes.unsqueeze(1).expand(-1, partitions, -1, -1)


def _separated_grid(codes: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
    """`(R * P, N, D)` -> `(R, P, N, D)`."""
    return codes.reshape(shape)


def _joint_grid(codes: Tensor, shape: tuple[int, int, int, int]) -> Tensor:
    """`(R, N, P * D)` -> `(R, P, N, D)`."""
    levels, partitions, n_embed, dim = shape
    return codes.reshape(levels, n_embed, partitions, dim).permute(0, 2, 1, 3)


@dataclass(frozen=True)
class Sharing:
    """Build-time layout of one codebook grid."""

    mode: str
    partitions: int
    rvq_levels: int
    n_embed: int
    e_dim: int
    tables: int  # independent base or embedding tables
    groups: int  # projector groups G
    out_dim: int  # projector output per group
    to_grid: Callable[[Tensor], Tensor]  # projector output -> (R, P, N, D)

    @property
    def non_shared(self) -> bool:
        """Each partition owns its codes, so indices carry partition offsets."""
        return self.mode != "shared"

    def base_shape(self, base_dim: int) -> tuple[int, ...]:
        """Stored base shape: `(P, N, b)` for indep, else `(N, b)`."""
        head = (self.tables,) if self.mode == "indep" else ()
        return (*head, self.n_embed, base_dim)

    def expand_base(self, base: Tensor) -> Tensor:
        """Stored base -> `(G, N, b)`, tables repeated once per level."""
        n_embed, base_dim = base.shape[-2:]
        repeats = self.groups // self.tables
        tabled = base.reshape(self.tables, n_embed, base_dim).unsqueeze(0)
        return tabled.expand(repeats, -1, -1, -1).reshape(
            self.groups, n_embed, base_dim
        )


def select_sharing(
    mode: str,
    partitions: int,
    rvq_levels: int,
    n_embed: int,
    e_dim: int,
    separated_proj: bool = True,
) -> Sharing:
    """Resolve a sharing mode; `separated_proj` only matters for hyper."""
    grid = (rvq_levels, partitions, n_embed, e_dim)
    match (mode, separated_proj):
        case ("shared", _):
            tables, groups, out_dim = 1, rvq_levels, e_dim
            to_grid = partial(_shared_grid, partitions=partitions)
        case ("indep", _):
            tables, groups, out_dim = partitions, rvq_levels * partitions, e_dim
            to_grid = partial(_separated_grid, shape=grid)
        case ("hyper", True):
            tables, groups, out_dim = 1, rvq_levels * partitions, e_dim
            to_grid = partial(_separated_grid, shape=grid)
        case ("hyper", False):
            tables, groups, out_dim = 1, rvq_levels, partitions * e_dim
            to_grid = partial(_joint_grid, shape=grid)
        case _:
            raise ValueError(f"partition_mode must be one of {MODES}, got {mode!r}")
    return Sharing(
        mode, partitions, rvq_levels, n_embed, e_dim, tables, groups, out_dim, to_grid
    )
