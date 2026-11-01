"""Codebook sources: `forward() -> (R, P, N, D)`."""

import torch
from torch import Tensor, nn

from latentdragon.vq.sharing import Sharing, select_sharing
from latentdragon_common.registry import Registry
from latentdragon_kernels import resolve

CODEBOOK = Registry("codebook")
MLP_ACTIVATIONS = ("silu", "mish", "gelu")


class Codebook(nn.Module):
    """Grid dimensions shared by every codebook source."""

    def __init__(self, sharing: Sharing):
        super().__init__()
        self.sharing = sharing
        self.n_embed = sharing.n_embed
        self.e_dim = sharing.e_dim
        self.partitions = sharing.partitions
        self.rvq_levels = sharing.rvq_levels
        self.non_shared = sharing.non_shared

    def bind_kernels(self, device: torch.device | str) -> None:
        """Resolve kernels for `device`; sources without kernels do nothing."""


@CODEBOOK.register("direct")
class DirectCodebook(Codebook):
    """A learned embedding table of `(T * N, R * D)`, `T` tables."""

    def __init__(
        self,
        n_embed: int,
        e_dim: int,
        partitions: int = 1,
        rvq_levels: int = 1,
        partition_mode: str = "shared",
    ):
        if partition_mode == "hyper":
            raise ValueError("partition_mode 'hyper' needs a projected codebook")
        super().__init__(
            select_sharing(partition_mode, partitions, rvq_levels, n_embed, e_dim)
        )
        tables = self.sharing.tables
        self.embedding = nn.Embedding(n_embed * tables, rvq_levels * e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self) -> Tensor:
        """Split the table into levels, expanded to every partition."""
        tables = self.sharing.tables
        split = self.embedding.weight.reshape(
            tables, self.n_embed, self.rvq_levels, self.e_dim
        )
        grid = split.permute(2, 0, 1, 3)
        return grid.expand(-1, self.partitions, -1, -1)


@CODEBOOK.register("projected")
class ProjectedCodebook(Codebook):
    """`scale * proj(base)` with a linear or two-layer MLP projector."""

    def __init__(
        self,
        n_embed: int,
        e_dim: int,
        partitions: int = 1,
        rvq_levels: int = 1,
        partition_mode: str = "shared",
        separated_proj: bool = True,
        mlp_projector: bool = False,
        mlp_hidden_dim: int = 1024,
        mlp_act: str = "silu",
        base_dim: int | None = None,
        trainable_base: bool = False,
        dim_scale: bool = False,
    ):
        if mlp_act not in MLP_ACTIVATIONS:
            raise ValueError(
                f"mlp_act must be one of {MLP_ACTIVATIONS}, got {mlp_act!r}"
            )
        super().__init__(
            select_sharing(
                partition_mode, partitions, rvq_levels, n_embed, e_dim, separated_proj
            )
        )
        self.base_dim = e_dim if base_dim is None else base_dim
        self.trainable_base = trainable_base
        self.mlp_act = mlp_act

        # Base table: a parameter if trainable, else a saved buffer
        base_std = 2 * e_dim**-0.5 if mlp_projector else e_dim**-0.5
        base = torch.empty(*self.sharing.base_shape(self.base_dim))
        nn.init.normal_(base, mean=0, std=base_std)
        if trainable_base:
            self.codebook_base = nn.Parameter(base)
        else:
            self.register_buffer("codebook_base", base, persistent=True)

        # Projector from base to codes
        if mlp_projector:
            self._init_mlp(mlp_hidden_dim)
        else:
            self._init_linear()

        # Per-level scale, optionally per-dim as well
        self.codebook_scale = nn.Parameter(torch.ones(rvq_levels))
        self._rescale = self._scale_levels
        if dim_scale:
            self.codebook_dim_scale = nn.Parameter(torch.ones(rvq_levels * e_dim))
            self._rescale = self._scale_levels_and_dims
        self.bind_kernels("cpu")

    def _init_linear(self) -> None:
        """Create the per-group linear projector parameters."""
        groups, out_dim = self.sharing.groups, self.sharing.out_dim
        self.codebook_proj_w = nn.Parameter(torch.empty(groups, self.base_dim, out_dim))
        self.codebook_proj_b = nn.Parameter(torch.zeros(groups, 1, out_dim))
        nn.init.normal_(self.codebook_proj_w, mean=0, std=self.base_dim**-0.5)
        self._project = self._project_linear

    def _init_mlp(self, hidden: int) -> None:
        """Create the per-group two-layer MLP projector parameters."""
        groups, out_dim = self.sharing.groups, self.sharing.out_dim
        self.codebook_proj_w1 = nn.Parameter(torch.empty(groups, self.base_dim, hidden))
        self.codebook_proj_b1 = nn.Parameter(torch.zeros(groups, 1, hidden))
        self.codebook_proj_w2 = nn.Parameter(torch.empty(groups, hidden, out_dim))
        self.codebook_proj_b2 = nn.Parameter(torch.zeros(groups, 1, out_dim))
        nn.init.normal_(self.codebook_proj_w1, mean=0, std=self.base_dim**-0.5)
        nn.init.normal_(self.codebook_proj_w2, mean=0, std=hidden**-0.5)
        self._project = self._project_mlp

    def bind_kernels(self, device: torch.device | str) -> None:
        """Resolve the MLP projection kernel for `device`."""
        self._mlp_project = resolve("mlp_project", device)

    def _project_linear(self, base: Tensor) -> Tensor:
        """Grouped linear projection of `(G, N, b)`."""
        return torch.bmm(base, self.codebook_proj_w) + self.codebook_proj_b

    def _project_mlp(self, base: Tensor) -> Tensor:
        """Grouped two-layer MLP projection of `(G, N, b)` through the kernel."""
        return self._mlp_project(
            base,
            self.codebook_proj_w1,
            self.codebook_proj_b1,
            self.codebook_proj_w2,
            self.codebook_proj_b2,
            self.mlp_act,
            self.trainable_base,
        )

    def _scale_levels(self, grid: Tensor) -> Tensor:
        """Scale each level of `(R, P, N, D)`."""
        return grid * self.codebook_scale.reshape(-1, 1, 1, 1)

    def _scale_levels_and_dims(self, grid: Tensor) -> Tensor:
        """Scale each level, then each code dimension, of `(R, P, N, D)`."""
        dims = self.codebook_dim_scale.reshape(self.rvq_levels, 1, 1, self.e_dim)
        return self._scale_levels(grid) * dims

    def forward(self) -> Tensor:
        """Project the base and scale it per level."""
        projected = self._project(self.sharing.expand_base(self.codebook_base))
        return self._rescale(self.sharing.to_grid(projected))
