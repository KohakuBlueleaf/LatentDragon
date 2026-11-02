"""The partition x residual grid quantizer behind every quantizer."""

import torch
from torch import Tensor, nn

from latentdragon.vq.codebook import Codebook
from latentdragon.vq.losses import commitment_codebook, ivq_terms, no_ivq_terms
from latentdragon.vq.stats import select_stats
from latentdragon_kernels import resolve


class GridQuantizer(nn.Module):
    """Quantize `(..., C)` over P partitions and R residual levels with STE.

    Indices are global: offset by partition for non-shared codes, then by level.
    """

    def __init__(
        self, codebook: Codebook, beta: float = 0.25, ivq_loss_weight: float = 0.0
    ):
        super().__init__()
        self.codebook = codebook
        self.beta = beta
        self.ivq_loss_weight = ivq_loss_weight
        self.n_embed = codebook.n_embed
        self.e_dim = codebook.e_dim
        self.partitions = partitions = codebook.partitions
        self.rvq_levels = levels = codebook.rvq_levels
        self.emb_dim = partitions * self.e_dim

        # Global index offsets per (partition, level)
        owned = partitions if codebook.non_shared else 1
        codes_per_level = self.n_embed * owned
        self.codebook_size = codes_per_level * levels
        partition_ids = torch.arange(partitions)
        level_ids = torch.arange(levels)
        partition_offsets = partition_ids * self.n_embed * codebook.non_shared
        offsets = partition_offsets[:, None] + level_ids * codes_per_level
        self.register_buffer("index_offsets", offsets, persistent=False)
        self.register_buffer("partition_ids", partition_ids[:, None], persistent=False)
        self.register_buffer("level_ids", level_ids[None, :], persistent=False)

        # Stats and ivq variants picked once from the grid shape
        self._index_tail = (partitions,) if levels == 1 else (partitions, levels)
        self._stats = select_stats(self.codebook_size, self.n_embed, partitions, levels)
        self._ivq_terms = ivq_terms if ivq_loss_weight > 0 else no_ivq_terms
        self.track_stats = True
        self.bind_kernels("cpu")

    def bind_kernels(self, device: torch.device | str) -> None:
        """Resolve every kernel for `device`, once."""
        self._quantize = resolve("quantize", device)
        self._nearest = resolve("nearest", device)
        self.codebook.bind_kernels(device)

    def _descend(self, z_p: Tensor, codebook: Tensor) -> tuple[list[Tensor], Tensor]:
        """Quantize level by level on the residual; local indices are `(..., P, R)`."""
        parts, local = [], []
        residual = z_p
        for level in codebook.unbind(0):
            z_q, residual, indices, _ = self._quantize(residual, level)
            parts.append(z_q)
            local.append(indices)
        return parts, torch.stack(local, dim=-1)

    def forward(self, z: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """`(..., C)` -> straight-through `(..., C)` and the info dict."""
        # Split channels into partitions and quantize level by level
        head = z.shape[:-1]
        z_p = z.reshape(*head, self.partitions, self.e_dim)
        codebook = self.codebook()
        parts, local = self._descend(z_p, codebook)
        z_q = sum(parts)

        # Commitment, codebook and optional inverse-vq losses
        commitment, codebook_loss = commitment_codebook(z_q, z_p)
        vq_loss = self.beta * commitment + codebook_loss
        terms = self._ivq_terms(codebook, z_p, parts, self._nearest)
        for term in terms.values():
            vq_loss = vq_loss + self.ivq_loss_weight * term

        # Straight-through output, global indices and usage stats
        out = (z_p + (z_q - z_p).detach()).reshape(*head, self.emb_dim)
        indices = local + self.index_offsets
        stats = {}
        if self.track_stats:
            with torch.no_grad():
                stats = self._stats(indices, local)
        info = {
            "indices": indices.reshape(*head, *self._index_tail),
            "vq_loss": vq_loss,
            "commitment_loss": commitment,
            "codebook_loss": codebook_loss,
            **stats,
            **terms,
        }
        return out, info

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """Global indices `(..., P)` or `(..., P, R)` -> summed codes `(..., C)`."""
        head = indices.shape[: -len(self._index_tail)]
        grid = indices.reshape(*head, self.partitions, self.rvq_levels)
        local = grid - self.index_offsets
        codes = self.codebook()[self.level_ids, self.partition_ids, local]
        return codes.sum(dim=-2).reshape(*head, self.emb_dim)
