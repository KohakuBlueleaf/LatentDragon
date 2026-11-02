"""Total code usage over a validation pass."""

import torch


class NoTally:
    """The tally of a latent def with no codes."""

    def add(self, indices) -> None:
        return None

    def report(self, sync) -> dict:
        return {}


class CodeTally:
    """Per-slot code counts; a slot is one (level, partition) pair of the grid."""

    def __init__(self, latent, device):
        self.quantizer = latent.quantizer
        self.counts = torch.zeros(
            self.quantizer.rvq_levels,
            self.quantizer.partitions,
            self.quantizer.n_embed,
            device=device,
            dtype=torch.long,
        )

    def add(self, indices: torch.Tensor) -> None:
        """Accumulate global indices of shape `(..., P)` or `(..., P, R)`."""
        levels, parts, n_embed = self.counts.shape
        flat = (indices % n_embed).reshape(-1, parts, levels)
        for level in range(levels):
            for part in range(parts):
                ids = flat[:, part, level]
                self.counts[level, part].scatter_add_(
                    0, ids, torch.ones_like(ids, dtype=torch.long)
                )

    def report(self, sync) -> dict:
        """Usage plus mean and product of per-slot perplexity rates."""
        sync.all_sum_(self.counts)
        levels, _, n_embed = self.counts.shape
        pooled = self.quantizer.codebook_size == levels * n_embed
        used = self.counts.sum(dim=1) if pooled else self.counts
        probs = self.counts.float()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1.0)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs).sum(dim=-1)
        rates = entropy.exp() / n_embed
        return {
            "val/vq_total_usage": (used > 0).float().mean().item(),
            "val/vq_total_perplexity_rate": rates.mean().item(),
            "val/vq_total_perplexity_rate_prod": rates.prod().item(),
        }
