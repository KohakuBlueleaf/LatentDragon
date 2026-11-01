"""Codebook usage and perplexity, kept in fp32 for any input dtype."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor


def _counts(flat: Tensor, size: int) -> Tensor:
    """Occurrences of each of `size` codes in `flat`."""
    counts = torch.zeros(size, device=flat.device, dtype=torch.float32)
    return counts.scatter_add_(
        0, flat, torch.ones(flat.shape[0], device=flat.device, dtype=torch.float32)
    )


def _perplexity(counts: Tensor) -> Tensor:
    """exp(entropy) of the empirical distribution behind `counts`."""
    probs = counts / counts.sum()
    log_probs = torch.where(
        probs > 0, probs * torch.log(probs), torch.zeros_like(probs)
    )
    return torch.exp(-log_probs.sum())


def usage_perplexity(indices: Tensor, size: int) -> tuple[Tensor, Tensor]:
    """Fraction of `size` codes used at least once, and perplexity."""
    counts = _counts(indices.flatten().long(), size)
    return (counts > 0).float().mean(), _perplexity(counts)


def partitioned_usage_perplexity(
    local: Tensor, n_embed: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-partition stats of local indices `(..., P)`.

    Returns the geometric mean of usages, the mean rate and the product of rates.
    """
    partitions = local.shape[-1]
    log_usage = []
    rates = []

    # Usage and perplexity rate of each partition
    for part in range(partitions):
        counts = _counts(local[..., part].flatten().long(), n_embed)
        log_usage.append((counts > 0).float().mean().clamp(min=1e-10).log())
        rates.append(_perplexity(counts) / n_embed)
    rates = torch.stack(rates)
    return torch.stack(log_usage).mean().exp(), rates.mean(), rates.prod()


def flat_stats(indices: Tensor, local: Tensor, size: int) -> dict[str, Tensor]:
    """Stats for one partition."""
    usage, perplexity = usage_perplexity(indices, size)
    return {
        "usage": usage,
        "perplexity": perplexity,
        "perplexity_rate": perplexity / size,
    }


def partition_stats(
    indices: Tensor, local: Tensor, size: int, n_embed: int
) -> dict[str, Tensor]:
    """Stats for P > 1, one level: usage is the partition geometric mean."""
    _, perplexity = usage_perplexity(indices, size)
    usage, rate, rate_prod = partitioned_usage_perplexity(local[..., 0], n_embed)
    return {
        "usage": usage,
        "perplexity": perplexity,
        "perplexity_rate": rate,
        "perplexity_rate_prod": rate_prod,
    }


def residual_partition_stats(
    indices: Tensor, local: Tensor, size: int, n_embed: int
) -> dict[str, Tensor]:
    """Stats for P > 1, R > 1: usage is pooled, rates are meaned over levels."""
    usage, perplexity = usage_perplexity(indices, size)
    levels = [
        partitioned_usage_perplexity(level, n_embed) for level in local.unbind(dim=-1)
    ]
    return {
        "usage": usage,
        "perplexity": perplexity,
        "perplexity_rate": torch.stack([rate for _, rate, _ in levels]).mean(),
        "perplexity_rate_prod": torch.stack([prod for _, _, prod in levels]).mean(),
    }


def select_stats(
    size: int, n_embed: int, partitions: int, rvq_levels: int
) -> Callable[[Tensor, Tensor], dict[str, Tensor]]:
    """Stats callable `(indices, local)` for one grid, both shaped `(..., P, R)`."""
    match (partitions > 1, rvq_levels > 1):
        case (False, _):
            return partial(flat_stats, size=size)
        case (True, False):
            return partial(partition_stats, size=size, n_embed=n_embed)
        case (True, True):
            return partial(residual_partition_stats, size=size, n_embed=n_embed)
