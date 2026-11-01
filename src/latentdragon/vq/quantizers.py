"""`QUANTIZER`: grid quantizers that differ only by codebook source and defaults."""

from latentdragon.vq.codebook import CODEBOOK
from latentdragon.vq.grid import GridQuantizer
from latentdragon_common.registry import Registry

QUANTIZER = Registry("quantizer")


def _grid(
    source: str,
    emb_dim: int,
    n_embed: int,
    partitions: int,
    rvq_levels: int,
    beta: float,
    ivq_loss_weight: float,
    codebook_config: dict,
) -> GridQuantizer:
    """Validate the grid and build its codebook source."""
    if emb_dim % partitions:
        raise ValueError(f"emb_dim {emb_dim} not divisible by partitions {partitions}")
    if rvq_levels < 1:
        raise ValueError(f"rvq_levels must be at least 1, got {rvq_levels}")
    codebook = CODEBOOK.get(source)(
        n_embed=n_embed,
        e_dim=emb_dim // partitions,
        partitions=partitions,
        rvq_levels=rvq_levels,
        **codebook_config,
    )
    return GridQuantizer(codebook, beta=beta, ivq_loss_weight=ivq_loss_weight)


@QUANTIZER.register("kohaku")
def kohaku(
    emb_dim: int,
    n_embed: int,
    partitions: int = 1,
    rvq_levels: int = 1,
    beta: float = 0.25,
    ivq_loss_weight: float = 0.0,
    **codebook_config,
) -> GridQuantizer:
    """Grid over a projected codebook; extra kwargs go to `ProjectedCodebook`."""
    return _grid(
        "projected",
        emb_dim,
        n_embed,
        partitions,
        rvq_levels,
        beta,
        ivq_loss_weight,
        codebook_config,
    )


@QUANTIZER.register("naive")
def naive(
    emb_dim: int,
    n_embed: int,
    partitions: int = 1,
    rvq_levels: int = 1,
    beta: float = 0.25,
    ivq_loss_weight: float = 0.0,
    **codebook_config,
) -> GridQuantizer:
    """Grid over a direct codebook; extra kwargs go to `DirectCodebook`."""
    return _grid(
        "direct",
        emb_dim,
        n_embed,
        partitions,
        rvq_levels,
        beta,
        ivq_loss_weight,
        codebook_config,
    )
