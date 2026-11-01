"""Gradient reducers: every rank's accumulated grads averaged once per bundle step."""

import torch

from latentdragon.training.parallel import sync


class NoReducer:
    """The single-process reducer."""

    def reduce(self, grads: list[torch.Tensor]) -> None:
        return None


class MeanReducer:
    """Flat mean all-reduce, optionally through a compressed wire dtype."""

    def __init__(self, dtype: torch.dtype | None = None):
        self.dtype = dtype

    def reduce(self, grads: list[torch.Tensor]) -> None:
        """Average `grads` across ranks in place through one flat buffer."""
        flat = torch.cat([g.reshape(-1) for g in grads])
        wire = flat if self.dtype is None else flat.to(self.dtype)
        sync.all_mean_(wire)
        restored = wire.to(flat.dtype)

        # Scatter the averaged buffer back into each gradient.
        offset = 0
        for grad in grads:
            size = grad.numel()
            grad.copy_(restored[offset : offset + size].view_as(grad))
            offset += size
