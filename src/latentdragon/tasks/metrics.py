"""Smoothed training scalars and sliding-window code usage."""

import torch


class ScalarEma:
    """`min(0.995, step / (step + 10))` EMA of named device scalars."""

    def __init__(self):
        self.values: dict[str, torch.Tensor] = {}

    def update(self, step: int, name: str, value: torch.Tensor) -> torch.Tensor:
        decay = min(0.995, step / (step + 10))
        previous = self.values.get(name)
        current = value.detach()
        smooth = (
            current if previous is None else previous * decay + current * (1 - decay)
        )
        self.values[name] = smooth
        return smooth


class NoUsage:
    """Usage window of a bottleneck with no codes."""

    def mark(self, indices) -> None:
        return None

    def advance(self, stride: int, sync) -> dict:
        return {}


class CodeUsage:
    """Fraction of codes used within the last `window` steps, across ranks."""

    def __init__(self, codebook_size: int, device, window: float = 1000.0):
        self.size = codebook_size
        self.window = window
        self.counts = torch.zeros(codebook_size, device=device)
        self.used = torch.zeros_like(self.counts)

    def mark(self, indices: torch.Tensor) -> None:
        """Note this micro-batch's codes."""
        self.used.scatter_(0, indices.reshape(-1).long(), 1.0)

    def advance(self, stride: int, sync) -> dict:
        """Close the step: share marks across ranks and move the window."""
        sync.all_max_(self.used)
        self.counts.sub_(stride).clamp_(min=0.0)
        self.counts.masked_fill_(self.used > 0, self.window)
        self.used.zero_()
        return {"vq/usage_ema": (self.counts > 0).float().mean()}
