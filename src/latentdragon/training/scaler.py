"""Gradient scalers: one per Bundle so networks never share scale state."""

import torch


class NoScaler:
    """The bf16 and fp32 policy: nothing is scaled."""

    value = 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale(self, grads: list[torch.Tensor]) -> None:
        return None

    def update(self, finite: bool) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class DynamicScaler:
    """Backs off on a non-finite step, grows after `interval` finite ones."""

    def __init__(
        self,
        init: float = 2.0**14,
        growth: float = 2.0,
        backoff: float = 0.5,
        interval: int = 2000,
        maximum: float = 2.0**15,
    ):
        self.value = init
        self.growth = growth
        self.backoff = backoff
        self.interval = interval
        self.maximum = maximum
        self.good = 0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.value

    def unscale(self, grads: list[torch.Tensor]) -> None:
        torch._foreach_mul_(grads, 1.0 / self.value)

    def update(self, finite: bool) -> None:
        if not finite:
            self.value *= self.backoff
            self.good = 0
            return
        self.good += 1
        if self.good >= self.interval:
            self.value = min(self.value * self.growth, self.maximum)
            self.good = 0

    def state_dict(self) -> dict:
        return {"value": self.value, "good": self.good}

    def load_state_dict(self, state: dict) -> None:
        self.value = state["value"]
        self.good = state["good"]
