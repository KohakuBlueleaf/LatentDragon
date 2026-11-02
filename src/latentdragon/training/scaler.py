"""Gradient scalers: one per Bundle so networks never share scale state."""

import torch


class NoScaler:
    """The bf16 and fp32 policy: nothing is scaled."""

    value = 1.0

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale(self, grads: list[torch.Tensor]) -> None:
        return None

    def update(self, bad: torch.Tensor) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class DynamicScaler:
    """Backs off on a non-finite step, grows after `interval` finite ones.

    Overflow flags add up on the device and are read every `poll` steps, so most steps stay
    free of host reads. A flagged step is skipped on the device whatever the scale does.
    """

    def __init__(
        self,
        init: float = 2.0**14,
        growth: float = 2.0,
        backoff: float = 0.5,
        interval: int = 2000,
        maximum: float = 2.0**15,
        poll: int = 8,
    ):
        self.value = init
        self.growth = growth
        self.backoff = backoff
        self.interval = interval
        self.maximum = maximum
        self.good = 0
        self.poll = poll
        self.waiting = 0
        self.flags: torch.Tensor | None = None

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.value

    def unscale(self, grads: list[torch.Tensor]) -> None:
        torch._foreach_mul_(grads, 1.0 / self.value)

    def update(self, bad: torch.Tensor) -> None:
        """Add this step's flag; settle the scale when a poll comes due."""
        self.flags = bad if self.flags is None else self.flags + bad
        self.waiting += 1
        if self.waiting >= self.poll:
            self._settle()

    def _settle(self) -> None:
        """Read the flags: back off on any overflow, otherwise count the clean steps."""
        if self.flags is None:
            return
        overflowed = self.flags.item() > 0
        steps, self.waiting, self.flags = self.waiting, 0, None
        if overflowed:
            self.value *= self.backoff
            self.good = 0
            return
        self.good += steps
        if self.good >= self.interval:
            self.value = min(self.value * self.growth, self.maximum)
            self.good = 0

    def state_dict(self) -> dict:
        """The scale and the unsettled steps; reading it changes nothing, as one rank saves."""
        flagged = self.flags is not None and self.flags.item() > 0
        return {
            "value": self.value,
            "good": self.good,
            "waiting": self.waiting,
            "flagged": flagged,
        }

    def load_state_dict(self, state: dict) -> None:
        self.value = state["value"]
        self.good = state["good"]
        self.waiting = state.get("waiting", 0)
        self.flags = torch.tensor(float(state.get("flagged", False)))
