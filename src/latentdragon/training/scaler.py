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
    """Backs off on a non-finite step, grows after `interval` finite ones, all on the device.

    The scale and the count of finite steps are device tensors, created on the loss's device at
    the first `scale`, and `update` changes them with PyTorch's own scale-update kernel, so no
    step reads the host. A flagged step is skipped on the device by the optimizer.
    """

    def __init__(
        self,
        init: float = 2.0**14,
        growth: float = 2.0,
        backoff: float = 0.5,
        interval: int = 2000,
        maximum: float = 2.0**15,
    ):
        self.growth = growth
        self.backoff = backoff
        self.interval = interval
        self.maximum = maximum

        # Host values until the first `scale` gives the device
        self._initial = (init, 0)
        self.current: torch.Tensor | None = None
        self.good: torch.Tensor | None = None

    @property
    def value(self) -> torch.Tensor | float:
        """The scale: a device tensor once bound, the initial value before."""
        return self._initial[0] if self.current is None else self.current

    def _bind(self, device: torch.device) -> None:
        """Move the scale and the finite-step count to `device`."""
        scale, good = self._initial
        self.current = torch.full((), scale, dtype=torch.float32, device=device)
        self.good = torch.full((), good, dtype=torch.int32, device=device)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        if self.current is None:
            self._bind(loss.device)
        return loss * self.current

    def unscale(self, grads: list[torch.Tensor]) -> None:
        torch._foreach_mul_(grads, self.current.reciprocal())

    def update(self, bad: torch.Tensor) -> None:
        """Back off on `bad`, else count a finite step and grow every `interval` of them."""
        found_inf = bad.to(device=self.current.device, dtype=torch.float32)
        torch._amp_update_scale_(
            self.current, self.good, found_inf, self.growth, self.backoff, self.interval
        )
        self.current.clamp_(max=self.maximum)

    def state_dict(self) -> dict:
        """The scale and the finite-step count; the read waits for the device, changes nothing."""
        if self.current is None:
            scale, good = self._initial
            return {"value": scale, "good": good}
        return {"value": self.current.item(), "good": int(self.good.item())}

    def load_state_dict(self, state: dict) -> None:
        self._initial = (float(state["value"]), int(state["good"]))
        if self.current is not None:
            self._bind(self.current.device)
