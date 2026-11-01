"""Weight EMA keyed by parameter name, held outside any module."""

import contextlib

import torch


class NoEMA:
    """The EMA-off policy."""

    def update(self) -> None:
        return None

    def swapped(self):
        return contextlib.nullcontext()

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class NamedEMA:
    """Shadow copies with the warm-up decay `min(decay, (1 + n) / (10 + n))`."""

    def __init__(self, named: dict[str, torch.nn.Parameter], decay: float = 0.999):
        self.named = named
        self.decay = decay
        self.updates = 0
        self.shadow = {k: p.detach().clone() for k, p in named.items()}
        self._live = list(named.values())
        self._copies = [self.shadow[k] for k in named]

    @torch.no_grad()
    def update(self) -> None:
        """Advance every shadow toward its live parameter."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        torch._foreach_lerp_(self._copies, self._live, 1.0 - decay)

    @contextlib.contextmanager
    def swapped(self):
        """Run with the shadow weights in the live parameters."""
        backup = [p.data for p in self._live]
        for param, copy in zip(self._live, self._copies):
            param.data = copy
        try:
            yield
        finally:
            for param, data in zip(self._live, backup):
                param.data = data

    def state_dict(self) -> dict:
        return {"updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.updates = state["updates"]
        for key, value in state["shadow"].items():
            if key in self.shadow:
                self.shadow[key].copy_(value)
