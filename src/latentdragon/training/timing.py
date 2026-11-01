"""Step-time breakdown; phases accumulate seconds, `report` means them per step."""

import contextlib
import time

import torch

from latentdragon_common.calls import nothing


class StepTimer:
    """Named phases; on a GPU sampled steps synchronize at phase edges."""

    def __init__(self, sample_every: int = 0, device: str = "cpu"):
        self.sample_every = sample_every
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self._drain = torch.cuda.synchronize if "cuda" in device else nothing
        self._edge = nothing

    def start_step(self, step: int) -> None:
        """Select whether this step's phase edges drain the device."""
        sampled = self.sample_every > 0 and step % self.sample_every == 0
        self._edge = self._drain if sampled else nothing

    @contextlib.contextmanager
    def phase(self, name: str):
        """Time a block under `name`."""
        self._edge()
        begin = time.perf_counter()
        yield
        self._edge()
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - begin
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self) -> dict[str, float]:
        """Mean seconds per occurrence of each phase."""
        return {k: self.seconds[k] / self.counts[k] for k in self.seconds}
