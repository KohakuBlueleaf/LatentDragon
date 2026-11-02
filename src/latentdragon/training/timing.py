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
        self.starts: list[float] = []
        self._drain = torch.cuda.synchronize if "cuda" in device else nothing
        self._edge = nothing

    def start_step(self, step: int) -> None:
        """Stamp the step and select whether its phase edges drain the device."""
        self.starts.append(time.perf_counter())
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
        """Mean seconds per occurrence of each phase, and the step interval stats."""
        report = {k: self.seconds[k] / self.counts[k] for k in self.seconds}
        return {**report, **self._intervals()}

    def _intervals(self) -> dict[str, float]:
        """Median and spread of the host interval between steps, past the warm-up tenth."""
        gaps = [b - a for a, b in zip(self.starts, self.starts[1:])]
        kept = sorted(gaps[len(gaps) // 10 :])
        if not kept:
            return {}
        return {
            "step_median": kept[len(kept) // 2],
            "step_p10": kept[len(kept) // 10],
            "step_p90": kept[len(kept) * 9 // 10],
            "step_mean": sum(kept) / len(kept),
            "steps_timed": len(kept),
        }
