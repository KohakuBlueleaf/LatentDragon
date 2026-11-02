"""Callback cadences, precomputed: `every` steps calls `on_step_end(state, batch)`."""

import torch


class Callback:
    """Base callback; `every` is in steps and `at_end` also fires after the last step.

    `batch` is the device batch the step trained on; the final call after the loop has none.
    """

    every = 0
    at_end = False

    def on_step_end(self, state, batch=None) -> None:
        raise NotImplementedError


class CallbackRunner:
    """Runs due callbacks at the step edge."""

    def __init__(self, callbacks: list[Callback]):
        self.periodic = [c for c in callbacks if c.every > 0]
        self.final = [c for c in callbacks if c.at_end]

    def step_end(self, state, batch=None) -> None:
        """Fire every callback whose cadence divides the step."""
        for callback in self.periodic:
            if state.step % callback.every == 0:
                with torch.cuda.nvtx.range(type(callback).__name__):
                    callback.on_step_end(state, batch)

    def finish(self, state) -> None:
        """Fire the final-step callbacks that did not already run on this step."""
        for callback in self.final:
            if callback.every <= 0 or state.step % callback.every != 0:
                callback.on_step_end(state)
