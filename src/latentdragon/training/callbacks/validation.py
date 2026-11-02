"""Validation, sample logging and checkpoint callbacks."""

from pathlib import Path

import torch

from latentdragon.domain.latents import pca_rgb
from latentdragon.model.core import tree
from latentdragon.training.callbacks.runner import Callback


class ValidationCallback(Callback):
    """Runs the evaluator on EMA weights and logs `val/*`."""

    at_end = True

    def __init__(self, every, evaluator, task, ema, sync, metric_log):
        self.every = every
        self.evaluator = evaluator
        self.task = task
        self.ema = ema
        self.sync = sync
        self.metric_log = metric_log

    def on_step_end(self, state, batch=None) -> None:
        """Evaluate with EMA weights swapped in, then log the metrics."""
        self.task.eval()
        with self.ema.swapped():
            metrics = self.evaluator.run(self.task, self.sync)
        self.task.train()
        self.metric_log.record(state.step, {**metrics, "epoch": state.epoch})


class SampleCallback(Callback):
    """Writes sample grids and publishes them: the step's own batch and a held batch.

    The first (up to) `count` items of the batch the step trained on go to
    `samples/train/<step>.webp` under `train/samples`; the held batch, the same at every step,
    goes to `samples/fixed/<step>.webp` under `train/samples_fixed`. Rows come from the
    domain's `to_grid`: the real samples, the first latent tensor of the latent def's continuous
    view as RGB, the reconstruction, then one reconstruction per eval variant of the latent def
    below the full latent. A domain without a grid draws none.
    """

    def __init__(self, every, task, held, out_dir, autocast, publish, count: int = 8):
        self.every = every
        self.task = task
        self.held = held
        self.out_dir = Path(out_dir)
        self.autocast = autocast
        self.publish = publish
        self.count = count

    @torch.no_grad()
    def on_step_end(self, state, batch=None) -> None:
        """Reconstruct both batches, save the grids and send them to the image sinks."""
        if self.task.model.domain.to_grid is None:
            return
        self.task.eval()
        if batch is not None:
            current = tree.map_tensors(lambda t: t[: self.count], batch)
            self._grid(current, state.step, "train", "train/samples")
        self._grid(self.held, state.step, "fixed", "train/samples_fixed")
        self.task.train()

    def _grid(self, batch, step: int, folder: str, key: str) -> None:
        """One grid of `batch` to `samples/<folder>/<step>.webp`, published under `key`."""
        domain = self.task.model.domain
        latent_def = self.task.model.latent
        with self.autocast():
            recon, encoded = self.task.reconstruct(batch)
            variants = latent_def.eval_variants()
            partial = [self.task.reconstruct(batch, v)[0] for v in variants[:-1]]

        fakes = [
            tree.map_tensors(lambda t: t.float(), fake) for fake in (recon, *partial)
        ]
        grid = domain.to_grid(batch, fakes, preview(latent_def.raw(encoded)))

        folder = self.out_dir / folder
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{step}.webp"
        domain.save_grid(grid, path)
        self.publish(step, key, path)


def preview(raw: dict) -> torch.Tensor | None:
    """The first float latent tensor with channels at dim 1 and spatial dims, as RGB."""
    for value in tree.named_tensors(raw).values():
        if value.is_floating_point() and value.ndim >= 4:
            return pca_rgb(value)
    return None


class CheckpointCallback(Callback):
    """Saves every interval and after the final step."""

    at_end = True

    def __init__(self, every, manager, task, bundles, ema):
        self.every = every
        self.manager = manager
        self.task = task
        self.bundles = bundles
        self.ema = ema

    def on_step_end(self, state, batch=None) -> None:
        """Save the full training state at this step."""
        self.manager.save(state, self.task.saved_modules(), self.bundles, self.ema)
