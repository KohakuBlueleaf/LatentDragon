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

    def on_step_end(self, state) -> None:
        """Evaluate with EMA weights swapped in, then log the metrics."""
        self.task.eval()
        with self.ema.swapped():
            metrics = self.evaluator.run(self.task, self.sync)
        self.task.train()
        self.metric_log.record(state.step, {**metrics, "epoch": state.epoch})


class SampleCallback(Callback):
    """Writes a sample grid of a held batch to `samples/<step>.webp` and publishes it.

    Rows come from the domain's `to_grid`: the real samples, the first latent tensor of the
    latent def's continuous view as RGB, the reconstruction, then one reconstruction per eval
    variant of the latent def below the full latent. A domain without a grid draws none.
    """

    KEY = "train/samples"

    def __init__(self, every, task, batch, out_dir, autocast, publish):
        self.every = every
        self.task = task
        self.batch = batch
        self.out_dir = Path(out_dir)
        self.autocast = autocast
        self.publish = publish

    @torch.no_grad()
    def on_step_end(self, state) -> None:
        """Reconstruct the held batch, save the grid and send it to the image sinks."""
        domain = self.task.model.domain
        if domain.to_grid is None:
            return
        latent_def = self.task.model.latent
        self.task.eval()
        with self.autocast():
            recon, encoded = self.task.reconstruct(self.batch)
            variants = latent_def.eval_variants()
            partial = [
                self.task.reconstruct(self.batch, key)[0] for key in variants[:-1]
            ]
        self.task.train()

        fakes = [
            tree.map_tensors(lambda t: t.float(), fake) for fake in (recon, *partial)
        ]
        grid = domain.to_grid(self.batch, fakes, preview(latent_def.raw(encoded)))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{state.step}.webp"
        domain.save_grid(grid, path)
        self.publish(state.step, self.KEY, path)


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

    def on_step_end(self, state) -> None:
        """Save the full training state at this step."""
        self.manager.save(state, self.task.saved_modules(), self.bundles, self.ema)
