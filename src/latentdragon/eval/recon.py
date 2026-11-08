"""Validation metrics per data slot: rFID, PSNR and SSIM, plus total code usage."""

import copy

import torch
import torch.distributed as dist
from sensecraft.metrics import psnr, ssim
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from latentdragon.eval.usage import CodeTally, NoTally
from latentdragon.model.core import tree
from latentdragon.training.parallel.sync import barrier


def new_fid(device) -> FrechetInceptionDistance:
    """An FID metric keeping its real features across resets.

    The first build downloads the Inception weights, so rank 0 builds first and the other
    ranks wait, then load the cached file instead of downloading it at the same time.
    """
    distributed = dist.is_available() and dist.is_initialized()
    first = not distributed or dist.get_rank() == 0
    if not first:
        barrier()
    metric = FrechetInceptionDistance(reset_real_features=False, normalize=True)
    if distributed and first:
        barrier()
    return metric.to(device).eval().requires_grad_(False)


def share_fid(first: FrechetInceptionDistance) -> FrechetInceptionDistance:
    """A second metric reusing `first`'s Inception and real statistics."""
    return copy.deepcopy(first, {id(first.inception): first.inception})


def _psnr(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return psnr(recon, real).mean()


def _ssim(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return ssim(recon, real, 1.0).mean()


FRAME_METRICS = {"psnr": _psnr, "ssim": _ssim}


class ReconEvaluator:
    """Holds a rank's validation samples; `run(task)` returns `val/*` metrics.

    Each data slot the domain lists gets its metrics; a single slot keeps the plain
    `val/<metric>` names, several get `val/<slot>/<metric>`. Every eval variant of the latent
    def is evaluated separately with a `-<variant>` suffix.
    """

    def __init__(self, dataset, batch_size, workers, device, precision, sampler=None):
        self.device = device
        self.device_type = torch.device(device).type
        self.precision = precision
        self.batch_size = batch_size
        self.loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, sampler=sampler
        )
        self.samples = None
        self.fids: dict[tuple, FrechetInceptionDistance] = {}

    def _prepare(self, domain, keeps: list) -> None:
        """Read the reference once; one Inception per slot feeds every variant's FID."""
        batches = list(self.loader)
        self.samples = tree.cat(batches)
        for slot, names in domain.metrics.items():
            if "rfid" not in names:
                continue
            first = new_fid(self.device)
            frames = domain.to_frames[slot]
            for batch in batches:
                real = frames(batch[slot].to(self.device)).float() * 0.5 + 0.5
                with torch.autocast(self.device_type, enabled=False):
                    first.update(real, True)
            for keep in keeps[1:]:
                self.fids[slot, keep] = share_fid(first)
            self.fids[slot, keeps[0]] = first

    @torch.no_grad()
    def run(self, task, sync) -> dict:
        """Evaluate every latent variant, or the whole latent."""
        model = task.model
        domain = model.domain
        variants = model.latent.eval_variants()
        keeps = list(variants) if variants else [None]
        if self.samples is None:
            self._prepare(domain, keeps)

        # Accumulators: one row of frame metrics per (slot, variant)
        names = {
            slot: [n for n in found if n in FRAME_METRICS]
            for slot, found in domain.metrics.items()
        }
        totals = {
            (slot, keep): torch.zeros(len(names[slot]), device=self.device)
            for slot in names
            for keep in keeps
        }
        size = getattr(model.latent, "codebook_size", 0)
        tally = CodeTally(model.latent, self.device) if size else NoTally()
        seen = torch.zeros((), device=self.device)
        for fid in self.fids.values():
            fid.reset()

        # Reconstruct every batch at every variant and accumulate metrics
        count = tree.batch_size(self.samples)
        for start in range(0, count, self.batch_size):
            part = tree.map_tensors(
                lambda t: t[start : start + self.batch_size], self.samples
            )
            batch = tree.to_device(part, self.device)
            size_now = tree.batch_size(batch)
            seen += size_now
            for keep in keeps:
                with self.precision.autocast(self.device_type):
                    recon, encoded = task.reconstruct(batch, keep)
                self._accumulate(domain, names, batch, recon, keep, size_now, totals)
            tally.add(model.latent.codes(encoded) if size else None)
        sync.all_sum_(seen)
        return self._metrics(names, keeps, totals, seen, tally, sync)

    def _accumulate(self, domain, names, batch, recon, keep, size, totals) -> None:
        """Frame metrics and FID features of one batch at one variant."""
        for slot, found in names.items():
            frames = domain.to_frames[slot]
            real = frames(batch[slot].float()) * 0.5 + 0.5
            fake = frames(recon[slot].float().clamp(-1, 1)) * 0.5 + 0.5
            with torch.autocast(self.device_type, enabled=False):
                if (slot, keep) in self.fids:
                    self.fids[slot, keep].update(fake, False)
                if found:
                    totals[slot, keep] += torch.stack(
                        [FRAME_METRICS[n](fake, real) * size for n in found]
                    )

    def _metrics(self, names, keeps, totals, seen, tally, sync) -> dict:
        """Reduce totals across ranks into `val/*` metrics."""
        single = len(names) == 1
        metrics = {}
        for (slot, keep), total in totals.items():
            suffix = "" if keep is None else f"-{keep}"
            section = "val" if single else f"val/{slot}"
            sync.all_sum_(total)
            values = total / seen
            if (slot, keep) in self.fids:
                metrics[f"{section}/rfid{suffix}"] = (
                    self.fids[slot, keep].compute().item()
                )
            for index, name in enumerate(names[slot]):
                metrics[f"{section}/{name}{suffix}"] = values[index].item()
        metrics.update(tally.report(sync))
        return metrics
