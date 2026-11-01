"""Epoch-seeded sampler and a batch stream that reports its position for resume."""

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, Sampler


class EpochSampler(Sampler):
    """A rank's share of a seeded permutation, starting at a batch offset."""

    def __init__(self, size: int, batch_size: int, rank=0, world=1, seed=0):
        self.size = size
        self.batch_size = batch_size
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = 0
        self.skip = 0

    def set_position(self, epoch: int, batches: int) -> None:
        """Start at `batches` already consumed in `epoch`."""
        self.epoch = epoch
        self.skip = batches * self.batch_size

    def __len__(self) -> int:
        return self.size // self.world - self.skip

    def __iter__(self):
        # Same permutation on every rank; each rank takes a strided slice
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)
        share = order[self.rank :: self.world][: self.size // self.world]
        return iter(share[self.skip :].tolist())


def epoch_batches(
    source,
    batch_size: int,
    workers: int = 4,
    rank: int = 0,
    world: int = 1,
    seed: int = 0,
    epoch: int = 0,
    batches: int = 0,
    pin_memory: bool = True,
) -> Iterator[tuple]:
    """Endless `(batch, epoch, batches_consumed)`; full batches only."""
    sampler = EpochSampler(len(source), batch_size, rank, world, seed)
    sampler.set_position(epoch, batches)
    loader = DataLoader(
        source,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
    )

    # Roll over to the next epoch with a fresh permutation when one is exhausted
    while True:
        for batch in loader:
            batches += 1
            yield batch, epoch, batches
        epoch += 1
        batches = 0
        sampler.set_position(epoch, 0)
