"""Stream state: causal caches threaded through a stage walk."""

import torch
import torch.nn as nn
from torch import Tensor


class FrameCache:
    """The per-block step of a streaming walk; collects new caches in `new`."""

    def __init__(
        self,
        cache: list[Tensor],
        has_keyframe: bool,
        device: torch.device,
        offload: bool = False,
    ):
        self.cache = cache
        self.has_keyframe = has_keyframe
        self.device = device
        self.store = torch.device("cpu") if offload else device
        self.new: list[Tensor] = []

    def __call__(self, block: nn.Module, x: Tensor) -> Tensor:
        """Run `block.stream` with its cache and store the fresh cache."""
        cached = None
        if self.cache:
            cached = self.cache[len(self.new)].to(self.device, non_blocking=True)
        x, fresh = block.stream(x, self.has_keyframe, cached)
        self.new.append(fresh.to(self.store, non_blocking=True))
        return x
