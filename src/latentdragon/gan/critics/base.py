"""The critic registry and contract."""

import torch
import torch.nn as nn

from latentdragon_common.registry import Registry

CRITIC = Registry("critic")


class Critic(nn.Module):
    """A slot dict to a raw logit map; `tracks_stats` marks state a forward mutates.

    A critic names the data slot it reads with `slot`; `judge` maps that slot's tensor.
    """

    tracks_stats = False

    def __init__(self, slot: str = "x"):
        super().__init__()
        self.slot = slot

    def forward(self, sample: dict) -> torch.Tensor:
        """Logits of the critic's slot."""
        return self.judge(sample[self.slot].contiguous())

    def judge(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
