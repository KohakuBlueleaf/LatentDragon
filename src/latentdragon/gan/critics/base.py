"""The critic registry and contract."""

import torch.nn as nn

from latentdragon_common.registry import Registry

CRITIC = Registry("critic")


class Critic(nn.Module):
    """Signal to raw logit map; `tracks_stats` marks state a forward mutates."""

    tracks_stats = False
