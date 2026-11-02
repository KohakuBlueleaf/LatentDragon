"""The SD-VAE encoder: residual conv stages with stride-2 downsampling."""

from collections.abc import Callable, Sequence
from typing import Any

import torch.nn as nn
from optimfactory import mup_init
from torch import Tensor

from latentdragon.model.core.contracts import ENCODER, TensorPart
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.stages import ConvStage, attn_slots, product


@ENCODER.register("sdvae")
class SDVAEEncoder(TensorPart):
    """SD-VAE image encoder; stages are `(scale, hidden, blocks)`."""

    def __init__(
        self,
        out_channels: int,
        stages: Sequence[Sequence[int]],
        in_channels: int = 3,
        attn: Sequence[int] | None = None,
        norm: str | Callable = "group32",
        block: Any = "res",
        input: str = "x",
        output: str = "h",
    ):
        super().__init__(input, output)
        norm_layer = resolve_norm(norm)
        self.scales = [scale for scale, _, _ in stages]
        self.conv_in = nn.Conv2d(in_channels, stages[0][1], 3, 1, 1)

        # One downsampling conv stage per config
        self.stages = nn.ModuleList()
        in_dim = stages[0][1]
        for scale, dim, blocks in stages:
            self.stages.append(
                ConvStage(
                    in_dim,
                    dim,
                    blocks,
                    scale=scale,
                    norm_layer=norm_layer,
                    downsample=True,
                    block=block,
                )
            )
            in_dim = dim

        # Output projection and attention slots
        self.norm_out = norm_layer(in_dim)
        self.nonlin_out = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dim, out_channels, 3, 1, 1)
        self.attn_blocks = attn_slots(attn, [cfg[1] for cfg in stages], norm_layer)
        mup_init(self.parameters())

    @property
    def last_layer(self) -> nn.Module:
        """The output projection."""
        return self.conv_out

    @property
    def spatial_scale(self) -> int:
        """Total spatial downsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Image encoders never scale frames."""
        return 1

    def tensor(self, x: Tensor) -> Tensor:
        """Signal to the encoder output; attention follows each stage."""
        x = self.conv_in(x)
        for stage, attn in zip(self.stages, self.attn_blocks):
            x = attn(stage(x))
        return self.conv_out(self.nonlin_out(self.norm_out(x)))
