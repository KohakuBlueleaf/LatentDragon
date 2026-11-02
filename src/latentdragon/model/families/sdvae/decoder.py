"""The SD-VAE decoder: residual conv stages with nearest upsampling."""

from collections.abc import Callable, Sequence
from fractions import Fraction
from math import ceil
from typing import Any

from optimfactory import mup_init
from torch import Tensor, nn

from latentdragon.model.core.contracts import DECODER, TensorPart
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.reach import Reach, declared, is_local
from latentdragon.model.nn.stages import ConvStage, attn_slots, product


@DECODER.register("sdvae")
class SDVAEDecoder(TensorPart):
    """SD-VAE image decoder; stages are `(scale, hidden, blocks)`, the first runs first."""

    # Its convolutions sit outside the per-function compile scope
    function_scope_gaps = True

    def __init__(
        self,
        in_channels: int,
        stages: Sequence[Sequence[int]],
        attn: Sequence[int] | None = None,
        norm: str | Callable = "group32",
        block: Any = "res",
        input: str = "z",
        output: str = "y",
    ):
        super().__init__(input, output)
        norm_layer = resolve_norm(norm)
        self.scales = [scale for scale, _, _ in stages]
        self.conv_in = nn.Conv2d(in_channels, stages[0][1], 3, 1, 1)

        # One conv stage per config; record the width entering each for attention
        self.stages = nn.ModuleList()
        in_dim = stages[0][1]
        entering = []
        for scale, dim, blocks in stages:
            entering.append(in_dim)
            self.stages.append(
                ConvStage(
                    in_dim, dim, blocks, scale=scale, norm_layer=norm_layer, block=block
                )
            )
            in_dim = dim

        self.attn_blocks = attn_slots(attn, entering, norm_layer)
        self.out_channels = in_dim
        mup_init(self.parameters())

    @property
    def spatial_scale(self) -> int:
        """Total spatial upsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Image decoders never scale frames."""
        return 1

    def reach(self, start: int = 0, tail: int = 0) -> Reach:
        """Reach of stages from `start` plus `tail` output pixels, on that stage's grid."""
        stages = list(self.stages[start:])
        blocks = [block for stage in stages for block in stage.blocks]
        attention = self.attn_blocks[start:]

        # The input conv, then each stage's blocks and its upsampling conv
        spatial = Fraction(0 if start else 1)
        spatial_step = 1
        for scale, stage in zip(self.scales[start:], stages):
            for block in stage.blocks:
                spatial += Fraction(block.spatial_reach or 0, spatial_step)
            spatial_step *= scale
            if scale > 1:
                spatial += Fraction(1, spatial_step)
        spatial += Fraction(tail, spatial_step)

        bounded = declared(blocks) and is_local([*stages, *attention])
        return Reach(ceil(spatial) if bounded else None)

    def tensor(self, z: Tensor) -> Tensor:
        """Decoder input to features; attention precedes each stage."""
        x = self.conv_in(z)
        for stage, attn in zip(self.stages, self.attn_blocks):
            x = stage(attn(x))
        return x
