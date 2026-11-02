"""LatentMaid decoders: pixel-shuffle stages of MLP blocks, for images and video.

Each reads one decoder slot and writes one feature slot, the output latent a head turns into
the signal. `out_channels` is the width of that feature slot. `writer_init="output"`
gives every block's residual writer the output-layer init instead of the hidden one.
"""

from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import partial
from itertools import islice
from math import ceil
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from optimfactory import mup_init
from torch import Tensor

from latentdragon.model.core.contracts import DECODER, TensorPart
from latentdragon.model.nn.blocks import init_writers
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.reach import Reach, declared, is_local
from latentdragon.model.nn.stages import (
    StageSpec,
    attn_slots,
    block_stack,
    call_block,
    product,
)
from latentdragon.model.nn.streaming import FrameCache
from latentdragon.model.nn.temporal import (
    SpatialConv2d,
    TemporalUpsample,
    make_fuser,
)


class ShuffleDecoder(TensorPart):
    """Pixel-shuffle stages run from the last spec to the first."""

    def __init__(
        self,
        in_channels: int,
        specs: Sequence[StageSpec],
        attn: Sequence[int] | None,
        norm: str | Callable,
        block: Any,
        conv: type[nn.Conv2d],
        input: str,
        output: str,
        writer_init: str = "hidden",
    ):
        super().__init__(input, output)
        norm_layer = resolve_norm(norm)

        # Stages run from the last spec to the first
        walk = list(reversed(specs))
        self.scales = [spec.scale for spec in walk]
        self.t_scales = [spec.t_scale for spec in walk]

        # Per stage: channel projection, temporal fuser and block stack
        stage_projs, fusers, stages = [], [], []
        dim = in_channels
        for spec in walk:
            hidden = spec.hidden
            stage_projs.append(conv(dim, hidden, 1) if dim != hidden else nn.Identity())
            fusers.append(make_fuser(TemporalUpsample, hidden, spec.t_scale))
            stages.append(block_stack(block, spec.blocks, hidden, spec.mlp, norm_layer))
            dim = hidden // spec.scale**2

        self.stage_projs = nn.ModuleList(stage_projs)
        self.temporal_fuser = nn.ModuleList(fusers)
        self.stages = nn.ModuleList(stages)

        flags = list(reversed(attn)) if attn else None
        self.attn_blocks = attn_slots(flags, [spec.hidden for spec in walk], norm_layer)
        self.out_channels = dim
        mup_init(self.parameters())
        init_writers(self.stages, writer_init)

    @property
    def spatial_scale(self) -> int:
        """Total spatial upsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Total temporal upsampling factor."""
        return product(self.t_scales)

    def reach(self, start: int = 0, tail: int = 0) -> Reach:
        """Reach of stages from `start` plus `tail` output pixels, on that stage's grid."""
        stages = list(self.stages[start:])
        blocks = [block for stage in stages for block in stage]
        attention = self.attn_blocks[start:]

        # Blocks read frames after the stage's fuser and pixels before its shuffle
        spatial, temporal = Fraction(0), Fraction(0)
        spatial_step, temporal_step = 1, 1
        for scale, t_scale, stage in zip(
            self.scales[start:], self.t_scales[start:], stages
        ):
            temporal_step *= t_scale
            for block in stage:
                spatial += Fraction(block.spatial_reach or 0, spatial_step)
                temporal += Fraction(block.temporal_reach, temporal_step)
            spatial_step *= scale
        spatial += Fraction(tail, spatial_step)

        bounded = declared(blocks) and is_local([*stages, *attention])
        return Reach(ceil(spatial) if bounded else None, ceil(temporal))

    def features(
        self,
        x: Tensor,
        step: Callable = call_block,
        start: int = 0,
        stop: int | None = None,
        **frames,
    ) -> Tensor:
        """Walk stages `start` to `stop`; `step` runs one block."""
        walk = zip(
            self.scales,
            self.stages,
            self.stage_projs,
            self.temporal_fuser,
            self.attn_blocks,
        )
        for scale, stage, proj, fuser, attn in islice(walk, start, stop):
            x = fuser(attn(proj(x)), **frames)
            for block in stage:
                x = step(block, x)
            x = F.pixel_shuffle(x, scale)
        return x

    def tensor(self, z: Tensor) -> Tensor:
        """Decoder input to features."""
        return self.features(z)

    def forward_early(self, z: Tensor, stop: int) -> Tensor:
        """Decoder input through the stages before `stop`."""
        return self.features(z, stop=stop)

    def forward_late(self, x: Tensor, start: int) -> Tensor:
        """`forward_early` output through the stages from `start` on."""
        return self.features(x, start=start)


@DECODER.register("latentmaid")
class LatentMaidDecoder(ShuffleDecoder):
    """Image decoder; stages are `(scale, hidden, mlp, blocks)`, the last runs first."""

    def __init__(
        self,
        in_channels: int,
        stages: Sequence[Sequence[int]],
        attn: Sequence[int] | None = None,
        norm: str | Callable = "rms",
        block: Any = "mlp",
        input: str = "z",
        output: str = "y",
        writer_init: str = "hidden",
    ):
        super().__init__(
            in_channels,
            [StageSpec.image(cfg) for cfg in stages],
            attn,
            norm,
            block,
            nn.Conv2d,
            input,
            output,
            writer_init,
        )


@DECODER.register("latentmaid_video")
class LatentMaidVideoDecoder(ShuffleDecoder):
    """`(B, C, T, H, W)` to `(B, T, C, H, W)` features; stages as the video encoder's."""

    def __init__(
        self,
        in_channels: int,
        stages: Sequence[Sequence[int]],
        norm: str | Callable = "rms",
        block: Any = "mlp3d",
        input: str = "z",
        output: str = "y",
        writer_init: str = "hidden",
    ):
        super().__init__(
            in_channels,
            [StageSpec.video(cfg) for cfg in stages],
            None,
            norm,
            block,
            SpatialConv2d,
            input,
            output,
            writer_init,
        )

    def tensor(self, z: Tensor, has_keyframe: bool = True) -> Tensor:
        """Decoder input `(B, C, T, H, W)` to `(B, T, C, H, W)` features."""
        step = partial(call_block, has_keyframe=has_keyframe)
        return self.features(z.transpose(1, 2), step, has_keyframe=has_keyframe)

    def forward_early(self, z: Tensor, stop: int, has_keyframe: bool = True) -> Tensor:
        """Decoder input through the stages before `stop`, as `(B, T, C, H, W)`."""
        step = partial(call_block, has_keyframe=has_keyframe)
        return self.features(
            z.transpose(1, 2), step, stop=stop, has_keyframe=has_keyframe
        )

    def forward_late(self, x: Tensor, start: int, has_keyframe: bool = True) -> Tensor:
        """`forward_early` output through the stages from `start` on."""
        step = partial(call_block, has_keyframe=has_keyframe)
        return self.features(x, step, start=start, has_keyframe=has_keyframe)

    def forward_stream(
        self,
        z: Tensor,
        has_keyframe: bool,
        cache: list[Tensor],
        offload_cache: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Run one latent chunk to features; pass the cache with the next chunk."""
        walk = FrameCache(cache, has_keyframe, z.device, offload_cache)
        return (
            self.features(z.transpose(1, 2), walk, has_keyframe=has_keyframe),
            walk.new,
        )
