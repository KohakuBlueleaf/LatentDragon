"""Trunks: decoder bodies up to, not including, the head."""

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

from latentdragon.model.layers import resolve_norm
from latentdragon.model.reach import Reach, declared, is_local
from latentdragon.model.stages import (
    ConvStage,
    StageSpec,
    attn_slots,
    block_stack,
    call_block,
    product,
)
from latentdragon.model.streaming import FrameCache
from latentdragon.model.temporal import (
    SpatialConv2d,
    TemporalUpsample,
    make_fuser,
)
from latentdragon_common.registry import Registry

TRUNK = Registry("trunk")


class ShuffleTrunk(nn.Module):
    """Pixel-shuffle stages run from the last spec to the first."""

    def __init__(
        self,
        latent_dim: int,
        specs: Sequence[StageSpec],
        attn_block_configs: Sequence[int] | None,
        norm_layer: str | Callable,
        block: Any,
        conv: type[nn.Conv2d],
    ):
        super().__init__()
        norm_layer = resolve_norm(norm_layer)
        self.latent_dim = latent_dim

        # Stages run from the last spec to the first
        walk = list(reversed(specs))
        self.scales = [spec.scale for spec in walk]
        self.t_scales = [spec.t_scale for spec in walk]

        # Per stage: channel projection, temporal fuser and block stack
        stage_projs, fusers, stages = [], [], []
        dim = latent_dim
        for spec in walk:
            hidden = spec.hidden
            stage_projs.append(conv(dim, hidden, 1) if dim != hidden else nn.Identity())
            fusers.append(make_fuser(TemporalUpsample, hidden, spec.t_scale))
            stages.append(block_stack(block, spec.blocks, hidden, spec.mlp, norm_layer))
            dim = hidden // spec.scale**2

        self.stage_projs = nn.ModuleList(stage_projs)
        self.temporal_fuser = nn.ModuleList(fusers)
        self.stages = nn.ModuleList(stages)

        flags = list(reversed(attn_block_configs)) if attn_block_configs else None
        self.attn_blocks = attn_slots(flags, [spec.hidden for spec in walk], norm_layer)
        self.out_dim = dim
        mup_init(self.parameters())

    @property
    def spatial_scale(self) -> int:
        """Total spatial upsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Total temporal upsampling factor."""
        return product(self.t_scales)

    @property
    def head_form(self) -> dict[str, Any]:
        """Head arguments matching this trunk's output."""
        return {"kernel_size": 1}

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

    def forward(self, z: Tensor) -> Tensor:
        """Latents to trunk features."""
        return self.features(z)

    def forward_early(self, z: Tensor, stop: int) -> Tensor:
        """Latents through the stages before `stop`."""
        return self.features(z, stop=stop)

    def forward_late(self, x: Tensor, start: int) -> Tensor:
        """`forward_early` output through the stages from `start` on."""
        return self.features(x, start=start)


@TRUNK.register("mlp")
class MLPTrunk(ShuffleTrunk):
    """Image trunk; stage tuples are `(scale, hidden, mlp, blocks)`, last runs first."""

    def __init__(
        self,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "mlp",
    ):
        super().__init__(
            latent_dim,
            [StageSpec.image(cfg) for cfg in block_configs],
            attn_block_configs,
            norm_layer,
            block,
            nn.Conv2d,
        )


@TRUNK.register("video")
class VideoTrunk(ShuffleTrunk):
    """Latents with channels at dim 1 to `(B, T, C, H, W)` features."""

    def __init__(
        self,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "mlp3d",
    ):
        if attn_block_configs and any(attn_block_configs):
            raise ValueError("video stages have no attention")
        super().__init__(
            latent_dim,
            [StageSpec.video(cfg) for cfg in block_configs],
            None,
            norm_layer,
            block,
            SpatialConv2d,
        )

    @property
    def head_form(self) -> dict[str, Any]:
        """Head arguments matching this trunk's output."""
        return {"kernel_size": 1, "conv": SpatialConv2d}

    def forward(self, z: Tensor, has_keyframe: bool = True) -> Tensor:
        """Latents `(B, C, T, H, W)` to `(B, T, C, H, W)` features."""
        step = partial(call_block, has_keyframe=has_keyframe)
        return self.features(z.transpose(1, 2), step, has_keyframe=has_keyframe)

    def forward_early(self, z: Tensor, stop: int, has_keyframe: bool = True) -> Tensor:
        """Latents `(B, C, T, H, W)` through the stages before `stop`, as `(B, T, C, H, W)`."""
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


@TRUNK.register("conv")
class ConvTrunk(nn.Module):
    """SD-style image trunk; tuples are `(scale, hidden, blocks)`, first runs first."""

    def __init__(
        self,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "res",
    ):
        super().__init__()
        norm_layer = resolve_norm(norm_layer)
        self.latent_dim = latent_dim
        self.norm_layer = norm_layer
        self.scales = [scale for scale, _, _ in block_configs]
        self.conv_in = nn.Conv2d(latent_dim, block_configs[0][1], 3, 1, 1)

        # One conv stage per config; record the width entering each for attention
        self.stages = nn.ModuleList()
        in_dim = block_configs[0][1]
        entering = []
        for scale, dim, blocks in block_configs:
            entering.append(in_dim)
            self.stages.append(
                ConvStage(
                    in_dim, dim, blocks, scale=scale, norm_layer=norm_layer, block=block
                )
            )
            in_dim = dim

        self.attn_blocks = attn_slots(attn_block_configs, entering, norm_layer)
        self.out_dim = in_dim
        mup_init(self.parameters())

    @property
    def spatial_scale(self) -> int:
        """Total spatial upsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Image trunks never scale frames."""
        return 1

    @property
    def head_form(self) -> dict[str, Any]:
        """Head arguments matching this trunk's output."""
        return {"kernel_size": 3, "norm_layer": self.norm_layer, "act": nn.Mish}

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

    def forward(self, z: Tensor) -> Tensor:
        """Latents to trunk features; attention precedes each stage."""
        x = self.conv_in(z)
        for stage, attn in zip(self.stages, self.attn_blocks):
            x = stage(attn(x))
        return x
