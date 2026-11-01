"""Encoders: the shuffle skeleton for image and video, and the conv encoder."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from optimfactory import mup_init
from torch import Tensor

from latentdragon.model.layers import resolve_norm
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
    TemporalDownsample,
    make_fuser,
)
from latentdragon_common.registry import Registry

ENCODER = Registry("encoder")


class ShuffleEncoder(nn.Module):
    """Pixel-unshuffle stages; block, conv and fuser are chosen at build."""

    def __init__(
        self,
        image_dim: int,
        latent_dim: int,
        specs: Sequence[StageSpec],
        attn_block_configs: Sequence[int] | None,
        norm_layer: str | Callable,
        block: Any,
        conv: type[nn.Conv2d],
        out_channels: int | None,
    ):
        super().__init__()
        norm_layer = resolve_norm(norm_layer)
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.scales = [spec.scale for spec in specs]
        self.t_scales = [spec.t_scale for spec in specs]

        # Per stage: temporal fuser, block stack and channel projection
        stages, stage_projs, fusers, hidden_dims = [], [], [], []
        dim = image_dim
        for spec in specs:
            dim = dim * spec.scale**2
            hidden = spec.hidden or dim
            fusers.append(make_fuser(TemporalDownsample, hidden, spec.t_scale))
            stages.append(block_stack(block, spec.blocks, hidden, spec.mlp, norm_layer))
            stage_projs.append(conv(dim, hidden, 1) if hidden != dim else nn.Identity())
            hidden_dims.append(hidden)
            dim = hidden

        self.stages = nn.ModuleList(stages)
        self.stage_projs = nn.ModuleList(stage_projs)
        self.temporal_stages = nn.ModuleList(fusers)
        self.attn_blocks = attn_slots(attn_block_configs, hidden_dims, norm_layer)
        self.out_proj = conv(dim, out_channels or latent_dim, 1)
        mup_init(self.parameters())

    @property
    def out_layer(self) -> nn.Module:
        """The latent projection."""
        return self.out_proj

    @property
    def spatial_scale(self) -> int:
        """Total spatial downsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Total temporal downsampling factor."""
        return product(self.t_scales)

    def features(self, x: Tensor, step: Callable = call_block, **frames) -> Tensor:
        """Walk the stages; `step` runs one block."""
        for scale, stage, proj, fuser, attn in zip(
            self.scales,
            self.stages,
            self.stage_projs,
            self.temporal_stages,
            self.attn_blocks,
        ):
            x = fuser(proj(F.pixel_unshuffle(x, scale)), **frames)
            for block in stage:
                x = step(block, x)
            x = attn(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Signal to the raw latent."""
        return self.out_proj(self.features(x))


@ENCODER.register("mlp")
class MLPEncoder(ShuffleEncoder):
    """Image encoder; stage tuples are `(scale, hidden, mlp, blocks)`."""

    def __init__(
        self,
        image_dim: int,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "mlp",
        out_channels: int | None = None,
    ):
        super().__init__(
            image_dim,
            latent_dim,
            [StageSpec.image(cfg) for cfg in block_configs],
            attn_block_configs,
            norm_layer,
            block,
            nn.Conv2d,
            out_channels,
        )


@ENCODER.register("video")
class VideoEncoder(ShuffleEncoder):
    """`(B, T, C, H, W)` to latents with channels at dim 1; no attention slots."""

    def __init__(
        self,
        image_dim: int,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "mlp3d",
        out_channels: int | None = None,
    ):
        if attn_block_configs and any(attn_block_configs):
            raise ValueError("video stages have no attention")
        super().__init__(
            image_dim,
            latent_dim,
            [StageSpec.video(cfg) for cfg in block_configs],
            None,
            norm_layer,
            block,
            SpatialConv2d,
            out_channels,
        )

    def forward(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """Clip `(B, T, C, H, W)` to latents with channels at dim 1."""
        step = partial(call_block, has_keyframe=has_keyframe)
        features = self.features(x, step, has_keyframe=has_keyframe)
        return self.out_proj(features).transpose(1, 2)

    def forward_stream(
        self,
        x: Tensor,
        has_keyframe: bool,
        cache: list[Tensor],
        offload_cache: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode one chunk; pass the returned cache with the next chunk."""
        walk = FrameCache(cache, has_keyframe, x.device, offload_cache)
        features = self.features(x, walk, has_keyframe=has_keyframe)
        return self.out_proj(features).transpose(1, 2), walk.new


@ENCODER.register("conv")
class ConvEncoder(nn.Module):
    """SD-style image encoder; stage tuples are `(scale, hidden, blocks)`."""

    def __init__(
        self,
        image_dim: int,
        latent_dim: int,
        block_configs: Sequence[Sequence[int]],
        attn_block_configs: Sequence[int] | None = None,
        norm_layer: str | Callable = "rms",
        block: Any = "res",
        out_channels: int | None = None,
    ):
        super().__init__()
        norm_layer = resolve_norm(norm_layer)
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.scales = [scale for scale, _, _ in block_configs]
        self.conv_in = nn.Conv2d(image_dim, block_configs[0][1], 3, 1, 1)

        # One downsampling conv stage per config
        self.stages = nn.ModuleList()
        in_dim = block_configs[0][1]
        for scale, dim, blocks in block_configs:
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
        self.conv_out = nn.Conv2d(in_dim, out_channels or latent_dim, 3, 1, 1)
        self.attn_blocks = attn_slots(
            attn_block_configs, [cfg[1] for cfg in block_configs], norm_layer
        )
        mup_init(self.parameters())

    @property
    def out_layer(self) -> nn.Module:
        """The latent projection."""
        return self.conv_out

    @property
    def spatial_scale(self) -> int:
        """Total spatial downsampling factor."""
        return product(self.scales)

    @property
    def temporal_scale(self) -> int:
        """Image encoders never scale frames."""
        return 1

    def forward(self, x: Tensor) -> Tensor:
        """Signal to the raw latent; attention follows each stage."""
        x = self.conv_in(x)
        for stage, attn in zip(self.stages, self.attn_blocks):
            x = attn(stage(x))
        return self.conv_out(self.nonlin_out(self.norm_out(x)))
