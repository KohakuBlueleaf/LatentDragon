"""LatentMaid encoders: pixel-unshuffle stages of MLP blocks, for images and video.

Each reads one data slot and writes one encoder slot with channels at dim 1. `out_channels` is
what the latent def downstream reads, e.g. twice the latent channels for `kl`. `writer_init="output"`
gives every block's residual writer the output-layer init instead of the hidden one.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from optimfactory import mup_init
from torch import Tensor

from latentdragon.model.core.contracts import ENCODER, TensorPart
from latentdragon.model.nn.blocks import init_writers
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.stages import (
    StageSpec,
    attn_slots,
    block_stack,
    call_block,
    keep_at,
    masked_step,
    product,
)
from latentdragon.model.nn.streaming import FrameCache
from latentdragon.model.nn.temporal import (
    SpatialConv2d,
    TemporalDownsample,
    make_fuser,
)


class ShuffleEncoder(TensorPart):
    """Pixel-unshuffle stages; block, conv and fuser are chosen at build."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
        self.scales = [spec.scale for spec in specs]
        self.t_scales = [spec.t_scale for spec in specs]

        # Per stage: temporal fuser, block stack and channel projection
        stages, stage_projs, fusers, hidden_dims = [], [], [], []
        dim = in_channels
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
        self.attn_blocks = attn_slots(attn, hidden_dims, norm_layer)
        self.out_proj = conv(dim, out_channels, 1)
        mup_init(self.parameters())
        init_writers(self.stages, writer_init)

    @property
    def last_layer(self) -> nn.Module:
        """The output projection."""
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

    def tensor(self, x: Tensor) -> Tensor:
        """Signal to the encoder output."""
        return self.out_proj(self.features(x))

    def masked_features(self, x: Tensor, keep: Tensor) -> Tensor:
        """`features` of an image as a sparse-conv walk over the kept cells, FCMAE style.

        `keep` is `(B, 1, h, w)` on the feature grid, 1 for a visible cell. Every stage sees whole
        cells, so only the blocks' mixers cross cells, and they read zeros for masked ones: a
        masked pixel never reaches a visible feature, and masked features are exactly zero.
        """
        refusal = mask_refusal(self)
        if refusal:
            raise ValueError(f"masked encoding: {refusal}")
        if x.dim() != 4:
            raise ValueError(f"masked encoding takes images, got {x.dim()}D input")
        hidden = self.features(x * keep_at(keep, x), partial(masked_step, keep=keep))
        return hidden * keep_at(keep, hidden)

    def masked_tensor(self, x: Tensor, keep: Tensor) -> Tensor:
        """`tensor` of a masked image: the encoder output the latent def would read, masked
        cells exactly zero."""
        out = self.out_proj(self.masked_features(x, keep))
        return out * keep_at(keep, out)


def mask_refusal(encoder: ShuffleEncoder) -> str | None:
    """Why `encoder` cannot encode masked images exactly, or None."""
    if any(not isinstance(attn, nn.Identity) for attn in encoder.attn_blocks):
        return "attention mixes masked cells into visible ones"
    for module in encoder.modules():
        if isinstance(module, (nn.GroupNorm, nn.modules.batchnorm._BatchNorm)):
            return f"{type(module).__name__} statistics span masked cells"
    return None


@ENCODER.register("latentmaid")
class LatentMaidEncoder(ShuffleEncoder):
    """Image encoder; stages are `(scale, hidden, mlp, blocks)`."""

    def __init__(
        self,
        out_channels: int,
        stages: Sequence[Sequence[int]],
        in_channels: int = 3,
        attn: Sequence[int] | None = None,
        norm: str | Callable = "rms",
        block: Any = "mlp",
        input: str = "x",
        output: str = "h",
        writer_init: str = "hidden",
    ):
        super().__init__(
            in_channels,
            out_channels,
            [StageSpec.image(cfg) for cfg in stages],
            attn,
            norm,
            block,
            nn.Conv2d,
            input,
            output,
            writer_init,
        )


@ENCODER.register("latentmaid_video")
class LatentMaidVideoEncoder(ShuffleEncoder):
    """`(B, T, C, H, W)` to `(B, C, T, H, W)`; stages are `(t_scale, scale, hidden, mlp, blocks)`."""

    def __init__(
        self,
        out_channels: int,
        stages: Sequence[Sequence[int]],
        in_channels: int = 3,
        norm: str | Callable = "rms",
        block: Any = "mlp3d",
        input: str = "x",
        output: str = "h",
        writer_init: str = "hidden",
    ):
        super().__init__(
            in_channels,
            out_channels,
            [StageSpec.video(cfg) for cfg in stages],
            None,
            norm,
            block,
            SpatialConv2d,
            input,
            output,
            writer_init,
        )

    def tensor(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """Clip `(B, T, C, H, W)` to `(B, C, T, H, W)`."""
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
