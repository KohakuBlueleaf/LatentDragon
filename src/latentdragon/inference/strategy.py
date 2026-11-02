"""Decode strategies: how a decoder input becomes output when it may not fit at once.

Strategies work on a single-slot decoder and head through their slot-local `tensor` calls,
plus the decoder capabilities each one names in `requires`.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from latentdragon.inference.tiles import causal_spans, stitch
from latentdragon.model import Autoencoder
from latentdragon_common.registry import Registry, build

DECODE = Registry("decode")


def is_video(z: Tensor) -> bool:
    """True for `(B, C, T, H, W)` latents."""
    return z.ndim == 5


def spatial_overlap(model: Autoencoder, overlap: int | None, start: int = 0) -> int:
    """`overlap` when set, otherwise the model's reach on the grid of stage `start`."""
    if overlap is not None:
        return overlap
    reach = model.reach(start).spatial
    if reach is None:
        raise ValueError(
            "this model holds attention or group norm, so no overlap is exact; "
            "set `overlap` to accept an approximate result"
        )
    return reach


class DecodeStrategy:
    """Decoder input to output; `requires` names the decoder methods the strategy calls."""

    requires: tuple[str, ...] = ()

    def check(self, model: Autoencoder) -> None:
        """Raise when the decoder or head lacks a required ability."""
        for part in (model.decoder, model.head):
            if not hasattr(part, "tensor"):
                kind = type(part).__name__
                raise ValueError(f"{kind} has no single-slot `tensor` call to tile")
        missing = [name for name in self.requires if not hasattr(model.decoder, name)]
        if missing:
            kind = type(model.decoder).__name__
            raise ValueError(f"{kind} cannot do this decode; needs {missing}")

    def __call__(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Tensor:
        raise NotImplementedError


@DECODE.register("full")
class FullDecode(DecodeStrategy):
    """The whole latent at once."""

    def __call__(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Tensor:
        frames = {"has_keyframe": has_keyframe} if is_video(z) else {}
        return model.head.tensor(model.decoder.tensor(z, **frames))


@DECODE.register("tiled")
class TiledDecode(DecodeStrategy):
    """Tiles of the latent, optionally in time, each decoded by `inner`.

    `tile` is `(height, width)` in output pixels and `tile_t` is output frames, 0 for no
    temporal tiling. `overlap` and `overlap_t` are latent cells, None for the model's reach.
    """

    def __init__(
        self,
        tile: Sequence[int] = (512, 512),
        tile_t: int = 0,
        overlap: int | None = None,
        overlap_t: int | None = None,
        inner: str | dict = "full",
    ):
        self.tile = tuple(tile)
        self.tile_t = tile_t
        self.overlap = overlap
        self.overlap_t = overlap_t
        self.inner = build(inner, DECODE)

    def check(self, model: Autoencoder) -> None:
        """Check this strategy and the one decoding each tile."""
        super().check(model)
        self.inner.check(model)

    def __call__(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Tensor:
        if self.tile_t and is_video(z):
            return self._over_time(model, z, has_keyframe)
        return self._over_space(model, z, has_keyframe)

    def _over_space(self, model: Autoencoder, z: Tensor, has_keyframe: bool) -> Tensor:
        """Stitch spatial tiles of one latent."""
        overlap = spatial_overlap(model, self.overlap)
        return stitch(
            z,
            self.tile,
            overlap,
            model.decoder.spatial_scale,
            lambda part, _: self.inner(model, part, has_keyframe),
        )

    def _over_time(self, model: Autoencoder, z: Tensor, has_keyframe: bool) -> Tensor:
        """Decode spans of latent frames and keep the frames past each span's context."""
        scale = model.decoder.temporal_scale
        overlap = self.overlap_t
        if overlap is None:
            overlap = model.reach().temporal

        # A keyframe latent decodes to one frame, every other latent to `scale`
        def frames_before(latent_frame: int) -> int:
            return 1 + (latent_frame - 1) * scale if latent_frame else 0

        pieces = []
        for span in causal_spans(z.shape[2], self.tile_t // scale, overlap):
            part = z[:, :, span.start : span.end]
            keyframe = has_keyframe and span.start == 0
            decoded = self._over_space(model, part, keyframe)

            kept = frames_before(span.keep_end) - frames_before(span.keep_start)
            pieces.append(decoded[:, -kept:])
        return torch.cat(pieces, dim=1)


@DECODE.register("staged_tiled")
class StagedTiledDecode(DecodeStrategy):
    """The stages before `stages` run whole; later stages run in spatial tiles.

    `tile` is `(height, width)` in output pixels and `overlap` is cells of the grid at
    stage `stages`, None for the model's reach.
    """

    requires = ("forward_early", "forward_late", "scales")

    def __init__(
        self,
        stages: int = 2,
        tile: Sequence[int] = (512, 512),
        overlap: int | None = None,
    ):
        self.stages = stages
        self.tile = tuple(tile)
        self.overlap = overlap

    def __call__(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Tensor:
        frames = {"has_keyframe": has_keyframe} if is_video(z) else {}
        decoder = model.decoder
        early = decoder.forward_early(z, self.stages, **frames)

        # The late stages' own scale, since the tiles live on their input grid
        scale = 1
        for value in decoder.scales[self.stages :]:
            scale *= value

        return stitch(
            early,
            self.tile,
            spatial_overlap(model, self.overlap, self.stages),
            scale,
            lambda part, _: model.head.tensor(
                decoder.forward_late(part, self.stages, **frames)
            ),
        )
