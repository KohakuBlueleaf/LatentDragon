"""Streaming: latent frames or video frames in chunks, threaded by causal caches."""

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from latentdragon.inference.strategy import DECODE, DecodeStrategy, spatial_overlap
from latentdragon.inference.tiles import stitch
from latentdragon.model import Autoencoder


def chunks(length: int, first: int, step: int) -> Iterator[tuple[int, int]]:
    """`(start, end)` of a first chunk of `first`, then chunks of `step`."""
    start, end = 0, min(first, length)
    while start < length:
        yield start, end
        start, end = end, min(end + step, length)


@DECODE.register("streaming")
class StreamingDecode(DecodeStrategy):
    """Chunks of latent frames; only the causal cache is held between them.

    The first chunk of `first_tokens` holds the keyframe; `offload_cache` keeps the cache
    on the CPU between chunks.
    """

    requires = ("forward_stream",)

    def __init__(
        self,
        first_tokens: int = 2,
        tokens_per_step: int = 1,
        offload_cache: bool = False,
    ):
        self.first_tokens = first_tokens
        self.tokens_per_step = tokens_per_step
        self.offload_cache = offload_cache

    def _spans(self, z: Tensor, has_keyframe: bool) -> Iterator[tuple[int, int]]:
        """Chunk bounds over the frames of `z`; a stream starts at a keyframe."""
        if not has_keyframe:
            raise ValueError("a stream starts at a keyframe")
        return chunks(z.shape[2], self.first_tokens, self.tokens_per_step)

    def stream(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Iterator[Tensor]:
        """Yield `(B, T, C, H, W)` output frames per chunk."""
        cache: list[Tensor] = []
        for start, end in self._spans(z, has_keyframe):
            features, cache = model.decoder.forward_stream(
                z[:, :, start:end], start == 0, cache, self.offload_cache
            )
            yield model.head.tensor(features)

    def __call__(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Tensor:
        return torch.cat(list(self.stream(model, z, has_keyframe)), dim=1)


@DECODE.register("streaming_tiled")
class StreamingTiledDecode(StreamingDecode):
    """`streaming` with spatial tiles, each tile holding its own cache.

    `tile` is `(height, width)` in output pixels and `overlap` is latent cells, None for
    the model's reach.
    """

    def __init__(
        self,
        tile: Sequence[int] = (1024, 1024),
        overlap: int | None = None,
        first_tokens: int = 2,
        tokens_per_step: int = 1,
        offload_cache: bool = False,
    ):
        super().__init__(first_tokens, tokens_per_step, offload_cache)
        self.tile = tuple(tile)
        self.overlap = overlap

    def stream(
        self, model: Autoencoder, z: Tensor, has_keyframe: bool = True
    ) -> Iterator[Tensor]:
        """Yield `(B, T, C, H, W)` output frames per chunk, stitched from tiles."""
        overlap = spatial_overlap(model, self.overlap)
        caches: dict[int, list[Tensor]] = {}

        for start, end in self._spans(z, has_keyframe):

            def decode(part: Tensor, index: int) -> Tensor:
                features, caches[index] = model.decoder.forward_stream(
                    part, start == 0, caches.get(index, []), self.offload_cache
                )
                return model.head.tensor(features)

            yield stitch(
                z[:, :, start:end],
                self.tile,
                overlap,
                model.decoder.spatial_scale,
                decode,
            )


@torch.no_grad()
def stream_encode(
    model: Autoencoder,
    x: Tensor,
    first_frames: int | None = None,
    frames_per_step: int | None = None,
    offload_cache: bool = False,
) -> Iterator[Tensor]:
    """Yield encoder outputs `(B, C, T, H, W)` per chunk of a `(B, T, C, H, W)` clip.

    Defaults to one latent per chunk: the first chunk takes the keyframe and one group.
    """
    encoder = model.encoder
    scale = encoder.temporal_scale
    first = first_frames or 1 + scale
    step = frames_per_step or scale

    cache: list[Tensor] = []
    for start, end in chunks(x.shape[1], first, step):
        raw, cache = encoder.forward_stream(
            x[:, start:end], start == 0, cache, offload_cache
        )
        yield raw
