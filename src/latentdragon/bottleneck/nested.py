"""Nested latents: per-sample channel masks around another bottleneck."""

from dataclasses import replace

import torch
from torch import Tensor, nn

from latentdragon.bottleneck.base import BOTTLENECK, Bottleneck
from latentdragon.bottleneck.vq import VQBottleneck
from latentdragon.types import BottleneckOut
from latentdragon_common.registry import build


def _channel_masks(
    latent_dim: int, latent_chs: list[int], ratio: list[float] | None
) -> Tensor:
    """`(count, C)` prefix masks; `ratio` repeats each mask by its weight."""
    masks = torch.zeros(len(latent_chs), latent_dim)
    for mask, ch in zip(masks, latent_chs):
        mask[:ch] = 1
    if not ratio:
        return masks
    smallest = min(ratio)
    repeats = [int(weight / smallest) for weight in ratio]
    return torch.cat(
        [
            mask[None].repeat(count, 1)
            for mask, count in zip(masks, repeats, strict=True)
        ]
    )


@BOTTLENECK.register("nested")
class NestedBottleneck(Bottleneck):
    """While training, keep `masks[(offset + i) % count]` of the sample `i` latent.

    `unit="partition"` reads `latent_chs` as partition counts of a vq inner bottleneck.
    """

    def __init__(
        self,
        latent_dim: int,
        latent_chs: list[int],
        inner: str | dict | Bottleneck = "identity",
        ratio: list[float] | None = None,
        unit: str = "channel",
        jitter: int = 0,
    ):
        super().__init__()
        self.inner = build(inner, BOTTLENECK, latent_dim=latent_dim)
        self.jitter = jitter

        # Normalize the requested sizes to channel counts
        align = self.inner.quantizer.e_dim if self._is_vq() else 1
        chs = sorted(latent_chs)
        if unit == "partition":
            chs = sorted({1, *chs, latent_dim // self._partition_width()})
            chs = [ch * align for ch in chs]
        elif unit != "channel":
            raise ValueError(f"unit must be 'channel' or 'partition', got {unit!r}")

        # Validate them and make sure the full latent is one of the sizes
        if any(ch % align for ch in chs if ch != latent_dim):
            raise ValueError(f"latent_chs must be multiples of {align} for this vq")
        if not all(0 < ch <= latent_dim for ch in chs):
            raise ValueError(f"latent_chs must be in (0, {latent_dim}], got {chs}")
        if chs[-1] != latent_dim:
            chs.append(latent_dim)
        self.latent_chs = chs

        masks = _channel_masks(latent_dim, chs, ratio)
        self.register_buffer("latent_masks", masks, persistent=False)
        self.mask_count = masks.shape[0]

    def _partition_width(self) -> int:
        """Channels of one partition of a vq inner."""
        if not self._is_vq():
            raise ValueError("unit='partition' needs a vq inner bottleneck")
        return self.inner.quantizer.e_dim

    def _is_vq(self) -> bool:
        """Whether the wrapped bottleneck is a vq."""
        return isinstance(self.inner, VQBottleneck)

    @property
    def codebook_size(self) -> int:
        """Codes of the wrapped vq."""
        return self.inner.codebook_size

    @property
    def quantizer(self):
        """The wrapped vq's quantizer."""
        return self.inner.quantizer

    def in_channels(self, latent_dim: int) -> int:
        """What the wrapped bottleneck needs."""
        return self.inner.in_channels(latent_dim)

    def init_encoder_out(self, layer: nn.Module, latent_dim: int) -> None:
        """Forward to the wrapped bottleneck."""
        self.inner.init_encoder_out(layer, latent_dim)

    def eval_masks(self) -> list[int]:
        """Every channel count, ascending."""
        return list(self.latent_chs)

    def bind_kernels(self, device: torch.device | str) -> None:
        """Forward to the wrapped bottleneck."""
        self.inner.bind_kernels(device)

    def forward(self, raw: Tensor, offset: int = 0) -> BottleneckOut:
        """Run the wrapped bottleneck, then mask each training latent's channels."""
        out = self.inner(raw, offset)
        if not self.training:
            return out
        z = out.z

        # Cycle mask ids over the batch, optionally jittered
        ids = (offset + torch.arange(z.shape[0], device=z.device)) % self.mask_count
        if self.jitter:
            shift = torch.randint(
                -self.jitter, self.jitter + 1, ids.shape, device=ids.device
            )
            ids = (ids + shift).clamp(0, self.mask_count - 1)

        # Broadcast the `(B, C)` masks over the trailing dims
        mask = self.latent_masks[ids].to(z.dtype)
        return replace(out, z=z * mask.reshape(*mask.shape, *[1] * (z.ndim - 2)))
