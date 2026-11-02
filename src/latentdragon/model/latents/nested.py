"""Nested latents: per-sample channel masks of the decoder input, around another def."""

import torch
from torch import Tensor

from latentdragon.model.core import tree
from latentdragon.model.core.contracts import LATENT, Encoded
from latentdragon.model.latents.base import TensorLatent
from latentdragon.model.latents.vq import VQLatent
from latentdragon_common.registry import build


def _channel_masks(
    dim: int, latent_chs: list[int], ratio: list[float] | None
) -> Tensor:
    """`(count, C)` prefix masks; `ratio` repeats each mask by its weight."""
    masks = torch.zeros(len(latent_chs), dim)
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


@LATENT.register("nested")
class NestedLatent(TensorLatent):
    """While training, keep `masks[(offset + i) % count]` of sample `i`'s decoder input.

    The mask travels in the latent as `<output>_mask`, so `deserialize` applies it after the
    wrapped def reads its slot. `dim` is the channel count of the decoder input;
    `unit="partition"` reads `latent_chs` as partition counts of a vq inner def.
    """

    def __init__(
        self,
        dim: int,
        latent_chs: list[int],
        inner: str | dict = "identity",
        ratio: list[float] | None = None,
        unit: str = "channel",
        jitter: int = 0,
        input: str = "h",
        output: str = "z",
    ):
        super().__init__(input, output)
        defaults = {"input": input, "output": output}
        if isinstance(inner, dict) and inner.get("name") in ("vq", "nested"):
            defaults["dim"] = dim
        self.inner = build(inner, LATENT, **defaults)
        self.jitter = jitter
        self.mask_key = f"{output}_mask"

        # Normalize the requested sizes to channel counts
        align = self.inner.quantizer.e_dim if self._is_vq() else 1
        chs = sorted(latent_chs)
        if unit == "partition":
            chs = sorted({1, *chs, dim // self._partition_width()})
            chs = [ch * align for ch in chs]
        elif unit != "channel":
            raise ValueError(f"unit must be 'channel' or 'partition', got {unit!r}")

        # Validate them and make sure the full latent is one of the sizes
        if any(ch % align for ch in chs if ch != dim):
            raise ValueError(f"latent_chs must be multiples of {align} for this vq")
        if not all(0 < ch <= dim for ch in chs):
            raise ValueError(f"latent_chs must be in (0, {dim}], got {chs}")
        if chs[-1] != dim:
            chs.append(dim)
        self.latent_chs = chs

        masks = _channel_masks(dim, chs, ratio)
        self.register_buffer("latent_masks", masks, persistent=False)
        self.mask_count = masks.shape[0]

    def _partition_width(self) -> int:
        """Channels of one partition of a vq inner."""
        if not self._is_vq():
            raise ValueError("unit='partition' needs a vq inner latent")
        return self.inner.quantizer.e_dim

    def _is_vq(self) -> bool:
        """Whether the wrapped def is a vq."""
        return isinstance(self.inner, VQLatent)

    def __getattr__(self, name: str):
        # Capabilities of the wrapped def (codebook_size, quantizer, codes) show through
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("codebook_size", "quantizer", "codes"):
                return getattr(self.inner, name)
            raise

    def init_from_encoder(self, layer) -> None:
        """Forward to the wrapped def."""
        self.inner.init_from_encoder(layer)

    def eval_variants(self) -> list[int]:
        """Every channel count, ascending."""
        return list(self.latent_chs)

    def bind_kernels(self, device) -> None:
        """Forward to the wrapped def."""
        self.inner.bind_kernels(device)

    def track_stats(self, on: bool) -> None:
        """Forward to the wrapped def."""
        self.inner.track_stats(on)

    def raw(self, encoded: Encoded) -> dict:
        """The wrapped def's continuous view."""
        return self.inner.raw(encoded)

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        """Run the wrapped def, then attach each training sample's channel mask."""
        out = self.inner.encode(raw, offset)
        if not self.training:
            return out

        # Cycle mask ids over the batch, optionally jittered
        batch = raw.shape[0]
        ids = (offset + torch.arange(batch, device=raw.device)) % self.mask_count
        if self.jitter:
            shift = torch.randint(
                -self.jitter, self.jitter + 1, ids.shape, device=ids.device
            )
            ids = (ids + shift).clamp(0, self.mask_count - 1)
        out.latent[self.mask_key] = self.latent_masks[ids]
        return out

    def variant(self, latent: dict, key: int) -> dict:
        """The latent with channels from `key` on masked off."""
        first = tree.leaves(latent)[0]
        mask = torch.zeros(
            first.shape[0], self.latent_masks.shape[1], device=first.device
        )
        mask[:, :key] = 1
        return {**latent, self.mask_key: mask}

    def read(self, value) -> Tensor:
        return self.inner.read(value)

    def deserialize(self, latent: dict) -> dict:
        """The wrapped def's decoder input, masked when the latent carries a mask."""
        slots = self.inner.deserialize(latent)
        mask = latent.get(self.mask_key)
        if mask is None:
            return slots
        z = slots[self.output]
        mask = mask.to(z.dtype).reshape(*mask.shape, *[1] * (z.ndim - 2))
        return {**slots, self.output: z * mask}

    def pack(self, latent: dict) -> dict:
        """The wrapped def's packed form; training masks are not stored."""
        return self.inner.pack(latent)
