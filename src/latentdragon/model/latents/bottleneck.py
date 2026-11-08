"""Bottleneck latents: encoder width down to the latent, the wrapped def, latent up to the
decoder width, so encoder and decoder never see which latent sits between them."""

from optimfactory import mup_init
from torch import Tensor, nn

from latentdragon.model.core.contracts import LATENT, Encoded
from latentdragon.model.latents.base import TensorLatent
from latentdragon_common.registry import build


@LATENT.register("bottleneck")
class Bottleneck(TensorLatent):
    """`h (in_channels) -> 1x1 down -> inner -> 1x1 up -> decoder input (out_channels)`.

    The encoder output stays at its own width, the state latent terms read; the wrapped def
    (identity, kl, vq, nested) works on `dim` channels and its latent is what gets stored. The
    down conv writes `dim` times the wrapped def's `input_factor` (twice for `kl`); `out_channels`
    None leaves out the up conv, so the decoder reads the latent itself. Both convs take the init
    of the encoder and decoder 1x1s they replace. A nested def goes inside, so its masks act on
    the latent before the up conv.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        out_channels: int | None = None,
        inner: str | dict = "identity",
        input: str = "h",
        output: str = "z",
    ):
        super().__init__(input, output)
        defaults = {"input": input, "output": output}
        if isinstance(inner, dict) and inner.get("name") in ("vq", "nested"):
            defaults["dim"] = dim
        self.inner = build(inner, LATENT, **defaults)
        self.down = nn.Conv2d(in_channels, dim * self.inner.input_factor, 1)
        self.up = None if out_channels is None else nn.Conv2d(dim, out_channels, 1)
        mup_init([*self.down.parameters(), *(self.up.parameters() if self.up else ())])
        self.inner.init_from_encoder(self.down)

    def __getattr__(self, name: str):
        # Capabilities of the wrapped def (codebook_size, quantizer, codes) show through
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("codebook_size", "quantizer", "codes"):
                return getattr(self.inner, name)
            raise

    def init_from_encoder(self, layer: nn.Module) -> None:
        """Nothing: the wrapped def was fitted to this def's own down conv."""

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        """Down, then the wrapped def."""
        return self.inner.encode(self.down(raw), offset)

    def read(self, value) -> Tensor:
        return self.inner.read(value)

    def deserialize(self, latent: dict) -> dict:
        """The wrapped def's decoder input, masks applied, then up."""
        slots = self.inner.deserialize(latent)
        if self.up is None:
            return slots
        return {**slots, self.output: self.up(slots[self.output])}

    def pack(self, latent: dict) -> dict:
        """The wrapped def's packed form."""
        return self.inner.pack(latent)

    def raw(self, encoded: Encoded) -> dict:
        """The wrapped def's continuous view."""
        return self.inner.raw(encoded)

    def eval_variants(self) -> list:
        """The wrapped def's."""
        return self.inner.eval_variants()

    def variant(self, latent: dict, key) -> dict:
        """The wrapped def's."""
        return self.inner.variant(latent, key)

    def bind_kernels(self, device) -> None:
        """Forward to the wrapped def."""
        self.inner.bind_kernels(device)

    def track_stats(self, on: bool) -> None:
        """Forward to the wrapped def."""
        self.inner.track_stats(on)
