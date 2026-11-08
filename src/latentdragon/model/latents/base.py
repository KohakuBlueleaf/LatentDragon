"""Single-slot latent defs and the identity def."""

from typing import Any

from torch import Tensor, nn

from latentdragon.model.core.contracts import LATENT, Encoded, LatentDef


class TensorLatent(LatentDef):
    """A latent def over one encoder slot and one latent slot of the same name downstream.

    Subclasses implement `encode(raw, offset) -> Encoded` on the tensor and, when the latent
    slot holds more than the decoder input, `read(value) -> Tensor`. `input_factor` is the
    encoder channels read per latent channel.
    """

    input_factor = 1

    def __init__(self, input: str = "h", output: str = "z"):
        super().__init__()
        self.input = input
        self.output = output
        self.inputs = (input,)
        self.outputs = (output,)
        self.decoder_slots = (output,)

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        raise NotImplementedError

    def read(self, value: Any) -> Tensor:
        """The decoder input held by a latent slot value; the value itself by default."""
        return value

    def init_from_encoder(self, layer: nn.Module) -> None:
        """Adjust the encoder's last layer for this def; nothing by default."""

    def serialize(self, slots: dict, offset: int = 0) -> Encoded:
        """Encode the input slot into the output slot."""
        return self.encode(slots[self.input], offset)

    def deserialize(self, latent: dict) -> dict:
        """The output slot's value as the decoder input."""
        return {self.output: self.read(latent[self.output])}

    def raw(self, encoded: Encoded) -> dict:
        """The value before sampling or quantization, when the def kept one."""
        return {self.output: encoded.extras.get("raw", encoded.latent[self.output])}


@LATENT.register("identity")
class IdentityLatent(TensorLatent):
    """The encoder output, untouched."""

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        """Pass `raw` through as the latent."""
        return Encoded(latent={self.output: raw})
