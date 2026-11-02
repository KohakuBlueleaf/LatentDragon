"""Vector quantization: indices are the stored latent, the codebook reads them back."""

import torch
from torch import Tensor

from latentdragon.model.core.contracts import LATENT, Encoded
from latentdragon.model.latents.base import TensorLatent
from latentdragon.vq import QUANTIZER
from latentdragon_common.registry import build


@LATENT.register("vq")
class VQLatent(TensorLatent):
    """A channels-first slot quantized by a channels-last grid quantizer.

    The latent slot holds `{"codes": straight-through vectors, "indices": global indices}`
    while training and evaluating; `pack` keeps only the indices, and `deserialize` looks a
    packed latent up in the same codebook. `dim` is the channel count of the slot.
    """

    def __init__(
        self, dim: int, quantizer: str | dict, input: str = "h", output: str = "z"
    ):
        super().__init__(input, output)
        self.vq = build(quantizer, QUANTIZER, emb_dim=dim)

    @property
    def quantizer(self):
        """The grid quantizer."""
        return self.vq

    @property
    def codebook_size(self) -> int:
        """Codes across every partition and level."""
        return self.vq.codebook_size

    def codes(self, encoded: Encoded) -> Tensor:
        """The global indices of an encoded batch, for code usage."""
        return encoded.latent[self.output]["indices"]

    def bind_kernels(self, device: torch.device | str) -> None:
        """Forward to the quantizer."""
        self.vq.bind_kernels(device)

    def track_stats(self, on: bool) -> None:
        """Forward to the quantizer."""
        self.vq.track_stats = on

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        """Quantize `raw`; the quantizer info splits into the loss and `vq/` logs."""
        # Channels-first to channels-last for the quantizer
        out, info = self.vq(raw.movedim(1, -1))

        indices = info.pop("indices")
        vq_loss = info.pop("vq_loss")
        return Encoded(
            latent={self.output: {"codes": out.movedim(-1, 1), "indices": indices}},
            losses={"vq": vq_loss},
            logs={f"vq/{key}": value.detach() for key, value in info.items()},
            extras={"raw": raw},
        )

    def read(self, value: dict) -> Tensor:
        """The live straight-through codes, or a codebook lookup of packed indices."""
        if "codes" in value:
            return value["codes"]
        return self.vq.get_codebook_entry(value["indices"]).movedim(-1, 1)

    def pack(self, latent: dict) -> dict:
        """Only the indices of the latent slot."""
        return {self.output: {"indices": latent[self.output]["indices"]}}
