"""Masked image modeling on the encoder output, FCMAE style, as the `mae` latent term.

The encoder walks the masked image as a sparse conv (`masked_tensor`) up to its output, the
pre-latent state the latent def reads; `MaeDecoder` fills the masked cells with a learned token,
mixes with isotropic LatentMaid blocks on that grid and predicts one pixel patch per cell;
`masked_mse` scores the masked cells against per-cell normalized pixels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from optimfactory import mup_init
from torch import Tensor

from latentdragon.latent.context import LatentCtx
from latentdragon.latent.terms import LATENT_TERM, LatentTerm
from latentdragon.model.nn.blocks import init_writers
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.stages import block_stack
from latentdragon.model.runtime.compile import SETUP


def sample_keep(
    batch: int,
    grid: tuple[int, int],
    ratio: float,
    unit: int = 1,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """`(B, 1, h, w)` cell mask, 1 for visible: the same masked count in every sample.

    Masking picks whole `unit x unit` squares of cells, `round(ratio * squares)` per sample.
    """
    rows, cols = grid
    if rows % unit or cols % unit:
        raise ValueError(
            f"mask unit {unit} does not divide the {rows}x{cols} cell grid"
        )
    squares = (rows // unit) * (cols // unit)
    masked = round(squares * ratio)
    if not 0 < masked < squares:
        raise ValueError(f"mask ratio {ratio} masks {masked} of {squares} squares")

    # The first `masked` squares of a random order per sample are hidden
    order = torch.rand(batch, squares, device=device, generator=generator).argsort(1)
    keep = torch.ones(batch, squares, device=device)
    keep.scatter_(1, order[:, :masked], 0.0)
    keep = keep.view(batch, 1, rows // unit, cols // unit)
    return keep.repeat_interleave(unit, -2).repeat_interleave(unit, -1)


class MaeDecoder(nn.Module):
    """Mask token, isotropic LatentMaid blocks on the feature grid, one patch per cell.

    `patch` is the pixel size of one feature cell (the encoder's spatial scale); the output is
    `(B, out_channels * patch**2, h, w)` in `pixel_unshuffle` order. The prediction head starts
    at zero.
    """

    def __init__(
        self,
        in_channels: int,
        patch: int,
        dim: int = 512,
        mlp: int = 1024,
        blocks: int = 4,
        kernel_size: int = 7,
        norm: str = "layer",
        out_channels: int = 3,
        writer_init: str = "hidden",
    ):
        super().__init__()
        norm_layer = resolve_norm(norm)
        self.patch = patch
        self.proj = nn.Conv2d(in_channels, dim, 1)
        block = {"name": "mlp", "kernel_size": kernel_size}
        self.blocks = block_stack(block, blocks, dim, mlp, norm_layer)
        self.norm = norm_layer(dim)
        self.pred = nn.Conv2d(dim, out_channels * patch**2, 1)
        mup_init(self.parameters())
        init_writers(self.blocks, writer_init)
        # Zero head: the decoder starts at every cell's mean, and the encoder sees no MAE
        # gradient until the head grows, so the term enters the shared backward gently
        nn.init.zeros_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)
        self.mask_token = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, features: Tensor, keep: Tensor) -> Tensor:
        """Patch predictions for every cell from the masked features."""
        x = self.proj(features)
        x = torch.where(keep.bool(), x, self.mask_token.to(x.dtype))
        for block in self.blocks:
            x = block(x)
        return self.pred(self.norm(x))


def patch_targets(pixels: Tensor, patch: int, eps: float = 1e-6) -> Tensor:
    """Pixels per cell in `pixel_unshuffle` order, each cell at zero mean and unit variance, fp32."""
    cells = F.pixel_unshuffle(pixels.float(), patch)
    mean = cells.mean(1, keepdim=True)
    var = cells.var(1, keepdim=True)
    return (cells - mean) / (var + eps).sqrt()


def masked_mse(pred: Tensor, target: Tensor, keep: Tensor) -> Tensor:
    """Mean squared error over the masked cells only, in fp32."""
    per_cell = (pred.float() - target).square().mean(1, keepdim=True)
    hidden = 1.0 - keep
    return (per_cell * hidden).sum() / hidden.sum()


@LATENT_TERM.register("mae")
class MaskedImageTerm(LatentTerm):
    """Masked image modeling on the encoder output: the masked cells' pixels from the visible.

    Each call masks `ratio` of the encoder's output cells in `unit x unit` squares, walks the
    masked image through the encoder as a sparse conv, and scores the MAE decoder's patches on
    the masked cells. `decoder` holds `MaeDecoder` keywords; its input width and patch size come
    from the encoder at bind. The decoder trains with the model and is never exported.
    """

    requires = ("encoder",)

    def __init__(self, ratio: float = 0.6, unit: int = 2, decoder: dict | None = None):
        super().__init__()
        self.ratio = ratio
        self.unit = unit
        self.decoder_spec = dict(decoder or {})
        self.decoder: MaeDecoder | None = None
        self._run = self._masked_patches

    def bind(self, model: torch.nn.Module) -> None:
        """Keep the encoder, size the decoder from its output, compile the masked pass."""
        encoder = model.encoder
        if not hasattr(encoder, "masked_tensor"):
            raise ValueError(f"{type(encoder).__name__} cannot encode masked images")
        # A plain attribute, not a submodule: the encoder's parameters stay the model's
        object.__setattr__(self, "encoder", encoder)
        self.patch = encoder.spatial_scale
        width = encoder.out_channels
        self.decoder = MaeDecoder(width, self.patch, **self.decoder_spec)
        if SETUP.scope == "model":
            self._run = torch.compile(self._masked_patches, **SETUP.kwargs())

    def _masked_patches(self, x: Tensor, keep: Tensor) -> Tensor:
        """Patch predictions for every cell of the masked image."""
        return self.decoder(self.encoder.masked_tensor(x, keep), keep)

    def forward(self, ctx: LatentCtx) -> tuple:
        x = ctx.x[self.encoder.input]
        grid = (x.shape[-2] // self.patch, x.shape[-1] // self.patch)
        keep = sample_keep(x.shape[0], grid, self.ratio, self.unit, x.device)
        loss = masked_mse(self._run(x, keep), patch_targets(x, self.patch), keep)
        return loss, {"train/mae/loss": loss.detach()}
