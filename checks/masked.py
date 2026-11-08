"""Masked encoding: a masked image walks the LatentMaid encoder as a sparse conv over the kept
cells, the MAE decoder and loss read only what they should, and the `mae` term trains in a run
without taking the model's parameters.

Run after changing `masked_features`, a block's `masked`, `keep_at`, the `mlp` block's mixer,
`latent/mae.py` or the term hooks `bind` and `post_step`.

    python checks/masked.py
"""

import json

import torch
from _tiny import knobs, run_checks, train, write_images
from safetensors.torch import load_file

from latentdragon.latent import LatentCtx
from latentdragon.latent.mae import (
    MaeDecoder,
    MaskedImageTerm,
    masked_mse,
    patch_targets,
    sample_keep,
)
from latentdragon.model import build_autoencoder
from latentdragon.model.families.latentmaid.encoders import LatentMaidEncoder
from latentdragon.model.nn.blocks import BasicBlock
from latentdragon.model.nn.layers import RMSNorm

# Three shuffle stages to f16, per-position norm, every block a plain `mlp`
STAGES = [[4, 16, 32, 2], [2, 16, 32, 1], [2, 32, 64, 2]]
SIZE = 64


# A whole f16 autoencoder at 32 px, a 2x2 cell grid; the encoder output is its last stage
F16_ARCH = {
    "encoder": {
        "name": "latentmaid",
        "out_channels": None,
        "stages": [[4, 16, 32, 1], [4, 16, 32, 1]],
        "norm": "layer",
    },
    "latent": {"name": "bottleneck", "in_channels": 16, "dim": 4, "out_channels": 16},
    "decoder": {
        "name": "latentmaid",
        "in_channels": 16,
        "stages": [[1, 8, 16, 1], [4, 16, 32, 1], [4, 16, 32, 1]],
        "norm": "layer",
    },
    "head": {"name": "conv", "in_channels": 8, "kernel_size": 1},
}
TERM = {"ratio": 0.5, "unit": 1, "decoder": {"dim": 8, "mlp": 16, "blocks": 1}}

# Widths group norm's 32 groups divide
WIDE_STAGES = [[4, 32, 32, 1], [2, 32, 32, 1], [2, 64, 64, 1]]


def encoder(norm: str = "layer", attn=None, stages=STAGES) -> LatentMaidEncoder:
    """A small f16 LatentMaid encoder in fp64."""
    torch.manual_seed(0)
    return LatentMaidEncoder(8, stages, norm=norm, attn=attn).double().eval()


def image_and_keep(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Random fp64 images and a 0.5 cell mask on their f16 grid."""
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 3, SIZE, SIZE, dtype=torch.float64, generator=generator)
    keep = sample_keep(batch, (SIZE // 16, SIZE // 16), 0.5, generator=generator)
    return x, keep.double()


def full_keep_is_forward(folder) -> None:
    """Keeping every cell gives `features` bit for bit."""
    model = encoder()
    x, keep = image_and_keep()
    assert torch.equal(
        model.masked_features(x, torch.ones_like(keep)), model.features(x)
    )


def masked_pixels_never_reach(folder) -> None:
    """New values in masked pixels leave every visible feature bitwise the same."""
    model = encoder()
    x, keep = image_and_keep()
    hidden = 1 - keep.repeat_interleave(16, -2).repeat_interleave(16, -1)
    other = x + hidden * torch.randn_like(x) * 100
    first, second = model.masked_features(x, keep), model.masked_features(other, keep)
    assert torch.equal(first, second)
    assert torch.equal(
        first * (1 - keep), torch.zeros_like(first)
    ), "masked cells not zero"


def masked_pixels_get_no_gradient(folder) -> None:
    """The gradient at every masked input pixel is exactly zero, and nonzero elsewhere."""
    model = encoder()
    x, keep = image_and_keep()
    x.requires_grad_(True)
    model.masked_features(x, keep).square().sum().backward()
    pixel_keep = keep.repeat_interleave(16, -2).repeat_interleave(16, -1).bool()
    assert torch.equal(
        x.grad[~pixel_keep.expand_as(x)],
        torch.zeros_like(x.grad[~pixel_keep.expand_as(x)]),
    )
    assert x.grad[pixel_keep.expand_as(x)].abs().amax() > 0


def visible_neighbours_are_read(folder) -> None:
    """A visible cell still reads its visible neighbours: unmasking one changes it."""
    model = encoder()
    x, _ = image_and_keep(1)
    grid = SIZE // 16
    one = torch.zeros(1, 1, grid, grid, dtype=torch.float64)
    one[..., 1, 1] = 1
    two = one.clone()
    two[..., 1, 2] = 1
    lone, paired = model.masked_features(x, one), model.masked_features(x, two)
    assert not torch.equal(
        lone[..., 1, 1], paired[..., 1, 1]
    ), "neighbours are not read"


def unsafe_encoders_refuse(folder) -> None:
    """Group norm statistics and attention both span cells, so both refuse."""
    x, keep = image_and_keep()
    for model in (encoder(norm="group32", stages=WIDE_STAGES), encoder(attn=[0, 0, 1])):
        try:
            model.masked_features(x, keep)
        except ValueError:
            continue
        raise AssertionError(f"{model} encoded a masked image")


def mixer_kernel_size(folder) -> None:
    """The default stays a 3x3 mixer with reach 1; a 7x7 one reaches 3 and keeps the shape."""
    torch.manual_seed(0)
    small, large = BasicBlock(8, 16, 8, RMSNorm), BasicBlock(
        8, 16, 8, RMSNorm, kernel_size=7
    )
    assert small.mixer.kernel_size == (3, 3) and small.spatial_reach == 1
    assert large.mixer.kernel_size == (7, 7) and large.spatial_reach == 3
    x = torch.randn(1, 8, 9, 9)
    assert large(x).shape == x.shape


def mask_sampling(folder) -> None:
    """Exact masked count per sample; a unit masks whole squares of cells."""
    keep = sample_keep(4, (16, 16), 0.75, unit=2)
    assert torch.equal(keep.sum((1, 2, 3)), torch.full((4,), 64.0))
    squares = keep.view(4, 1, 8, 2, 8, 2)
    assert torch.equal(squares.amax((3, 5)), squares.amin((3, 5))), "a square is split"


def loss_reads_masked_cells_only(folder) -> None:
    """Predictions and targets of visible cells never move the loss; the mask token trains."""
    torch.manual_seed(0)
    decoder = MaeDecoder(32, 16, dim=16, mlp=32, blocks=2).double()
    model = encoder()
    x, keep = image_and_keep()
    features = model.masked_features(x, keep)
    pred = decoder(features, keep)
    target = patch_targets(x, 16).double()
    assert pred.shape == target.shape == (2, 768, SIZE // 16, SIZE // 16)

    loss = masked_mse(pred, target, keep)
    noise = torch.randn_like(pred) * keep
    assert torch.equal(masked_mse(pred + noise, target + noise, keep), loss)
    loss.backward()
    assert decoder.mask_token.grad.abs().amax() > 0, "mask token gets no gradient"


def term_keeps_model_parameters(folder) -> None:
    """The bound term owns only its decoder; its loss reaches the encoder and the decoder."""
    torch.manual_seed(0)
    model = build_autoencoder(F16_ARCH)
    term = MaskedImageTerm(**TERM)
    term.bind(model)
    assert all(
        key.startswith("decoder.") for key in term.state_dict()
    ), "owns a model part"
    model_ids = {id(p) for p in model.parameters()}
    assert not any(id(p) in model_ids for p in term.parameters())

    loss, logs = term(LatentCtx({"x": torch.randn(2, 3, 32, 32)}, None, 0))
    assert set(logs) == {"train/mae/loss"}
    loss.backward()
    for name, part in (("encoder", model.encoder), ("mae decoder", term.decoder)):
        assert any(
            p.grad is not None and p.grad.abs().amax() > 0 for p in part.parameters()
        ), name
    assert (
        model.decoder.stages[0][0].mixer.weight.grad is None
    ), "the AE decoder is reached"


def term_trains_in_a_run(folder) -> None:
    """A run logs the MAE loss, calls `post_step` every step and checkpoints the decoder."""
    calls = []
    original = MaskedImageTerm.post_step
    MaskedImageTerm.post_step = lambda self, step, sync: calls.append(step)
    try:
        run_dir = train(
            knobs(
                folder,
                ARCH=F16_ARCH,
                SIZE=32,
                TRAIN_SOURCES=[
                    {
                        "name": "image_folder",
                        "root": write_images(folder / "d32", size=32),
                    }
                ],
                LATENT_TERMS=[{"term": {"name": "mae", **TERM}}],
            )
        )
    finally:
        MaskedImageTerm.post_step = original
    assert len(calls) == 2, calls

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    assert any("train/mae/loss" in row for row in rows)
    saved = load_file(str(next(run_dir.rglob("trainable.safetensors"))))
    assert "latent_terms.0.decoder.mask_token" in saved
    assert not any("mask_token" in key for key in saved if key.startswith("model."))


if __name__ == "__main__":
    run_checks(
        full_keep_is_forward,
        masked_pixels_never_reach,
        masked_pixels_get_no_gradient,
        visible_neighbours_are_read,
        unsafe_encoders_refuse,
        mixer_kernel_size,
        mask_sampling,
        loss_reads_masked_cells_only,
        term_keeps_model_parameters,
        term_trains_in_a_run,
    )
