"""Unit-row blocks: the raw weight scale never reaches the output, gradients are tangent, and
`writer_init="output"` moves to the writer gain; both block types train.

Run after changing `UnitRows`, the `mlp` or `res` block, the norms' `affine` switch or
`init_writers`.

    python checks/unit_rows.py
"""

import math

import torch
from _tiny import TINY_ARCH, knobs, run_checks, train

from latentdragon.model.nn.blocks import BasicBlock, BasicResBlock, init_writers
from latentdragon.model.nn.layers import (
    RMSNorm,
    UnitRowConv2d,
    UnitRowLinear,
    UnitRows,
    unit_rows,
)


def blocks(unit: bool) -> dict:
    """One `mlp` and one channel-changing `res` block, fp32, seeded."""
    torch.manual_seed(0)
    return {
        "mlp": BasicBlock(8, 16, 8, RMSNorm, unit_rows=unit).double(),
        "res": BasicResBlock(8, 12, RMSNorm, unit_rows=unit).double(),
    }


def scale_rows(module: torch.nn.Module) -> None:
    """Multiply every row of every unit-row weight by its own random factor in [0.1, 10]."""
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, UnitRows):
                rows = layer.weight.shape[0]
                factor = 10 ** (torch.rand(rows, dtype=layer.weight.dtype) * 2 - 1)
                layer.weight.mul_(factor.view(-1, *[1] * (layer.weight.dim() - 1)))


def layers_apply_unit_rows(folder) -> None:
    """Linear, conv and depthwise conv: unit rows, the gain scales its row, the bias adds."""
    torch.manual_seed(0)
    linear = UnitRowLinear(6, 4, gain=True).double()
    conv = UnitRowConv2d(6, 4, 3, 1, 1, gain=True).double()
    depthwise = UnitRowConv2d(6, 6, 3, 1, 1, groups=6).double()
    for layer in (linear, conv, depthwise):
        norms = layer.unit_weight().flatten(1).norm(dim=1)
        if layer.gain is None:
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), norms
        else:
            with torch.no_grad():
                layer.gain.copy_(torch.arange(1.0, 5.0, dtype=torch.float64))
            norms = layer.unit_weight().flatten(1).norm(dim=1)
            assert torch.allclose(norms, layer.gain, rtol=1e-3), norms

    x = torch.randn(3, 6, dtype=torch.float64)
    expected = x @ unit_rows(linear.weight).T * linear.gain + linear.bias
    assert torch.allclose(linear(x), expected)

    # A 16-bit weight keeps unit rows to 16-bit precision
    half = unit_rows(torch.randn(64, 32).half() * 300)
    error = (half.float().norm(dim=1) - 1).abs().max().item()
    assert error < 2e-3, error


def blocks_ignore_raw_scale(folder) -> None:
    """Rescaling every raw row by up to 10x either way leaves each block's output unchanged."""
    x = torch.randn(2, 8, 6, 6, dtype=torch.float64)
    for name, block in blocks(True).items():
        before = block(x)
        scale_rows(block)
        after = block(x)
        error = ((after - before).abs().max() / before.abs().max()).item()
        assert error < 1e-6, (name, error)

    # The flag-off blocks have no gains and keep their norm affines
    for name, block in blocks(False).items():
        keys = set(block.state_dict())
        assert not any(key.endswith("gain") for key in keys), (name, keys)
        assert any(
            key.startswith("norm") and key.endswith("weight") for key in keys
        ), name


def flag_layout(folder) -> None:
    """Norms carry no affine; only the writer carries a gain; every conv and linear is unit-row."""
    for name, block in blocks(True).items():
        for module in block.modules():
            if isinstance(module, RMSNorm):
                assert module.weight is None, name
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                assert isinstance(module, UnitRows), (name, module)
        gains = [m for m in block.modules() if getattr(m, "gain", None) is not None]
        assert gains == [block.writer], (name, gains)


def gradients_are_tangent(folder) -> None:
    """Each raw row's gradient is orthogonal to the row: no step can change the function's scale."""
    x = torch.randn(2, 8, 6, 6, dtype=torch.float64)
    for name, block in blocks(True).items():
        block(x).square().mean().backward()
        for layer in block.modules():
            if isinstance(layer, UnitRows):
                rows = layer.weight.flatten(1)
                grads = layer.weight.grad.flatten(1)
                radial = (rows * grads).sum(1).abs() / (
                    rows.norm(dim=1) * grads.norm(dim=1) + 1e-30
                )
                assert radial.max().item() < 1e-8, (name, layer, radial.max().item())


def writer_init_moves_to_gain(folder) -> None:
    """`writer_init="output"` scales a unit-row writer's gain by fan_in ** -0.5, and a plain
    writer's weight as before."""
    for unit in (True, False):
        stack = torch.nn.ModuleList(blocks(unit).values())
        before = [block.writer.weight.detach().clone() for block in stack]
        init_writers(stack, "output")
        for block, weight in zip(stack, before):
            fan_in = math.prod(weight.shape[1:])
            if unit:
                expected = torch.full_like(block.writer.gain, fan_in**-0.5)
                assert torch.allclose(block.writer.gain, expected), block.writer.gain
                assert torch.equal(block.writer.weight, weight)
            else:
                assert torch.allclose(block.writer.weight, weight * fan_in**-0.5)
            assert block.writer.bias.abs().max().item() == 0


def both_blocks_train(folder) -> None:
    """A tiny SD-VAE arch on `res` blocks and a tiny LatentMaid arch on `mlp` blocks train."""
    sdvae = {
        **TINY_ARCH,
        "encoder": {
            **TINY_ARCH["encoder"],
            "block": {"name": "res", "unit_rows": True},
        },
        "decoder": {
            **TINY_ARCH["decoder"],
            "block": {"name": "res", "unit_rows": True},
        },
    }
    latentmaid = {
        **TINY_ARCH,
        "encoder": {
            "name": "latentmaid",
            "out_channels": 4,
            "stages": [[2, 8, 16, 1]],
            "block": {"name": "mlp", "unit_rows": True},
            "writer_init": "output",
        },
        "decoder": {
            "name": "latentmaid",
            "in_channels": 4,
            # 32 channels shuffle down to the tiny head's 8
            "stages": [[2, 32, 16, 1]],
            "block": {"name": "mlp", "unit_rows": True},
            "writer_init": "output",
        },
    }
    for name, arch in (("sdvae", sdvae), ("latentmaid", latentmaid)):
        run_dir = train(knobs(folder / name, ARCH=arch, MAX_STEPS=4, BATCH_SIZE=4))
        assert (run_dir / "metrics.jsonl").exists(), name


if __name__ == "__main__":
    run_checks(
        layers_apply_unit_rows,
        blocks_ignore_raw_scale,
        flag_layout,
        gradients_are_tangent,
        writer_init_moves_to_gain,
        both_blocks_train,
    )
