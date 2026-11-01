"""Error statistics of a result against an fp64 reference, next to the eager reference."""

import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

KEYS = ("cos_deficit", "abs_mean", "abs_median", "rel_mean", "rel_median")


def stats(out: Tensor, ref: Tensor) -> dict[str, float | bool]:
    """Cosine to `ref` and the abs and rel error mean, median and max of `out`."""
    flat_out = out.detach().double().flatten()
    flat_ref = ref.detach().double().flatten()

    # Absolute and relative error (relative over nonzero reference entries)
    diff = (flat_out - flat_ref).abs()
    nonzero = flat_ref != 0
    rel = (
        diff[nonzero] / flat_ref[nonzero].abs()
        if bool(nonzero.any())
        else diff.new_zeros(1)
    )
    if flat_out.equal(flat_ref):
        cos = 1.0
    else:
        cos = cosine_similarity(flat_out, flat_ref, dim=0).item()

    row = {
        "finite": bool(torch.isfinite(flat_out).all()),
        "cos": cos,
        "cos_deficit": 1.0 - cos,
    }
    for name, values in (("abs", diff), ("rel", rel)):
        row[f"{name}_mean"] = values.mean().item()
        row[f"{name}_median"] = values.median().item()
        row[f"{name}_max"] = values.max().item()
    return row


def floors(out: Tensor, ref64: Tensor) -> dict[str, float]:
    """The rounding scale of `out`'s dtype per statistic of `KEYS`."""
    eps = torch.finfo(out.dtype).eps
    scale = ref64.detach().double().abs().flatten().median().item()
    return {
        "cos_deficit": eps,
        "abs_mean": eps * scale,
        "abs_median": eps * scale,
        "rel_mean": eps,
        "rel_median": eps,
    }


def compare(out: Tensor, eager: Tensor, ref64: Tensor) -> dict[str, dict]:
    """The stats table: `kernel`, `eager` and the rounding `floor` against `ref64`."""
    return {
        "kernel": stats(out, ref64),
        "eager": stats(eager, ref64),
        "floor": floors(out, ref64),
    }


def worse(table: dict[str, dict], margin: float = 4.0) -> list[str]:
    """Statistics where the kernel passes `margin` x the larger of eager and the floor."""
    kernel, eager, floor = table["kernel"], table["eager"], table["floor"]
    bad = [key for key in KEYS if kernel[key] > margin * max(eager[key], floor[key])]
    return ["non-finite", *bad] if eager["finite"] and not kernel["finite"] else bad
