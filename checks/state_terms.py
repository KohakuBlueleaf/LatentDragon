"""State terms: the RMS term pulls the named state, the fp32 head stays fp32, both grids land.

Run after changing latent terms, the task's states, the conv head or the sample callback.

    python checks/state_terms.py
"""

import json

import torch
from _tiny import TINY_ARCH, knobs, run_checks, train

from latentdragon.latent import LatentCtx, StateRms
from latentdragon.model.heads.conv import ConvHead
from latentdragon.training.precision import cast_weights

# The pre-latent and output-latent RMS terms the production recipe uses
TERMS = [
    {
        "term": {"name": "state_rms", "at": "encoder", "groups": 2},
        "weight": {"name": "fixed", "weight": 0.01},
    },
    {
        "term": {"name": "state_rms", "at": "decoder"},
        "weight": {"name": "fixed", "weight": 0.01},
    },
]


def rms_matches_by_hand(folder) -> None:
    """Grouped per-position RMS and its loss, against a direct computation."""
    torch.manual_seed(0)
    value = (torch.randn(2, 8, 3, 3) * 3).requires_grad_()
    term = StateRms(at="decoder", groups=2)
    ctx = LatentCtx({}, None, 0, states={"decoder": {"y": value}})
    loss, logs = term(ctx)

    groups = value.detach().unflatten(1, (2, 4))
    rms = groups.square().mean(2).add(1e-6).sqrt()
    expected = (rms - 1).square().mean()
    assert torch.allclose(loss, expected), (loss, expected)
    assert set(logs) == {
        "train/state_rms/decoder/y_loss",
        "train/state_rms/decoder/y_rms",
    }

    # Scaled up the gradient shrinks the state; a unit-RMS state is a minimum
    loss.backward()
    assert (value.grad * value).sum() > 0
    unit = value.detach() / rms.repeat_interleave(4, 1)
    assert term(LatentCtx({}, None, 0, states={"decoder": {"y": unit}}))[0] < 1e-8


def fp32_head_stays_fp32(folder) -> None:
    """The head's weights skip a 16-bit cast; its output is fp32 under autocast."""
    head = ConvHead(8, 3, fp32=True)
    plain = ConvHead(8, 3)
    cast_weights(head, torch.bfloat16, ())
    cast_weights(plain, torch.bfloat16, ())
    assert head.proj.weight.dtype == torch.float32
    assert plain.proj.weight.dtype == torch.bfloat16
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head.tensor(torch.randn(1, 8, 4, 4, dtype=torch.bfloat16))
    assert out.dtype == torch.float32


def terms_train_and_grids_land(folder) -> None:
    """Both state terms train and log; each sample step writes a train and a fixed grid."""
    arch = {**TINY_ARCH, "head": {**TINY_ARCH["head"], "fp32": True}}
    run_dir = train(
        knobs(
            folder,
            ARCH=arch,
            LATENT_TERMS=TERMS,
            MAX_STEPS=4,
            BATCH_SIZE=4,
            SAMPLE_EVERY=2,
        )
    )
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    keys = set().union(*rows)
    for key in (
        "train/state_rms/encoder/h_loss",
        "train/state_rms/decoder/y_rms",
        "encoder_output/h_std",
        "output_latent/y_std",
    ):
        assert key in keys, key
    for folder_name in ("train", "fixed"):
        grids = sorted(p.name for p in (run_dir / "samples" / folder_name).iterdir())
        assert grids == ["2.webp", "4.webp"], (folder_name, grids)


if __name__ == "__main__":
    run_checks(rms_matches_by_hand, fp32_head_stays_fp32, terms_train_and_grids_land)
