"""Question: do a capped pre-norm and weight-normalized projections bound the residual stream and f's hidden states, where a standard pre-norm transformer drifts or diverges?

CPU only. A byte-level causal language model trains on the Python standard library's source at
several learning rates, once per variant. Variants add one change at a time:

- `baseline`: RMSNorm with gain, plain linears, GPT-2 init (writers scaled by 1/sqrt(2 depth)).
- `cap`: every norm reads `CAP * tanh(x / CAP)`, so a stream past `CAP` loses information.
- `wn_out`: writers (attention out, MLP down) have unit rows and a per-channel gain.
- `ws_out`: writers have zero-mean unit rows and a per-channel gain.
- `cap_wn_out`: `cap` and `wn_out` together.
- `cap_wn_all`: `cap` and every matrix with unit rows; norms carry no gain, the embedding has
  unit-RMS rows, writers and the head keep a gain.
- `convention`: `cap_wn_all` with QK-norm (RMSNorm on q and k, with a gain).
- `wn_all`, `wn_all_qk`: `cap_wn_all` and `convention` without the cap, to tell the cap's share
  from the unit rows'.
- `wn_out_anchor`: `wn_out` with the unit-RMS embedding, to tell the fixed-scale input's share
  from the unit-row readers'.
- `muon_baseline`, `muon_wn_all_qk`: `baseline` and `wn_all_qk` trained with Muon on the block
  matrices (qkv, out, gate/up, down) and Adam on the rest. Muon's update is scaled by
  0.2 * sqrt(max(fan_out, fan_in)) to match Adam's update RMS, so both share the lr grid.

- `proj_all_qk`: `wn_all_qk` with unit rows kept by the optimizer instead: weights start with
  unit rows (the embedding with unit-RMS rows), the forward uses them as they are, gradients and
  Adam's first moment are projected onto each row's tangent plane before the step, and rows are
  put back on the sphere after it.
- `both_all_qk`: `wn_all_qk` (normalize at use) plus the same retraction after each step, so the
  raw rows stay at unit norm and the effective learning rate does not decay.

A variant may carry its own `lrs`; others use `LRS`. Runs already in `OUT` with the same variant
spec and knobs are kept, and only missing ones run.

Logged every `LOG_EVERY` steps on a held-out batch: loss, the stream's RMS and max after every
block, the MLP gate and up pre-activations, q, k and v, the attention logits, and the raw weight
norms of the readers and writers.

    python experiments/prenorm/toy_drift.py
"""

import json
import math
import os
import sysconfig
import time
from multiprocessing import get_context
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS_BYTES: int = 8_000_000
CONTEXT: int = 64
BATCH: int = 32
DIM: int = 128
DEPTH: int = 6
HEADS: int = 4
HIDDEN: int = 384
STEPS: int = 3000
WARMUP: int = 100
LOG_EVERY: int = 50
LRS: tuple = (3e-3, 1e-2, 3e-2)
WIDE_LRS: tuple = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
CAP: float = 8.0
VARIANTS: dict[str, dict] = {
    "baseline": {"cap": None, "writer": "plain", "reader": "plain"},
    "cap": {"cap": CAP, "writer": "plain", "reader": "plain"},
    "wn_out": {"cap": None, "writer": "wn", "reader": "plain"},
    "ws_out": {"cap": None, "writer": "ws", "reader": "plain"},
    "cap_wn_out": {"cap": CAP, "writer": "wn", "reader": "plain"},
    "cap_wn_all": {"cap": CAP, "writer": "wn", "reader": "wn"},
    "convention": {"cap": CAP, "writer": "wn", "reader": "wn", "qk_norm": True},
    "wn_all": {"cap": None, "writer": "wn", "reader": "wn"},
    "wn_all_qk": {"cap": None, "writer": "wn", "reader": "wn", "qk_norm": True},
    "wn_out_anchor": {"cap": None, "writer": "wn", "reader": "plain", "anchor": True},
    "muon_baseline": {
        "cap": None,
        "writer": "plain",
        "reader": "plain",
        "optimizer": "muon",
    },
    "muon_wn_all_qk": {
        "cap": None,
        "writer": "wn",
        "reader": "wn",
        "qk_norm": True,
        "optimizer": "muon",
    },
    "proj_all_qk": {
        "cap": None,
        "writer": "proj",
        "reader": "proj",
        "qk_norm": True,
        "lrs": WIDE_LRS,
    },
    "both_all_qk": {
        "cap": None,
        "writer": "both",
        "reader": "both",
        "qk_norm": True,
        "lrs": WIDE_LRS,
    },
}
# The baseline and normalize-at-use runs also get the low learning rates, for a fair best-lr pick
for _name in ("baseline", "wn_all_qk"):
    VARIANTS[_name]["lrs"] = WIDE_LRS
THREADS: int = 6
WORKERS: int = 7
SEED: int = 0
OUT: Path = Path("runs/prenorm/toy_drift.json")


def corpus() -> torch.Tensor:
    """The first `CORPUS_BYTES` bytes of the standard library's .py files, sorted by path."""
    root = Path(sysconfig.get_paths()["stdlib"])
    chunks, total = [], 0
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        chunks.append(data)
        total += len(data)
        if total >= CORPUS_BYTES:
            break
    return torch.frombuffer(
        bytearray(b"".join(chunks)[:CORPUS_BYTES]), dtype=torch.uint8
    )


class Norm(nn.Module):
    """RMSNorm, optionally reading `cap * tanh(x / cap)` and optionally with a gain."""

    def __init__(self, dim: int, cap: float | None, gain: bool):
        super().__init__()
        self.cap = cap
        self.gain = nn.Parameter(torch.ones(dim)) if gain else None

    def forward(self, x):
        if self.cap is not None:
            x = self.cap * torch.tanh(x / self.cap)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return x if self.gain is None else x * self.gain


def unit_rows(weight: torch.Tensor, norm: float = 1.0) -> torch.Tensor:
    """`weight` with each row (dim 0) scaled to L2 norm `norm`."""
    return weight * (norm / weight.norm(dim=1, keepdim=True).clamp_min(1e-6))


class Linear(nn.Module):
    """A bias-free linear with an optional per-channel gain. `plain`; unit rows normalized at use
    (`wn`, and `both`, whose rows the optimizer also keeps at unit norm); zero-mean unit rows
    (`ws`); or unit rows kept only by the optimizer (`proj`), used as they are in the forward.
    Unit rows give a unit-variance output for a unit-variance input.
    """

    def __init__(self, fan_in: int, fan_out: int, mode: str, scale: float = 1.0):
        super().__init__()
        self.mode = mode
        self.weight = nn.Parameter(torch.randn(fan_out, fan_in) / math.sqrt(fan_in))
        if mode in ("proj", "both"):
            self.weight.data = unit_rows(self.weight.data)
        if mode == "plain":
            self.weight.data.mul_(scale)
            self.gain = None
        else:
            self.gain = nn.Parameter(torch.full((fan_out,), scale))

    def effective(self):
        weight = self.weight
        if self.mode == "ws":
            weight = weight - weight.mean(1, keepdim=True)
        if self.mode not in ("plain", "proj"):
            weight = unit_rows(weight)
        return weight

    def forward(self, x):
        out = F.linear(x, self.effective())
        return out if self.gain is None else out * self.gain


class Block(nn.Module):
    """Pre-norm attention and SwiGLU MLP, each added to the stream."""

    def __init__(self, spec: dict):
        super().__init__()
        gain = spec["reader"] == "plain"
        writer_scale = 1 / math.sqrt(2 * DEPTH)
        self.norm1 = Norm(DIM, spec["cap"], gain)
        self.qkv = Linear(DIM, 3 * DIM, spec["reader"])
        self.out = Linear(DIM, DIM, spec["writer"], writer_scale)
        self.qk_norm = (
            nn.ModuleList(Norm(DIM // HEADS, None, True) for _ in range(2))
            if spec.get("qk_norm")
            else None
        )
        self.norm2 = Norm(DIM, spec["cap"], gain)
        self.gate_up = Linear(DIM, 2 * HIDDEN, spec["reader"])
        self.down = Linear(HIDDEN, DIM, spec["writer"], writer_scale)
        self.stats = None

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, HEADS, DIM // HEADS)
            .permute(2, 0, 3, 1, 4)
        )
        if self.qk_norm is not None:
            q, k = self.qk_norm[0](q), self.qk_norm[1](k)
        mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(mixed.transpose(1, 2).reshape(batch, length, DIM))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        x = x + self.down(F.silu(gate) * up)
        if self.stats is not None:
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(DIM // HEADS)
            mask = torch.ones(length, length, dtype=torch.bool).tril()
            self.stats.update(
                stream_rms=x.square().mean().sqrt().item(),
                stream_max=x.abs().max().item(),
                gate_max=gate.abs().max().item(),
                up_max=up.abs().max().item(),
                up_rms=up.square().mean().sqrt().item(),
                q_rms=q.square().mean().sqrt().item(),
                v_rms=v.square().mean().sqrt().item(),
                logit_max=logits.masked_select(mask).abs().max().item(),
            )
        return x


class Model(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = spec
        self.embed = nn.Parameter(torch.randn(256, DIM))
        if spec["reader"] in ("proj", "both"):
            self.embed.data = unit_rows(self.embed.data, math.sqrt(DIM))
        self.pos = nn.Parameter(torch.randn(CONTEXT, DIM) * 0.1)
        self.blocks = nn.ModuleList(Block(spec) for _ in range(DEPTH))
        self.norm = Norm(DIM, spec["cap"], spec["reader"] == "plain")
        self.head = Linear(DIM, 256, spec["reader"])

    def forward(self, tokens):
        embed = self.embed
        if self.spec.get("anchor", self.spec["reader"] not in ("plain", "proj")):
            embed = embed * torch.rsqrt(embed.square().mean(1, keepdim=True) + 1e-6)
        x = embed[tokens.long()] + self.pos[: tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def batch_of(data: torch.Tensor, generator: torch.Generator):
    starts = torch.randint(len(data) - CONTEXT - 1, (BATCH,), generator=generator)
    window = torch.stack([data[s : s + CONTEXT + 1] for s in starts.tolist()]).long()
    return window[:, :-1], window[:, 1:]


def snapshot(model: Model, val) -> dict:
    """Held-out loss, per-block activation stats and raw weight norms."""
    for block in model.blocks:
        block.stats = {}
    with torch.no_grad():
        logits = model(val[0])
        loss = F.cross_entropy(logits.flatten(0, 1), val[1].flatten()).item()
    blocks = [block.stats for block in model.blocks]
    for block in model.blocks:
        block.stats = None

    def norms(name):
        return [getattr(b, name).weight.norm().item() for b in model.blocks]

    def gains(name):
        return [
            getattr(b, name).gain.abs().mean().item()
            for b in model.blocks
            if getattr(b, name).gain is not None
        ]

    row = {"val_loss": loss, "logits_max": logits.abs().max().item()}
    for key in blocks[0]:
        row[key] = max(stats[key] for stats in blocks)
    row["stream_rms_last"] = blocks[-1]["stream_rms"]
    for name in ("qkv", "gate_up", "out", "down"):
        row[f"{name}_norm"] = max(norms(name))
    for name in ("out", "down"):
        if gains(name):
            row[f"{name}_gain"] = max(gains(name))
    return row


def orthogonalize(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon's quintic Newton-Schulz iteration: `grad` with its singular values pushed near 1."""
    x = grad / (grad.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * gram @ gram) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, orthogonalized, scaled to Adam's update RMS (Moonlight's rule)."""

    def __init__(self, params, lr: float, momentum: float = 0.95):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                buffer = self.state[param].setdefault(
                    "momentum", torch.zeros_like(param)
                )
                buffer.mul_(group["momentum"]).add_(param.grad)
                update = param.grad.add(buffer, alpha=group["momentum"])
                scale = 0.2 * math.sqrt(max(param.shape))
                param.add_(orthogonalize(update), alpha=-group["lr"] * scale)


def optimizers(model: Model, lr: float) -> list[torch.optim.Optimizer]:
    """Adam on everything, or Muon on the block matrices and Adam on the rest."""
    if model.spec.get("optimizer", "adam") != "muon":
        return [torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))]
    matrices = [
        getattr(block, name).weight
        for block in model.blocks
        for name in ("qkv", "out", "gate_up", "down")
    ]
    chosen = {id(param) for param in matrices}
    rest = [param for param in model.parameters() if id(param) not in chosen]
    return [
        Muon(matrices, lr=lr),
        torch.optim.Adam(rest, lr=lr, betas=(0.9, 0.95)),
    ]


def sphere_params(model: Model) -> list[tuple[torch.Tensor, float]]:
    """Weights the optimizer keeps on the sphere, with their row norm: unit for matrices,
    sqrt(DIM) (unit RMS) for the embedding."""
    if model.spec["reader"] not in ("proj", "both"):
        return []
    rows = [(model.embed, math.sqrt(DIM))]
    rows += [
        (module.weight, 1.0)
        for module in model.modules()
        if isinstance(module, Linear) and module.mode in ("proj", "both")
    ]
    return rows


def tangent(vector: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`vector` minus its component along each row of `weight`."""
    direction = unit_rows(weight)
    return vector - (vector * direction).sum(1, keepdim=True) * direction


@torch.no_grad()
def project_grads(sphere: list) -> None:
    """Before the step: gradients onto each row's tangent plane (`proj` only; under `both` the
    normalize-at-use forward already gives tangent gradients)."""
    for param, _ in sphere:
        if param.grad is not None:
            param.grad.copy_(tangent(param.grad, param))


@torch.no_grad()
def retract(sphere: list, optimizer: torch.optim.Optimizer) -> None:
    """After the step: rows back to their norm, Adam's first moment back onto the tangent plane."""
    for param, norm in sphere:
        param.copy_(unit_rows(param, norm))
        moment = optimizer.state.get(param, {}).get("exp_avg")
        if moment is not None:
            moment.copy_(tangent(moment, param))


def run(job: tuple[str, float]) -> tuple[str, float, dict]:
    name, lr = job
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    data = corpus()
    split = int(len(data) * 0.95)
    train, held = data[:split], data[split:]
    generator = torch.Generator().manual_seed(SEED)
    val = batch_of(held, torch.Generator().manual_seed(SEED + 1))
    model = Model(VARIANTS[name])
    steppers = optimizers(model, lr)
    sphere = sphere_params(model)
    log, started, diverged = [], time.time(), None
    for step in range(STEPS + 1):
        if step % LOG_EVERY == 0:
            row = snapshot(model, val)
            row["step"] = step
            log.append(row)
            if not math.isfinite(row["val_loss"]) or row["val_loss"] > 20:
                diverged = step
                break
        if step == STEPS:
            break
        for optimizer in steppers:
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (step + 1) / WARMUP)
        tokens, targets = batch_of(train, generator)
        loss = F.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())
        model.zero_grad(set_to_none=True)
        loss.backward()
        if model.spec["reader"] == "proj":
            project_grads(sphere)
        for optimizer in steppers:
            optimizer.step()
        if sphere:
            retract(sphere, steppers[0])
    print(
        f"{name:11s} lr {lr:.0e}  val {log[-1]['val_loss']:.3f}  "
        f"stream_max {log[-1]['stream_max']:.1f}  up_max {log[-1]['up_max']:.1f}  "
        f"logit_max {log[-1]['logit_max']:.1f}  diverged {diverged}  "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    return name, lr, {"log": log, "diverged": diverged}


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    knobs = {
        "context": CONTEXT,
        "batch": BATCH,
        "dim": DIM,
        "depth": DEPTH,
        "heads": HEADS,
        "hidden": HIDDEN,
        "steps": STEPS,
        "warmup": WARMUP,
        "cap": CAP,
        "lrs": list(LRS),
    }

    def lrs_of(name: str) -> tuple:
        return VARIANTS[name].get("lrs", LRS)

    def spec(variants: dict, name: str) -> dict | None:
        """A variant's spec without its lr list, so adding learning rates keeps earlier runs."""
        found = variants.get(name)
        return None if found is None else {k: v for k, v in found.items() if k != "lrs"}

    # Keep earlier runs whose knobs and variant spec are unchanged
    kept = {}
    if OUT.exists():
        earlier = json.loads(OUT.read_text())
        if earlier["knobs"] == knobs:
            kept = {
                key: result
                for key, result in earlier["runs"].items()
                if spec(earlier["variants"], key.split("@")[0])
                == spec(VARIANTS, key.split("@")[0])
            }
    jobs = [
        (name, lr)
        for name in VARIANTS
        for lr in lrs_of(name)
        if f"{name}@{lr:g}" not in kept
    ]
    with get_context("spawn").Pool(WORKERS) as pool:
        results = pool.map(run, jobs, chunksize=1)
    runs = kept | {f"{name}@{lr:g}": result for name, lr, result in results}
    out = {
        "knobs": knobs,
        "variants": VARIANTS,
        "runs": {
            f"{name}@{lr:g}": runs[f"{name}@{lr:g}"]
            for name in VARIANTS
            for lr in lrs_of(name)
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
