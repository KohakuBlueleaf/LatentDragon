"""Question: where does the first non-finite value of a skipped fp16 step come from?

Trains a recipe on one GPU through the real task, driver and bundles, compiled as configured.
Every step it checks, on the device, the model's slot boundaries, the loss terms, the gradient
seeds and every parameter gradient, and keeps the weights and batch from before the step. On a
step whose gradient norm is not finite it writes one JSON row of what went non-finite, saves the
weights and batch, and replays that step eagerly with a finite check after every module, so the
row names the first module whose output was not finite. Real compute.

    kogine run experiments/precision/nonfinite_probe.py --set DEVICE=cuda:1 --set SEED=1
"""

import json
from pathlib import Path

import torch
from kohakuengine import load_config_file

from latentdragon.domain import DOMAIN
from latentdragon.model.core import tree
from latentdragon.training.build.assemble import build_bundles, build_ema, build_task
from latentdragon.training.drivers.native import Driver
from latentdragon.training.entry import _sources
from latentdragon.training.parallel import sync
from latentdragon.training.precision import PRECISION
from latentdragon.training.state import TrainState
from latentdragon.training.timing import StepTimer
from latentdragon_common.knobs import supplied_knobs
from latentdragon_common.registry import build
from latentdragon_data import epoch_batches

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train" / "ae.py"

RECIPE: str = "ae/recipes/sdvae"  # a config under configs/, no .py
DEVICE: str = "cuda:0"
SEED: int = 0
MAX_STEPS: int = 4000
STOP_AFTER: int = 3  # stop after this many non-finite steps
NUM_WORKERS: int = 8
OUT: Path = Path("runs/precision/nonfinite.jsonl")
DUMPS: Path = Path("runs/precision/nonfinite")


def knobs_of() -> dict:
    """Script defaults under the recipe, on one device, no logging or checkpoints."""
    values = supplied_knobs(dict(load_config_file(SCRIPT).globals_dict))
    path = ROOT / "configs" / f"{RECIPE}.py"
    values.update(supplied_knobs(dict(load_config_file(path).globals_dict)))
    return {
        **values,
        "DEVICES": [DEVICE],
        "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": 0,
        "MAX_STEPS": MAX_STEPS,
    }


def finite(value) -> torch.Tensor:
    """One device flag: every tensor of `value` is finite."""
    flags = [
        torch.isfinite(t).all() for t in tree.leaves(value) if t.is_floating_point()
    ]
    if not flags:
        return torch.ones((), dtype=torch.bool, device=torch.cuda.current_device())
    return torch.stack(flags).all()


def amax(value) -> torch.Tensor:
    """The largest finite magnitude among the tensors of `value`."""
    found = [
        t.detach().float().nan_to_num(0, 0, 0).abs().amax()
        for t in tree.leaves(value)
        if t.is_floating_point() and t.numel()
    ]
    if not found:
        return torch.zeros((), device=torch.cuda.current_device())
    return torch.stack(found).amax()


class Watch:
    """Device flags and magnitudes recorded during one step, read only when it goes bad."""

    def __init__(self):
        self.flags: dict[str, torch.Tensor] = {}
        self.peaks: dict[str, torch.Tensor] = {}

    def note(self, name: str, value) -> None:
        self.flags[name] = finite(value)
        self.peaks[name] = amax(value)

    def bad(self) -> dict:
        names = list(self.flags)
        flags = torch.stack([self.flags[n] for n in names]).cpu().tolist()
        peaks = torch.stack([self.peaks[n].float() for n in names]).cpu().tolist()
        return {n: p for n, f, p in zip(names, flags, peaks) if not f}, dict(
            zip(names, peaks)
        )


def instrument(task, watch: Watch) -> None:
    """Record the slot boundaries, the loss terms and the seeds of every micro-step.

    The wrappers sit on the task's own calls, outside any compiled graph.
    """
    encoder, decode, render = task._encoder, task._decode, task._render
    serialize = task._serialize

    def watched_encoder(x):
        out = encoder(x)
        watch.note("out/encoder", out)
        return out

    def watched_serialize(slots, offset=0):
        encoded = serialize(slots, offset)
        watch.note("out/latent", encoded.latent)
        watch.note("loss/latent", encoded.losses)
        return encoded

    def watched_decode(latent):
        out = decode(latent)
        watch.note("out/decoder", out)
        return out

    def watched_render(features):
        out = render(features)
        watch.note("out/head", out)
        return out

    task._encoder, task._serialize = watched_encoder, watched_serialize
    task._decode, task._render = watched_decode, watched_render
    seeded = task.objective.seeded

    def watched_seeded(x, x_hat, scale):
        seeds, total, terms = seeded(x, x_hat, scale)
        watch.note("seed", seeds)
        watch.note("loss/terms", terms)
        return seeds, total, terms

    task.objective.seeded = watched_seeded


def watch_grads(bundle, model, watch: Watch) -> None:
    """Record every parameter gradient's flag before the bundle unscales and steps."""
    names = {id(p): n for n, p in model.named_parameters()}
    step = bundle.step

    def watched_step():
        for group in bundle.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    watch.note(f"grad/{names.get(id(param), '?')}", param.grad)
        return step()

    bundle.step = watched_step


def replay(task, weights: dict, batch, precision, device, scale: float) -> dict:
    """Rerun one step eagerly at `scale`; the first modules whose outputs or input grads
    were not finite, in call order, and the largest output magnitudes."""
    model = task.model
    compiled = {
        m: m._compiled_call_impl for m in model.modules() if m._compiled_call_impl
    }
    for module in compiled:
        module._compiled_call_impl = None
    model.load_state_dict(weights)
    model.zero_grad(set_to_none=True)

    forward, backward = [], []
    named = [(name, module) for name, module in model.named_modules() if name]
    handles = [
        m.register_forward_hook(
            lambda _, a, out, n=n: forward.append((n, finite(out), amax(out)))
        )
        for n, m in named
    ]
    handles += [
        m.register_full_backward_hook(
            lambda _, gin, gout, n=n: backward.append(
                (n, finite([g for g in gin if g is not None]), amax(list(gout)))
            )
        )
        for n, m in named
    ]
    with precision.autocast(torch.device(device).type):
        out, encoded = model(batch)
        fake = tree.map_tensors(lambda t: t.float(), out)
        seeds, _, _ = task.objective.seeded(batch, fake, scale)
        aux = sum(encoded.losses.values())
    (tree.dot(fake, seeds) * scale + aux * scale).backward()
    for handle in handles:
        handle.remove()
    for module, call in compiled.items():
        module._compiled_call_impl = call

    def summary(records):
        flags = torch.stack([f for _, f, _ in records]).cpu().tolist()
        peaks = torch.stack([p.float() for _, _, p in records]).cpu().tolist()
        names = [n for n, _, _ in records]
        first = [
            {"module": names[i], "peak_before": peaks[i - 1] if i else None}
            for i, ok in enumerate(flags)
            if not ok
        ][:5]
        return first, sorted(zip(peaks, names), reverse=True)[:8]

    forward_first, forward_top = summary(forward)
    backward_first, backward_top = summary(backward)
    grads = {
        n: bool(finite(p.grad))
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return {
        "forward_first_nonfinite": forward_first,
        "forward_top_peaks": forward_top,
        "backward_first_nonfinite": backward_first,
        "backward_top_grad_peaks": backward_top,
        "seeds_finite": bool(finite(seeds)),
        "nonfinite_param_grads": [n for n, ok in grads.items() if not ok][:10],
    }


def main() -> None:
    """Train until `STOP_AFTER` non-finite steps; one row per such step."""
    torch.manual_seed(SEED)
    torch.cuda.set_device(DEVICE)
    k = knobs_of()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DUMPS.mkdir(parents=True, exist_ok=True)

    domain = build(k["ARCH"].get("domain", "image"), DOMAIN)
    precision = build(k["PRECISION"], PRECISION)
    source = _sources(k["TRAIN_SOURCES"], domain, k["SIZE"], True)
    task = build_task(k, DEVICE)
    bundles = build_bundles(task, k, MAX_STEPS)
    ema = build_ema(task, k)
    watch = Watch()
    instrument(task, watch)
    watch_grads(bundles["model"], task.model, watch)

    timer = StepTimer(0, DEVICE)
    driver = Driver(task, bundles, ema, precision, DEVICE, k["GRAD_ACC"], sync, timer)
    batches = epoch_batches(source, k["BATCH_SIZE"], NUM_WORKERS, 0, 1, SEED)
    state, found = TrainState(), 0
    task.train()
    for batch, state.epoch, state.batches in batches:
        if state.step >= MAX_STEPS or found >= STOP_AFTER:
            break
        batch = task.to_device(batch, DEVICE)
        before = {n: t.detach().clone() for n, t in task.model.state_dict().items()}
        watch.flags.clear()
        watch.peaks.clear()
        metrics = driver.step(batch, state)
        state.step += 1
        if bool(torch.isfinite(metrics["optim/grad_norm/model"])):
            if state.step % 200 == 0:
                print(f"step {state.step} finite", flush=True)
            continue

        # A non-finite step: what the watch saw, then an eager replay of the same step
        found += 1
        bad, peaks = watch.bad()
        dump = DUMPS / f"seed{SEED}_step{state.step:06d}.pt"
        torch.save(
            {"weights": before, "batch": tree.map_tensors(torch.Tensor.cpu, batch)},
            dump,
        )
        scale = float(
            metrics["optim/loss_scale/model"]
        )  # the scale this step ran under
        eager = replay(task, before, batch, precision, DEVICE, scale)
        task.model.load_state_dict(before)
        task.train()
        row = {
            "seed": SEED,
            "step": state.step,
            "scale": scale,
            "nonfinite": {n: p for n, p in list(bad.items())[:20]},
            "nonfinite_count": len(bad),
            "boundary_peaks": {
                n: p for n, p in peaks.items() if not n.startswith("grad/")
            },
            "eager": eager,
            "dump": str(dump),
        }
        with OUT.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row)[:600], flush=True)


if __name__ == "__main__":
    main()
