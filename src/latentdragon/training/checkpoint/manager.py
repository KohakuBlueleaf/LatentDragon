"""Atomic checkpoints: `.partial` directory renamed last, `state.json` written last."""

import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from latentdragon.training.parallel import sync
from latentdragon_common.log import get_logger

log = get_logger(__name__)
PREFIX = "step_"
WRITE_ERRORS = (OSError, RuntimeError, SafetensorError)


def step_dir(root: Path, step: int) -> Path:
    """The final directory of a step's checkpoint."""
    return Path(root) / f"{PREFIX}{step:08d}"


def latest(root: Path) -> Path | None:
    """The newest complete checkpoint under `root`, or None."""
    root = Path(root)
    if not root.is_dir():
        return None
    done = sorted(
        p
        for p in root.iterdir()
        if p.name.startswith(PREFIX)
        and not p.name.endswith(".partial")
        and (p / "state.json").exists()
    )
    return done[-1] if done else None


def flat_state(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    """Name-keyed tensors of every saved module, cloned so safetensors accepts them."""
    return {
        f"{name}.{key}": value.detach().cpu().clone().contiguous()
        for name, module in modules.items()
        for key, value in module.state_dict().items()
    }


def save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Write tensors with their fp32 or integer dtype preserved."""
    plain = {key: value.contiguous() for key, value in tensors.items()}
    save_file(plain, str(path))


def rng_state() -> dict:
    """This rank's torch RNG states."""
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


class CheckpointManager:
    """Writes and restores the full training state of one run."""

    def __init__(self, root: Path, keep: int = 0, rank: int = 0, retries: int = 3):
        self.root = Path(root)
        self.keep = keep
        self.rank = rank
        self.retries = retries

    def save(self, state, modules, bundles, ema, extra: dict | None = None) -> None:
        """Save one step; a failed write logs and training continues once a checkpoint exists."""
        final = step_dir(self.root, state.step)
        partial = final.with_name(final.name + ".partial")

        # Retry the write with exponential backoff.
        written = False
        for attempt in range(self.retries):
            try:
                self._write(partial, state, modules, bundles, ema, extra or {})
                written = True
                break
            except WRITE_ERRORS as err:
                log.warning("checkpoint write failed", attempt=attempt, error=err)
                time.sleep(2**attempt)
        sync.barrier()

        # Rank 0 publishes the finished directory and drops old ones.
        if self.rank == 0 and (partial / "state.json").exists():
            os.replace(partial, final)
            self._prune()
        if not written and latest(self.root) is None:
            raise OSError(f"no checkpoint could be written under {self.root}")

    def _write(self, partial, state, modules, bundles, ema, extra) -> None:
        """Fill the `.partial` directory; every rank writes RNG, rank 0 the rest."""
        partial.mkdir(parents=True, exist_ok=True)
        torch.save(rng_state(), partial / f"rng.{self.rank}.pt")
        if self.rank != 0:
            return

        # Weights, EMA shadow and optimizer state.
        save_tensors(flat_state(modules), partial / "trainable.safetensors")
        ema_state = ema.state_dict()
        if ema_state:
            shadow = {k: v.cpu().clone() for k, v in ema_state["shadow"].items()}
            save_tensors(shadow, partial / "ema.safetensors")
        optim = {name: b.state_dict() for name, b in bundles.items()}
        torch.save(optim, partial / "optim.pt")

        # state.json goes last: its presence marks the checkpoint complete.
        payload = {
            **state.to_dict(),
            "ema_updates": ema_state.get("updates", 0),
            **extra,
        }
        text = json.dumps(payload, indent=2)
        (partial / "state.json").write_text(text, encoding="utf-8")

    def _prune(self) -> None:
        """Delete all but the newest `keep` finished checkpoints; 0 keeps all."""
        if self.keep <= 0:
            return
        done = sorted(p for p in self.root.iterdir() if p.name.startswith(PREFIX))
        for old in [p for p in done if not p.name.endswith(".partial")][: -self.keep]:
            shutil.rmtree(old)

    def restore(self, path: Path, state, modules, bundles, ema) -> dict:
        """Load a checkpoint into live objects; returns its `state.json` payload."""
        path = Path(path)
        payload = json.loads((path / "state.json").read_text(encoding="utf-8"))

        # Module weights, split back out by their name prefix.
        tensors = load_file(str(path / "trainable.safetensors"))
        for name, module in modules.items():
            prefix = f"{name}."
            sub = {
                k.removeprefix(prefix): v
                for k, v in tensors.items()
                if k.startswith(prefix)
            }
            module.load_state_dict(sub, strict=True)

        # Optimizer bundles and EMA shadow.
        optim = torch.load(path / "optim.pt", map_location="cpu", weights_only=True)
        for name, bundle in bundles.items():
            bundle.load_state_dict(optim[name])
        ema_path = path / "ema.safetensors"
        if ema_path.exists():
            ema.load_state_dict(
                {"updates": payload["ema_updates"], "shadow": load_file(str(ema_path))}
            )

        # This rank's RNG streams, when saved.
        rng_path = path / f"rng.{self.rank}.pt"
        if not rng_path.exists():
            log.warning("no rng state for this rank", path=str(rng_path))
            rng_path = None
        if rng_path is not None:
            rng = torch.load(rng_path, map_location="cpu", weights_only=True)
            torch.set_rng_state(rng["cpu"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        state.step = payload["step"]
        state.epoch = payload["epoch"]
        state.batches = payload["batches"]
        return payload
