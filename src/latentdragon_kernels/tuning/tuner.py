"""Pick one config per (kernel, problem key): stored winner, else time every candidate.

JSON tables per card + arch + torch version; mode "off" binds the first candidate.
"""

import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from latentdragon_common.log import get_logger
from latentdragon_kernels.errors import NoImplementationError
from latentdragon_kernels.hardware.arch import arch_of

CACHE_ENV = "LATENTDRAGON_KERNEL_CACHE"
MODES = ("on", "off")
ROUNDS = 5
log = get_logger("latentdragon.kernels")
_STATE = {"mode": "on"}
_CHOSEN: dict[tuple[str, str, str], dict] = {}


def set_mode(mode: str) -> None:
    """Set the tuning mode and forget every settled choice."""
    if mode not in MODES:
        raise ValueError(f"tuning mode {mode!r} not in {MODES}")
    _STATE["mode"] = mode
    reset()


def get_mode() -> str:
    return _STATE["mode"]


def reset() -> None:
    """Forget every choice settled in this process."""
    _CHOSEN.clear()


def cache_dir() -> Path:
    """Root of the tables: `CACHE_ENV`, else under the user cache."""
    root = os.environ.get(CACHE_ENV)
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "latentdragon_kernels"


def card_key(device: torch.device | str) -> str:
    """Table folder name of `device`: card, arch and torch version."""
    device = torch.device(device)
    arch = arch_of(device)
    card = "cpu" if arch == "cpu" else torch.cuda.get_device_name(device)
    slug = re.sub(r"[^a-z0-9]+", "-", card.lower()).strip("-")
    return f"{slug}-{arch}-torch{torch.__version__}"


def table_path(kernel: str, device: torch.device | str) -> Path:
    return cache_dir() / "tuning" / card_key(device) / f"{kernel}.json"


def read_table(path: Path) -> dict[str, dict]:
    """The rows at `path`; a missing or unreadable file reads as empty."""
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return rows if isinstance(rows, dict) else {}


def write_row(path: Path, key: str, row: dict) -> None:
    """Merge `row` into the table atomically; a failed write logs and keeps going."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = {**read_table(path), key: row}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as err:
        log.warning("cannot write tuning table", path=path, error=err)


def _bucket(size: int) -> int:
    """Smallest power of two >= `size`."""
    return 1 << max(size - 1, 0).bit_length()


def shape_key(**dims: int) -> str:
    """Problem key: each dimension in its power-of-two bucket."""
    return ",".join(f"{name}={_bucket(size)}" for name, size in dims.items())


def config_name(config: dict) -> str:
    return ",".join(f"{key}={value}" for key, value in sorted(config.items()))


def _flush(device: torch.device) -> None:
    """Wait for queued work on a CUDA `device`."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn: Callable[[], object], device: torch.device | str) -> float:
    """Best-of-`ROUNDS` milliseconds of `fn` after one warm-up call."""
    device = torch.device(device)
    fn()
    _flush(device)
    best = float("inf")
    for _ in range(ROUNDS):
        began = time.perf_counter()
        fn()
        _flush(device)
        best = min(best, (time.perf_counter() - began) * 1e3)
    return best


def _time_all(kernel, device, key, candidates, launch, errors) -> dict:
    """Time every candidate, store the fastest; a config raising `errors` is skipped."""
    timings, failures, best = {}, {}, None

    # Time each candidate, tracking the fastest
    for config in candidates:
        name = config_name(config)
        try:
            timings[name] = measure(launch(config), device)
        except torch.OutOfMemoryError:
            raise
        except errors as err:
            failures[name] = f"{type(err).__name__}: {err}"[:300]
            continue
        if best is None or timings[name] < timings[config_name(best)]:
            best = config
    if best is None:
        raise NoImplementationError(
            f"{kernel} {key}: every candidate failed: {failures}"
        )

    # Persist the winner with all timings and failures
    row = {"config": best, "timings_ms": timings, "failures": failures}
    write_row(table_path(kernel, device), key, row)
    log.info("kernel tuned", kernel=kernel, key=key, config=config_name(best))
    return best


def select(
    kernel: str,
    device: torch.device | str,
    key: str,
    candidates: Sequence[dict],
    launch: Callable[[dict], Callable[[], object]],
    errors: tuple[type[BaseException], ...] = (RuntimeError,),
) -> dict:
    """The config for problem `key`; `launch(config)` is a callable running it once."""
    # Settled earlier in this process
    ident = (str(device), kernel, key)
    chosen = _CHOSEN.get(ident)
    if chosen is not None:
        return chosen

    # Untuned mode or a single candidate binds the first; otherwise try the stored winner
    if get_mode() == "off" or len(candidates) == 1:
        chosen = candidates[0]
    else:
        row = read_table(table_path(kernel, device)).get(key, {})
        stored = row.get("config")
        chosen = stored if stored in candidates else None

    # No stored winner: time every candidate
    if chosen is None:
        chosen = _time_all(kernel, device, key, candidates, launch, errors)
    _CHOSEN[ident] = chosen
    return chosen
