"""Run folder: config snapshot, git revision and resume drift."""

import json
import os
import subprocess
import time
from pathlib import Path

DRIFT_IGNORED = {
    "RESUME",
    "RESUME_IGNORE",
    "RUN_ID",
    "INIT_FROM",
    "WANDB_RUN_ID",
    "DIST_TIMEOUT_HOURS",
}


def jsonable(values: dict) -> dict:
    """Knobs as JSON values; anything else becomes its repr."""
    return json.loads(json.dumps(values, default=repr))


def git_revision() -> dict:
    """Commit and dirty flag of the source tree, or nulls outside a checkout."""
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return done.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "-s"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write through a temp file and rename."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def new_run_id() -> str:
    """A sortable run id."""
    return time.strftime("%Y%m%d-%H%M%S")


def resolve_run(knobs: dict) -> Path:
    """The run folder of this launch: the `RESUME` folder or a fresh one."""
    if knobs["RESUME"] is not None:
        return Path(knobs["RESUME"])
    return Path(knobs["RUNS_DIR"]) / knobs["NAME"] / (knobs["RUN_ID"] or new_run_id())


def open_run(run: Path, knobs: dict) -> None:
    """Create the folder and snapshot config and revision."""
    run.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run / "config.json", jsonable(knobs))
    write_json_atomic(run / "git.json", git_revision())


def drift_problems(run: Path, knobs: dict) -> list[str]:
    """Knobs that changed since the resumed run started, minus ignored ones."""
    saved_path = run / "config.json"
    if not saved_path.exists():
        return [f"RESUME {run} has no config.json"]
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    current = jsonable(knobs)
    ignored = DRIFT_IGNORED | set(knobs["RESUME_IGNORE"])
    changed = {
        key
        for key in set(saved) | set(current)
        if key not in ignored and saved.get(key) != current.get(key)
    }
    return [
        f"resume drift on {key}: was {saved.get(key)!r}, now {current.get(key)!r}; "
        "add it to RESUME_IGNORE to accept"
        for key in sorted(changed)
    ]
