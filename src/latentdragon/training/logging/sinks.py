"""Metric sinks: a row is `{"step": int, **scalars}`."""

import json
from pathlib import Path

from latentdragon_common.log import get_logger

try:
    import wandb
except ImportError:
    wandb = None

log = get_logger(__name__)


class JsonlSink:
    """Appends rows to `metrics.jsonl`; `truncate_after` drops rows past a resume."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def truncate_after(self, step: int) -> None:
        """Remove rows later than `step`."""
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if json.loads(line)["step"] <= step]
        self.path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    def close(self) -> None:
        return None


class ConsoleSink:
    """Logs the headline keys of each row."""

    KEYS = ("train/eloss", "train/psnr", "vq/usage")

    def write(self, row: dict) -> None:
        fields = {k: round(row[k], 4) for k in self.KEYS if k in row}
        log.info("step", step=row["step"], **fields)

    def close(self) -> None:
        return None


class WandbSink:
    """Optional wandb run; construction fails loudly when wandb is missing."""

    def __init__(self, project: str, name: str, config: dict, **kwargs):
        if wandb is None:
            raise ImportError("WANDB_PROJECT is set but wandb is not installed")
        self.run = wandb.init(project=project, name=name, config=config, **kwargs)

    def write(self, row: dict) -> None:
        self.run.log({k: v for k, v in row.items() if k != "step"}, step=row["step"])

    def close(self) -> None:
        self.run.finish()
