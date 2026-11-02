"""Metric sinks: a row is `{"step": int, "epoch": int, **scalars}`; images go by file path.

Every sink writes rows; a sink with `log_image` also takes sample grids.
"""

import json
from pathlib import Path

from tqdm import tqdm

from latentdragon_common.log import get_logger

try:
    import wandb
except ImportError:
    wandb = None

log = get_logger(__name__)

# The headline keys Lightning's progress bar showed in the research trainer
HEADLINE = ("train/eloss", "train/psnr", "train/ssim", "vq/usage")
STEP_KEY = "trainer/global_step"


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
    """Logs the headline keys and throughput of each row, for logs read after the fact."""

    def write(self, row: dict) -> None:
        keys = (*HEADLINE, "perf/samples_per_sec")
        fields = {key: round(row[key], 4) for key in keys if key in row}
        log.info("step", step=row["step"], epoch=row.get("epoch"), **fields)

    def close(self) -> None:
        return None


class ProgressBar:
    """A step bar with rate and ETA, its postfix the headline keys of the latest row."""

    def __init__(self, total: int, start: int, name: str):
        self.bar = tqdm(total=total, initial=start, desc=name, dynamic_ncols=True)

    def advance(self, step: int) -> None:
        """Move the bar to `step`."""
        self.bar.update(step - self.bar.n)

    def write(self, row: dict) -> None:
        postfix = {
            key.split("/")[-1]: f"{row[key]:.4g}" for key in HEADLINE if key in row
        }
        self.bar.set_postfix(postfix, refresh=False)

    def close(self) -> None:
        self.bar.close()


class WandbSink:
    """A wandb run that behaves like Lightning's `WandbLogger`.

    Every row carries `trainer/global_step` and `epoch`, every metric is plotted against
    `trainer/global_step`, and rows are logged without wandb's own step, so a resumed run or a
    late validation row never trips wandb's increasing-step rule.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        run_dir: Path,
        files: list[Path],
        run_id: str | None = None,
        resume: str | None = None,
        offline: bool = False,
    ):
        if wandb is None:
            raise ImportError("WANDB_PROJECT is set but wandb is not installed")
        self.run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume=resume,
            config=config,
            dir=str(run_dir),
            mode="offline" if offline else "online",
        )
        self.run.define_metric(STEP_KEY)
        self.run.define_metric("*", step_metric=STEP_KEY, step_sync=True)

        # The script and the resolved knobs, as the research trainer saved its script
        for path in files:
            self.run.save(str(path), base_path=str(Path(path).parent), policy="now")

    def write(self, row: dict) -> None:
        values = {key: value for key, value in row.items() if key != "step"}
        self.run.log({**values, STEP_KEY: row["step"]})

    def log_image(self, step: int, key: str, path: Path) -> None:
        """Upload one sample grid at `step`."""
        self.run.log({key: wandb.Image(str(path)), STEP_KEY: step})

    def summary(self, totals: dict, path: Path) -> None:
        """Parameter totals as `model/` summary values, and the table file."""
        for key, value in totals.items():
            self.run.summary[f"model/{key}"] = value
        self.run.save(str(path), base_path=str(Path(path).parent), policy="now")

    def watch(self, model, every: int) -> None:
        """Gradient and parameter histograms of `model` every `every` steps."""
        self.run.watch(model, log="all", log_freq=every)

    def close(self) -> None:
        self.run.finish()
