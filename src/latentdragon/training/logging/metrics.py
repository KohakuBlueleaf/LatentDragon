"""Non-blocking metric log: device values are staged, read one flush late, written off-thread.

A step's tensor values are packed into one device vector when recorded, so a flush is one
concatenation, one cross-rank mean and one host copy. A 1-D tensor under key `K` is a series
and becomes `K00`, `K01`, ... when written. Sinks run on a writer thread, in order, so the
training thread never waits on wandb or file writes.
"""

import queue
import threading

import torch

from latentdragon.training.parallel import sync

STOP = object()


class SinkWriter:
    """One thread calling the sinks in the order work was queued."""

    def __init__(self, sinks: list):
        self.sinks = sinks
        self.work: queue.Queue = queue.Queue()
        self.thread = threading.Thread(
            target=self._loop, name="metric-sinks", daemon=True
        )
        self.thread.start()

    def _loop(self) -> None:
        while (item := self.work.get()) is not STOP:
            method, args = item
            for sink in self.sinks:
                if hasattr(sink, method):
                    getattr(sink, method)(*args)

    def put(self, method: str, *args) -> None:
        """Call `sink.method(*args)` on every sink that has it, after earlier work."""
        self.work.put((method, args))

    def close(self) -> None:
        """Finish queued work, then close every sink on the calling thread."""
        self.work.put(STOP)
        self.thread.join()
        for sink in self.sinks:
            sink.close()


class MetricLog:
    """Collects per-step values; a flush starts a copy and emits the previous one."""

    def __init__(self, sinks: list, every: int = 1):
        self.sinks = sinks
        self.every = max(every, 1)
        self.writer = SinkWriter(sinks)
        self._rows: list[tuple] = []
        self._staged: tuple | None = None
        self._device: torch.device | None = None

    def record(self, step: int, values: dict) -> None:
        """Hold one step's values, its tensors packed into one device vector."""
        tensors = [(k, v) for k, v in values.items() if isinstance(v, torch.Tensor)]
        plain = {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)}
        layout = [(k, v.numel() if v.ndim else 0) for k, v in tensors]
        pack = None
        if tensors:
            pack = torch.cat([v.detach().float().reshape(-1) for _, v in tensors])
            self._device = pack.device
        self._rows.append((step, plain, layout, pack))
        if len(self._rows) >= self.every:
            with torch.cuda.nvtx.range("log_flush"):
                self.flush()

    def flush(self) -> None:
        """Emit the staged rows, then stage the held ones."""
        self._emit()
        rows, self._rows = self._rows, []
        if not rows:
            return

        # One vector for every held row: mean it across ranks and start the host copy
        packs = [pack for *_, pack in rows if pack is not None]
        joined, done = None, None
        if packs:
            joined = sync.all_mean_(torch.cat(packs)).to("cpu", non_blocking=True)
            done = _copy_event(packs[0].device)
        self._staged = (rows, joined, done)

    def _emit(self) -> None:
        """Hand the staged rows, read from their host copy, to the writer thread."""
        if self._staged is None:
            return
        rows, joined, done = self._staged
        self._staged = None
        if done is not None:
            done.synchronize()

        values = joined.tolist() if joined is not None else []
        offset = 0
        for step, plain, layout, _ in rows:
            row = {"step": step, **plain}
            for key, size in layout:
                if size == 0:
                    row[key] = values[offset]
                    offset += 1
                    continue
                for index in range(size):
                    row[f"{key}{index:02}"] = values[offset + index]
                offset += size
            self.writer.put("write", row)

    def image(self, step: int, key: str, path) -> None:
        """Send a sample grid file to every sink that takes images, after earlier rows."""
        self.writer.put("log_image", step, key, path)

    def close(self) -> None:
        """Emit everything, finish the writer and close the sinks."""
        self.flush()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._emit()
        self.writer.close()


def _copy_event(device: torch.device) -> torch.cuda.Event | None:
    """An event that fires once the queued host copy on `device` has landed."""
    if device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record()
    return event
