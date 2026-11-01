"""Non-blocking metric log: device scalars are staged and read one flush late."""

import torch

from latentdragon.training.parallel import sync


class MetricLog:
    """Collects per-step scalars; a flush starts a copy and emits the previous one."""

    def __init__(self, sinks: list, every: int = 1):
        self.sinks = sinks
        self.every = max(every, 1)
        self._rows: list[tuple[int, dict]] = []
        self._staged: tuple | None = None
        self._device: torch.device | None = None

    def record(self, step: int, values: dict) -> None:
        """Hold one step's scalars; tensors stay on their device."""
        self._rows.append((step, values))
        if len(self._rows) >= self.every:
            self.flush()

    def flush(self) -> None:
        """Emit the staged rows, then stage the held ones."""
        self._emit()
        rows, self._rows = self._rows, []
        if not rows:
            return

        # Flatten to (row, key, value) cells; tensor cells are packed into one vector.
        cells = [
            (i, key, value)
            for i, (_, values) in enumerate(rows)
            for key, value in values.items()
        ]
        tensors = [v.detach().float().reshape(()) for _, _, v in cells if _is_tensor(v)]

        # Mean across ranks and start the async host copy; read at the next emit.
        pack = None
        if tensors:
            pack = sync.all_mean_(torch.stack(tensors)).to("cpu", non_blocking=True)
            self._device = tensors[0].device
        self._staged = (rows, cells, pack)

    def _emit(self) -> None:
        """Write the staged rows to every sink, reading tensor cells from the host copy."""
        if self._staged is None:
            return
        rows, cells, pack = self._staged
        self._staged = None

        # Rebuild rows, taking tensor cells from the packed values in order.
        out = [{"step": step} for step, _ in rows]
        read = iter(pack.tolist()) if pack is not None else iter(())
        for i, key, value in cells:
            out[i][key] = next(read) if _is_tensor(value) else value

        for row in out:
            for sink in self.sinks:
                sink.write(row)

    def close(self) -> None:
        """Emit everything and close the sinks."""
        self.flush()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._emit()
        for sink in self.sinks:
            sink.close()


def _is_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)
