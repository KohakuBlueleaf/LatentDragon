"""Structured logger: `log.info("message", key=value)`, rank 0 writes by default."""

import logging
import os
import sys


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


class Logger:
    """Thin wrapper over `logging` that renders keyword fields after the message."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)

    def _emit(self, level: int, message: str, fields: dict) -> None:
        if _rank() != 0:
            return
        text = " ".join(f"{k}={v}" for k, v in fields.items())
        self._log.log(level, f"{message} {text}" if text else message)

    def debug(self, message: str, **fields) -> None:
        self._emit(logging.DEBUG, message, fields)

    def info(self, message: str, **fields) -> None:
        self._emit(logging.INFO, message, fields)

    def warning(self, message: str, **fields) -> None:
        self._emit(logging.WARNING, message, fields)


def get_logger(name: str) -> Logger:
    """A named logger; the first call installs a stderr handler on the root package."""
    root = logging.getLogger("latentdragon")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return Logger(f"latentdragon.{name}")
