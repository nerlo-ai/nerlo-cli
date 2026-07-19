"""Minimal, dependency-free structured logger for the Nerlo CLI.

Mirrors the `logger.info(event, **kwargs)` call surface the commands use (a
structlog-style key/value shape) without shipping structlog: debug/info are
silent unless `NERLO_DEBUG=1`; warnings and errors always go to stderr. Keeping
the CLI's only two runtime deps `click` + `httpx`.
"""

from __future__ import annotations

import os
import sys
from typing import Any


class _CliLogger:
    def __init__(self, name: str) -> None:
        self._name = name
        self._debug = os.environ.get("NERLO_DEBUG") == "1"

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        if level in ("debug", "info") and not self._debug:
            return
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"[{level}] {event}" + (f" {extras}" if extras else "")
        print(line, file=sys.stderr)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("debug", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("info", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit("warning", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("error", event, **fields)


def get_logger(name: str) -> _CliLogger:
    return _CliLogger(name)
