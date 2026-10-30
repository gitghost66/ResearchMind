"""
Structured JSON logging setup.

Every line is one JSON object. Fields a caller passes in ``extra`` (``run_id``,
``agent``, ...) are included, and so are the fields bound with
:func:`log_context`: a worker binds its ID once at startup and a run's ID while
it executes, so every line logged inside a run — by the backend, the agents
package, or a library — carries ``run_id`` and ``worker_id`` without being
passed them. Context variables are inherited by the tasks a run starts, so
LangGraph's parallel branches carry them too.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

__all__ = ["JSONFormatter", "bind_log_context", "log_context", "setup_logging"]

# Fields copied from a record's ``extra`` when present.
_EXTRA_FIELDS = ("run_id", "worker_id", "agent", "latency_ms", "cost_usd")

_CONTEXT: dict[str, ContextVar[str | None]] = {
    "run_id": ContextVar("log_run_id", default=None),
    "worker_id": ContextVar("log_worker_id", default=None),
}


def bind_log_context(**fields: str | None) -> None:
    """Bind fields for the rest of the current context (e.g. a worker's ID)."""
    for name, value in fields.items():
        _CONTEXT[name].set(value)


@contextmanager
def log_context(**fields: str | None) -> Iterator[None]:
    """Bind fields (``run_id``, ``worker_id``) to every log line inside the block."""
    tokens = [(_CONTEXT[name], _CONTEXT[name].set(value)) for name, value in fields.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is None and key in _CONTEXT:
                value = _CONTEXT[key].get()
            if value is not None:
                log_entry[key] = value
        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure root logger."""
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )

    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
