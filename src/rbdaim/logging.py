"""Structured logging.

JSON logs in production (machine-parseable, one event per line); a readable
console renderer when attached to a TTY. Use ``get_logger(__name__)``.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", *, json_logs: bool | None = None) -> None:
    """Configure structlog once at process start.

    Args:
        level: Root log level.
        json_logs: Force JSON (True) or console (False). Defaults to JSON when
            stdout is not a TTY (i.e. in containers and CI).

    """
    if json_logs is None:
        json_logs = not sys.stdout.isatty()
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
