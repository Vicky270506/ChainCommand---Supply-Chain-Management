"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys

import structlog

from chaincommand.config import settings

_logging_initialized = False


def setup_logging(quiet: bool = False) -> None:
    """Configure structlog + stdlib logging.

    Args:
        quiet: If True, suppress logs below WARNING to avoid interfering
               with Rich terminal UI output.

    This function is guarded against double initialization. Subsequent calls
    are no-ops.
    """
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    log_level = (
        logging.WARNING if quiet
        else getattr(logging, settings.log_level.upper(), logging.INFO)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=log_level,
        stream=sys.stderr,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
