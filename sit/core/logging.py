"""Structured logging for SIT.

Provides a logger factory that includes run_id in all log records,
supporting auditability and traceability of every pipeline step.
"""

import logging
import sys
from typing import Optional


_CONFIGURED = False


def setup_logging(level: str = "INFO", run_id: Optional[str] = None) -> None:
    """Configure the SIT root logger.

    Sets up a structured format with timestamps and optional run_id.
    Safe to call multiple times; only configures on the first call
    unless force is needed.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        run_id: Optional run identifier to include in all messages.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger("sit")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if run_id:
        fmt = f"%(asctime)s | %(levelname)-8s | run={run_id} | %(name)s | %(message)s"
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    root_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the SIT namespace.

    Args:
        name: Logger name (will be prefixed with 'sit.').

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(f"sit.{name}")


def reset_logging() -> None:
    """Reset the logging configuration. Primarily for testing."""
    global _CONFIGURED
    root_logger = logging.getLogger("sit")
    root_logger.handlers.clear()
    _CONFIGURED = False
