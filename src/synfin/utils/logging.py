"""Logging setup for synfin."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure the root logger with console (and optional file) handlers.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG").
        log_file: Optional path to write log file.
        log_format: Log message format string.
        date_format: Date/time format string.

    Returns:
        Configured root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(log_format, datefmt=date_format)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
