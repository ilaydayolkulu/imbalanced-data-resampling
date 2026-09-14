"""Dual-channel logging utility optimized for CLI execution and persistent file auditing."""

import logging
from pathlib import Path
import sys
from typing import Optional


def setup_logger(
    name: str = "ResamplingEngine",
    level: int = logging.INFO,
    log_file_path: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a dual-channel logger flushing to stdout and optional file.

    Args:
        name: Logger name.
        level: Logging severity level.
        log_file_path: Optional path to append log entries in UTF-8 encoding.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure stdout console handler is attached exactly once
    has_console = any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in logger.handlers)
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file_path is not None:
        attach_file_handler(logger, log_file_path)

    return logger


def attach_file_handler(logger: logging.Logger, log_file_path: Path) -> None:
    """Attaches a persistent UTF-8 FileHandler to an existing logger instance.

    Args:
        logger: Target logger instance.
        log_file_path: Destination path for the log file.
    """
    resolved_path = Path(log_file_path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if a FileHandler targeting this exact file is already registered
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_path:
            return

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(str(resolved_path), encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
