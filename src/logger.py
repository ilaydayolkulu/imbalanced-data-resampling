"""Logging utility optimized for CLI execution and C++/Qt QProcess real-time streaming."""

import json
import logging
import sys
from typing import Any, Dict, Optional


def setup_logger(name: str = "ResamplingEngine", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger that flushes stdout immediately for QProcess.

    Args:
        name: Logger name.
        level: Logging severity level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Standard stream handler with flush guarantee
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def log_qt_event(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    """Emits a structured progress/success telemetry message to stdout for C++/Qt.

    Args:
        event_type: Telemetry event identifier (e.g., 'PIPELINE_STARTED', 'PIPELINE_SUCCESS').
        data: Key-value payload associated with the event.
    """
    payload = {
        "source": "resampling_pipeline",
        "event": event_type,
        "payload": data or {}
    }
    sys.stdout.write(f"@QT_EVENT:{json.dumps(payload)}\n")
    sys.stdout.flush()
