"""Console logging utility optimized for CLI execution and real-time terminal feedback."""

import logging
import sys


def setup_logger(
    name: str = "ResamplingEngine",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger flushing to stdout.

    Args:
        name: Logger name.
        level: Logging severity level.

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

    return logger

