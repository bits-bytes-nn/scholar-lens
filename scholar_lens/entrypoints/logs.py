"""Logging of the entrypoints: console, and a dated log file outside AWS."""

import functools
import logging
import os
from datetime import datetime
from pathlib import Path

from scholar_lens.adapters.aws.environment import is_running_in_aws
from scholar_lens.domain.constants import EnvVars, LocalPaths

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def _log_file() -> Path:
    """The day's log file under the repository's logs directory."""
    logs_dir = Path(__file__).resolve().parent.parent.parent / LocalPaths.LOGS_DIR.value
    logs_dir.mkdir(parents=True, exist_ok=True)
    name, ext = LocalPaths.LOGS_FILE.value.rsplit(".", 1)
    return logs_dir / f"{name}_{datetime.now():%Y-%m-%d}.{ext}"


def get_default_logger(name: str = "app") -> logging.Logger:
    """``name``'s logger writing to the console and, outside AWS, the day's
    log file, at the level of ``LOG_LEVEL`` (default INFO); its handlers
    are replaced, so a second call does not duplicate records."""
    logger = logging.getLogger(name)
    level = os.getenv(EnvVars.LOG_LEVEL.value, "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    formatter = logging.Formatter(_FORMAT)
    logger.handlers.clear()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if not is_running_in_aws():
        handlers.append(logging.FileHandler(_log_file(), encoding="utf-8", delay=True))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


@functools.cache
def configure_logging() -> logging.Logger:
    """Send the package's records (every module logs through
    ``logging.getLogger(__name__)``) to the console and, outside AWS, the
    day's log file; the level comes from ``LOG_LEVEL``. Once per process."""
    return get_default_logger("scholar_lens")
