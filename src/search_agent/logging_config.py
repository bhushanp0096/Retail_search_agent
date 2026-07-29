"""
Centralized logging setup.

Call `setup_logging()` once, early, from any entrypoint (demo scripts,
the compiled graph's `main`, tests' conftest, etc). Every other module
just does `logger = logging.getLogger(__name__)` and inherits the config.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from search_agent.config import LOG_DIR, settings

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None, log_to_file: bool | None = None) -> None:
    """Idempotent logging setup. Safe to call multiple times (e.g. once per
    entrypoint, once per test module) — configuration only happens once.

    Args:
        level: Overrides `settings.log_level` (e.g. "DEBUG", "INFO", "WARNING").
        log_to_file: Overrides `settings.log_to_file`.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = (level or settings.log_level).upper()
    resolved_log_to_file = settings.log_to_file if log_to_file is None else log_to_file

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if resolved_log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"search_agent_{session_ts}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Quiet down noisy third-party loggers unless we're in DEBUG mode.
    if resolved_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

    _CONFIGURED = True
