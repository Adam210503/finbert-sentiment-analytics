"""
config/logging_config.py
────────────────────────
Call setup_logging() once at the top of any entry-point script
(e.g. src/scheduler.py) to configure the root logger.

Convention used throughout this project:
    logger = logging.getLogger(__name__)
    logger.info("...")
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_file: Path, level_file: int = logging.DEBUG) -> None:
    """
    Configure root logger with two handlers:
      - Console (INFO): concise, human-readable
      - Rotating file (DEBUG): full detail, max 5 × 2MB files

    Parameters
    ----------
    log_file : Path
        Destination file for structured logs.
    level_file : int
        Logging level for the file handler (default DEBUG).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_console = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)

    # Rotating file — DEBUG and above, 2 MB × 5 backups
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level_file)
    fh.setFormatter(fmt_file)
    root.addHandler(fh)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
