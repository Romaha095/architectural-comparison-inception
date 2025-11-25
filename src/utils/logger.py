from __future__ import annotations

import logging as py_logging
from pathlib import Path
from typing import Optional

_LOGGER_INITIALIZED = False


def setup_logging(
    log_dir: str | Path,
    log_level: str = "INFO",
    log_to_file: bool = True,
) -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(py_logging, log_level.upper(), py_logging.INFO)

    formatter = py_logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = py_logging.getLogger()
    root_logger.setLevel(level)

    console_handler = py_logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        log_file = log_dir / "training.log"
        file_handler = py_logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> py_logging.Logger:
    return py_logging.getLogger(name)
