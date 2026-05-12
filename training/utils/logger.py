from __future__ import annotations

import logging
import sys
from pathlib import Path


class _MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


class _MinLevelFilter(logging.Filter):
    def __init__(self, min_level: int) -> None:
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


class _OneLineFormatter(logging.Formatter):
    """Formatter that keeps exception output on a single parse-friendly line."""

    def formatException(self, ei: tuple[type[BaseException], BaseException, Any] | tuple[Any, Any, Any]) -> str:
        text = super().formatException(ei)
        return text.replace("\n", " | ").replace("\r", " ")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text.replace("\n", " | ").replace("\r", " ")


_RUN_ERROR_LOG_PATH: Path | None = None


def _remove_run_error_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_run_error_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _attach_run_error_handler(logger: logging.Logger) -> None:
    if _RUN_ERROR_LOG_PATH is None:
        return

    for handler in logger.handlers:
        if getattr(handler, "_run_error_handler", False) and getattr(handler, "_run_error_path", None) == str(_RUN_ERROR_LOG_PATH):
            return

    _RUN_ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = _OneLineFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_file_handler = logging.FileHandler(_RUN_ERROR_LOG_PATH, encoding="utf-8")
    error_file_handler.setFormatter(formatter)
    error_file_handler.addFilter(_MinLevelFilter(logging.WARNING))
    setattr(error_file_handler, "_run_error_handler", True)
    setattr(error_file_handler, "_run_error_path", str(_RUN_ERROR_LOG_PATH))
    logger.addHandler(error_file_handler)


def configure_run_error_log(run_dir: str | Path) -> None:
    """Configure run-scoped error log for all training.* loggers.

    Warnings/errors/exceptions are written to <run_dir>/error.log as one-line entries.
    """

    global _RUN_ERROR_LOG_PATH
    _RUN_ERROR_LOG_PATH = Path(run_dir) / "error.log"

    logger_dict = logging.Logger.manager.loggerDict
    for name, obj in logger_dict.items():
        if isinstance(obj, logging.Logger) and (name == "training" or name.startswith("training.")):
            _remove_run_error_handlers(obj)
            _attach_run_error_handler(obj)


def get_logger(name: str = "training") -> logging.Logger:
    """Return a configured logger instance without duplicate handlers.

    Routing policy:
    - DEBUG/INFO -> stdout
    - WARNING/ERROR/CRITICAL -> stderr
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(_MinLevelFilter(logging.WARNING))

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    _attach_run_error_handler(logger)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger