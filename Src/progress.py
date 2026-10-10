# time stamp of process of project.
import logging
import sys
import time
from pathlib import Path


def setup_logger(log_file: Path | None = None, name: str = "transcribe") -> logging.Logger:
    """Timestamped logger to stdout (+ optional file), line-buffered."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class Phase:
    """Context manager that logs START / DONE / FAIL with elapsed time.

    Usage:
        with Phase(logger, "Load model"):
            ...
    """

    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"START  {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc_type:
            self.logger.error(f"FAIL   {self.name} after {dt:.1f}s: {exc}")
            return False
        self.logger.info(f"DONE   {self.name} in {dt:.1f}s")
        return False


def hhmmss(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    s = int(seconds or 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"