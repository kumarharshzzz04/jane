"""Minimal structured logging."""
import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str = "jane") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
