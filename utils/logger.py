from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger("fcd_unet3d")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
