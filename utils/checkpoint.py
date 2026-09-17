from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


def next_run_dir(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.split("_")[-1]) for path in root.glob("checkpoint_*") if path.name.split("_")[-1].isdigit()]
    run_dir = root / f"checkpoint_{max(numbers, default=0) + 1}"
    run_dir.mkdir()
    return run_dir


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_validation_metric: float, config: Mapping[str, Any]) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_validation_metric": best_validation_metric,
        "config": dict(config),
    }, path)
    with path.with_name("config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
