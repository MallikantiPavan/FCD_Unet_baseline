from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FCD3DDataset, load_subject_dataframe, split_dataframe
from dataset.augmentation import build_eval_transforms, build_train_transforms
from models import UNet3D
from utils.checkpoint import next_run_dir, save_checkpoint
from utils.config import load_config
from utils.logger import configure_logger
from utils.losses import bce_loss, dice_loss,focal_loss
from utils.metrics import segmentation_metrics
from utils.seed import seed_everything


def _device(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _build_model(config: dict[str, Any]) -> UNet3D:
    return UNet3D(**{key: config["model"][key] for key in ("in_channels", "out_channels", "base_channels", "num_levels", "norm", "activation")})


def _run_epoch(model, loader, config, device, threshold, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, leave=False)):
        image, mask = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
        with autocast(enabled=scaler is not None):
            logits = model(image)
            loss_type = config["loss"]["type"]

            if loss_type == "focal_bce":
                loss = (
                    config["loss"]["focal_weight"] * focal_loss(
                        logits,
                        mask,
                        alpha=config["loss"]["focal_alpha"],
                        gamma=config["loss"]["focal_gamma"],
                    )
                    + config["loss"]["bce_weight"] * bce_loss(logits, mask)
                )
            elif loss_type == "dice_bce":
                loss = (
                    config["loss"]["dice_weight"] * dice_loss(
                        logits,
                        mask,
                        smooth=config["loss"]["smooth"],
                    )
                    + config["loss"]["bce_weight"] * bce_loss(logits, mask)
                )
            elif loss_type == "focal_dice":
                loss = (
                    config["loss"]["focal_weight"] * focal_loss(
                        logits,
                        mask,
                        alpha=config["loss"]["focal_alpha"],
                        gamma=config["loss"]["focal_gamma"],
                    )
                    + config["loss"]["dice_weight"] * dice_loss(
                        logits,
                        mask,
                        smooth=config["loss"]["smooth"],
                    )
                )
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")


        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_metrics = segmentation_metrics(logits.detach(), mask, threshold)
        totals["loss"] += float(loss.item())
        for key in batch_metrics:
            totals[key] += batch_metrics[key]
    return {key: value / len(loader) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3D FCD U-Net")
    parser.add_argument(
        "--config",
        help="Path to the YAML configuration file, for example config/config.yaml",
        default="config/config.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    run_dir = next_run_dir(config["checkpoint"]["root_dir"])
    logger = configure_logger(run_dir / "train.log")
    run_config = json.loads(json.dumps(config))
    train_frame = load_subject_dataframe(config["data"]["train_csv"])
    test_frame = load_subject_dataframe(config["data"]["test_csv"])
    train_frame, val_frame = split_dataframe(train_frame, float(config["split"]["train_ratio"]), int(config["seed"]))
    if train_frame.empty or val_frame.empty:
        raise ValueError("The train.csv split must produce non-empty training and validation datasets")
    sets = {"train": set(train_frame.participant_id), "validation": set(val_frame.participant_id), "test": set(test_frame.participant_id)}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"Subject overlap between {left} and {right}: {sorted(overlap)}")
    run_config["split_subjects"] = {key: sorted(value) for key, value in sets.items()}
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_config, handle, sort_keys=False)
    logger.info("Device: %s | Seed: %s", _device(config), config["seed"])
    logger.info("Train subjects: %d | Validation subjects: %d | Test subjects: %d", len(train_frame), len(val_frame), len(test_frame))
    

    logger.info("Loss type: %s", config["loss"]["type"])
    
    train_dataset = FCD3DDataset(train_frame, config, build_train_transforms(config))
    val_dataset = FCD3DDataset(val_frame, config, build_eval_transforms())
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"])
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"])
    device = _device(config)
    model = _build_model(config).to(device)
    logger.info("Model parameters: %d", sum(parameter.numel() for parameter in model.parameters()))
    optimizer = AdamW(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode=config["scheduler"]["mode"], factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"], min_lr=config["scheduler"]["min_lr"])
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    best_dice, stale_epochs = float("-inf"), 0
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, config, device, config["metrics"]["threshold"], optimizer, scaler if amp_enabled else None)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, config, device, config["metrics"]["threshold"])
        scheduler.step(val_metrics["dice"])
        logger.info("Epoch %d/%d | train loss %.5f | val loss %.5f | val dice %.5f | val IoU %.5f | lr %.3g", epoch, config["training"]["epochs"], train_metrics["loss"], val_metrics["loss"], val_metrics["dice"], val_metrics["iou"], optimizer.param_groups[0]["lr"])
        if val_metrics["dice"] > best_dice:
            best_dice, stale_epochs = val_metrics["dice"], 0
            save_checkpoint(run_dir / "best_point.pth", model, optimizer, scheduler, epoch, best_dice, run_config)
            logger.info("New best validation Dice %.5f saved to %s", best_dice, run_dir / "best_point.pth")
        else:
            stale_epochs += 1
        patience = int(config["training"].get("early_stopping_patience", 0))
        if patience and stale_epochs >= patience:
            logger.info("Early stopping after %d stale epochs", stale_epochs)
            break
    logger.info("Finished. Checkpoint directory: %s", run_dir)


if __name__ == "__main__":
    main()
