from __future__ import annotations

import torch


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    target = (target >= 0.5).float().flatten(1)
    tp = (prediction * target).sum(1)
    fp = (prediction * (1 - target)).sum(1)
    fn = ((1 - prediction) * target).sum(1)
    dice = ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).mean()
    iou = ((tp + smooth) / (tp + fp + fn + smooth)).mean()
    precision = ((tp + smooth) / (tp + fp + smooth)).mean()
    recall = ((tp + smooth) / (tp + fn + smooth)).mean()
    return {"dice": float(dice.item()), "iou": float(iou.item()), "precision": float(precision.item()), "recall": float(recall.item())}
