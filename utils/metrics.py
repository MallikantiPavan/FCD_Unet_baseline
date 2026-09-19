from __future__ import annotations

import torch


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    target = (target >= 0.5).float().flatten(1)
    tp = (prediction * target).sum(1)
    fp = (prediction * (1 - target)).sum(1)
    fn = ((1 - prediction) * target).sum(1)
    target_is_empty = (tp + fn) == 0
    prediction_is_empty = (tp + fp) == 0
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    empty_case_score = prediction_is_empty.float()
    dice = torch.where(target_is_empty, empty_case_score, dice)
    iou = torch.where(target_is_empty, empty_case_score, iou)
    precision = torch.where(target_is_empty, empty_case_score, precision)
    recall = torch.where(target_is_empty, torch.full_like(recall, float("nan")), recall)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(torch.nanmean(recall).item()),
    }
