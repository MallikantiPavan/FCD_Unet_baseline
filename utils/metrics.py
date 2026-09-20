from __future__ import annotations

import torch


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> dict[str, float]:
    prediction = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    target = (target >= 0.5).float().flatten(1)
    tp = (prediction * target).sum(1)
    fp = (prediction * (1 - target)).sum(1)
    fn = ((1 - prediction) * target).sum(1)
    has_roi = target.sum(1) > 0
    dice_per_sample = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou_per_sample = (tp + smooth) / (tp + fp + fn + smooth)
    precision_per_sample = (tp + smooth) / (tp + fp + smooth)
    recall_per_sample = (tp + smooth) / (tp + fn + smooth)
    positive_count = int(has_roi.sum().item())
    empty_count = int((~has_roi).sum().item())
    empty_false_positive_count = int(((~has_roi) & (prediction.sum(1) > 0)).sum().item())
    if positive_count:
        positive_metrics = {
            "dice": dice_per_sample[has_roi].mean(),
            "iou": iou_per_sample[has_roi].mean(),
            "precision": precision_per_sample[has_roi].mean(),
            "recall": recall_per_sample[has_roi].mean(),
        }
    else:
        positive_metrics = {key: logits.new_tensor(0.0) for key in ("dice", "iou", "precision", "recall")}
    return {
        **{key: float(value.item()) for key, value in positive_metrics.items()},
        "positive_count": positive_count,
        "empty_count": empty_count,
        "empty_false_positive_count": empty_false_positive_count,
    }
