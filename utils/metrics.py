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
    empty_predictions = prediction.sum(1)[~has_roi]
    empty_false_positive_count = int((empty_predictions > 0).sum().item())
    fp_volume_hc = float(empty_predictions.mean().item()) if empty_count else 0.0
    if positive_count:
        positive_metrics = {
            "dice_fcd": dice_per_sample[has_roi].mean(),
            "iou_fcd": iou_per_sample[has_roi].mean(),
            "precision": precision_per_sample[has_roi].mean(),
            "recall": recall_per_sample[has_roi].mean(),
        }
    else:
        positive_metrics = {key: logits.new_tensor(0.0) for key in ("dice_fcd", "iou_fcd", "precision", "recall")}
    return {
        **{key: float(value.item()) for key, value in positive_metrics.items()},
        "positive_count": positive_count,
        "empty_count": empty_count,
        "empty_false_positive_count": empty_false_positive_count,
        "fp_volume_hc": fp_volume_hc,
    }
