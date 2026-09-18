from __future__ import annotations

import torch


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).flatten(1)
    target = target.float().flatten(1)
    intersection = (probabilities * target).sum(dim=1)
    denominator = probabilities.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    return 1 - dice.mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float())


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    target = target.float()
    binary_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    probability_of_correct_class = torch.exp(-binary_cross_entropy)
    focal_factor = (1.0 - probability_of_correct_class).pow(gamma)
    alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_factor * focal_factor * binary_cross_entropy).mean()


