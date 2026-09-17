from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        target = target.float().flatten(1)
        intersection = (probabilities * target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class CombinedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.bce(logits, target)
