from __future__ import annotations

from typing import Any, Mapping

from monai.transforms import (
    Compose,
    Flipd,
    RandAdjustContrastd,
    RandAffined,
    EnsureTyped,
)


def build_train_transforms(config: Mapping[str, Any]) -> list[Compose] | Compose:
    aug = config.get("augmentation", {})
    if not aug.get("enabled", True):
        return build_eval_transforms()

    keys = ["image", "mask"]
    original = Compose([EnsureTyped(keys=keys)])
    flipped = Compose([
        Flipd(keys=keys, spatial_axis=(0, 1, 2)),
        EnsureTyped(keys=keys),
    ])
    rotated = Compose([
        RandAffined(
            keys=["image", "mask"],
            prob=1.0,
            rotate_range=tuple(float(aug.get("max_rotate_degrees", 10)) * 3.141592653589793 / 180 for _ in range(3)),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        EnsureTyped(keys=keys),
    ])
    intensity = Compose([
        RandAdjustContrastd(
            keys=["image"],
            prob=1.0,
            gamma=(1.0 - float(aug.get("intensity_scale", 0.1)), 1.0 + float(aug.get("intensity_scale", 0.1))),
        ),
        EnsureTyped(keys=keys),
    ])
    return [original, flipped, rotated, intensity]


def build_eval_transforms() -> Compose:
    return Compose([EnsureTyped(keys=["image", "mask"])])
