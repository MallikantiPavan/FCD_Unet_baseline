from __future__ import annotations

from typing import Any, Mapping

from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    EnsureTyped,
)


def build_train_transforms(config: Mapping[str, Any]) -> Compose:
    aug = config.get("augmentation", {})
    if not aug.get("enabled", True):
        return build_eval_transforms()
    transforms = [
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=aug.get("flip_prob", 0.5)),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=aug.get("flip_prob", 0.5)),
        RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=aug.get("flip_prob", 0.5)),
        RandAffined(
            keys=["image", "mask"],
            prob=aug.get("rotate_prob", 0.2),
            rotate_range=tuple(float(aug.get("max_rotate_degrees", 10)) * 3.141592653589793 / 180 for _ in range(3)),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=aug.get("intensity_prob", 0.2),
            gamma=(1.0 - float(aug.get("intensity_scale", 0.1)), 1.0 + float(aug.get("intensity_scale", 0.1))),
        ),
        EnsureTyped(keys=["image", "mask"]),
    ]
    return Compose(transforms)


def build_eval_transforms() -> Compose:
    return Compose([EnsureTyped(keys=["image", "mask"])])
