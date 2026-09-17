from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_central_slice(image: np.ndarray, mask: np.ndarray, prediction: np.ndarray, output_path: str | Path) -> None:
    depth = image.shape[-3] // 2
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image[0, depth], cmap="gray")
    axes[1].imshow(image[1, depth], cmap="gray")
    axes[2].imshow(mask[depth], cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(prediction[depth], cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes, ("T1w", "FLAIR", "Ground truth", "Prediction")):
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
