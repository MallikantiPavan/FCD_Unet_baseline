from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import FCD3DDataset
from dataset.augmentation import build_eval_transforms
from models import UNet3D


def main() -> None:
    shape = (17, 19, 21)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "cropped"
        subject = root / "sub-test"
        subject.mkdir(parents=True)
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32), affine), subject / "T1w_brain.nii.gz")
        nib.save(nib.Nifti1Image(np.ones(shape, np.float32) * 2, affine), subject / "FLAIR_brain.nii.gz")
        nib.save(nib.Nifti1Image((np.indices(shape)[0] == 8).astype(np.uint8), affine), subject / "FLAIR_roi.nii.gz")
        config = yaml.safe_load(Path("config/config.yaml").read_text())
        config["data"].update(cropped_root=str(root), expected_shape=list(shape), validate_expected_shape=True)
        config["training"]["num_workers"] = 0
        dataset = FCD3DDataset(__import__("pandas").DataFrame({"participant_id": ["sub-test"]}), config, build_eval_transforms())
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        model = UNet3D(in_channels=2, out_channels=1, base_channels=2, num_levels=3, norm="instance", activation="relu")
        output = model(batch["image"])
        assert tuple(batch["image"].shape) == (1, 2, *shape)
        assert tuple(batch["mask"].shape) == (1, 1, *shape)
        assert tuple(output.shape) == (1, 1, *shape)
        print("smoke test passed", tuple(batch["image"].shape), tuple(output.shape))


if __name__ == "__main__":
    main()
