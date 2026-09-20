from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


def load_subject_dataframe(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "participant_id" not in frame.columns:
        raise ValueError(f"{csv_path} must contain a participant_id column")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame["participant_id"].duplicated().any():
        duplicates = frame.loc[frame["participant_id"].duplicated(), "participant_id"].tolist()
        raise ValueError(f"Duplicate participant_id values in {csv_path}: {duplicates}")
    return frame.reset_index(drop=True)


def split_dataframe(frame: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_index = int(round(len(shuffled) * train_ratio))
    split_index = min(max(split_index, 1), len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    return shuffled.iloc[:split_index].reset_index(drop=True), shuffled.iloc[split_index:].reset_index(drop=True)


def _normalize(array: np.ndarray, bounds: Mapping[str, Any], clip: bool) -> np.ndarray:
    lower, upper = float(bounds["min"]), float(bounds["max"])
    denominator = upper - lower
    if denominator <= 0:
        raise ValueError(f"Invalid normalization bounds: min={lower}, max={upper}")
    normalized = (array.astype(np.float32) - lower) / denominator
    return np.clip(normalized, 0.0, 1.0) if clip else normalized


class FCD3DDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, config: Mapping[str, Any], transforms=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.config = config
        self.root = Path(config["data"]["cropped_root"])
        self.transforms = transforms if isinstance(transforms, (list, tuple)) else [transforms]
        self.expected_shape = tuple(config["data"].get("expected_shape", []))
        self.validate_expected_shape = bool(config["data"].get("validate_expected_shape", True))
        self.clip = bool(config["normalization"].get("clip", True))
        self.stats = {"subjects_with_roi": 0, "subjects_without_roi": 0, "positive_roi_voxels": 0}
        if len(self.dataframe) == 0:
            raise ValueError("Dataset is empty")

    def __len__(self) -> int:
        return len(self.dataframe) * len(self.transforms)

    def _load(self, path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
        image = nib.load(str(path))
        array = np.asanyarray(image.dataobj)
        if not np.isfinite(array).all():
            raise ValueError(f"NaN or Inf values found in {path}")
        return np.asarray(array), image

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_index, transform_index = divmod(index, len(self.transforms))
        participant_id = str(self.dataframe.iloc[subject_index]["participant_id"])
        subject_dir = self.root / participant_id
        if not subject_dir.is_dir():
            raise FileNotFoundError(f"Missing subject directory for {participant_id}: {subject_dir}")
        t1_path, flair_path = subject_dir / "T1w_brain.nii.gz", subject_dir / "FLAIR_brain.nii.gz"
        roi_path = subject_dir / "FLAIR_roi.nii.gz"
        for path in (t1_path, flair_path):
            if not path.is_file():
                raise FileNotFoundError(f"{participant_id}: missing required file {path}")
        t1, t1_img = self._load(t1_path)
        flair, flair_img = self._load(flair_path)
        if t1.shape != flair.shape:
            raise ValueError(f"{participant_id}: T1 shape {t1.shape} != FLAIR shape {flair.shape}")
        if self.validate_expected_shape and t1.shape != self.expected_shape:
            raise ValueError(f"{participant_id}: expected shape {self.expected_shape}, got {t1.shape}")
        if roi_path.is_file():
            roi, roi_img = self._load(roi_path)
            if roi.shape != flair.shape:
                raise ValueError(f"{participant_id}: ROI shape {roi.shape} != MRI shape {flair.shape}")
            unique_roi = np.unique(roi)
            if not np.isin(unique_roi, (0, 1)).all():
                raise ValueError(f"{participant_id}: ROI contains unsupported values {unique_roi[:20].tolist()}; expected strictly 0/1")
            self.stats["subjects_with_roi"] += 1
        else:
            roi = np.zeros(flair.shape, dtype=np.uint8)
            self.stats["subjects_without_roi"] += 1
            LOGGER.warning("%s: ROI missing -> using zero mask", participant_id)
        mask = (roi > 0).astype(np.float32)
        self.stats["positive_roi_voxels"] += int(mask.sum())
        image = np.stack([
            _normalize(t1, self.config["normalization"]["T1w_brain"], self.clip),
            _normalize(flair, self.config["normalization"]["FLAIR_brain"], self.clip),
        ], axis=0).astype(np.float32)
        sample = {
            "image": image,
            "mask": torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0),
            "participant_id": participant_id,
        }
        transform = self.transforms[transform_index]
        if transform is not None:
            sample = transform(sample)
        sample["image"] = torch.as_tensor(sample["image"], dtype=torch.float32)
        sample["mask"] = (torch.as_tensor(sample["mask"], dtype=torch.float32) > 0.5).float()
        return sample
