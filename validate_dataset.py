from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from dataset import load_subject_dataframe
from utils.config import load_config


def inspect(csv_path: str, config: dict) -> list[str]:
    frame = load_subject_dataframe(csv_path)
    root = Path(config["data"]["cropped_root"])
    expected = tuple(config["data"].get("expected_shape", []))
    tolerance = float(config["data"].get("affine_tolerance", 1e-3))
    errors = []
    missing_roi = 0
    for participant_id in frame["participant_id"]:
        subject = root / participant_id
        if not subject.is_dir():
            errors.append(f"{participant_id}: missing subject directory")
            continue
        paths = {name: subject / filename for name, filename in (("T1w", "T1w_brain.nii.gz"), ("FLAIR", "FLAIR_brain.nii.gz"), ("ROI", "FLAIR_roi.nii.gz"))}
        if not paths["T1w"].is_file() or not paths["FLAIR"].is_file():
            errors.append(f"{participant_id}: missing T1w or FLAIR")
            continue
        images = {name: nib.load(str(path)) for name, path in paths.items() if path.is_file()}
        arrays = {name: np.asanyarray(image.dataobj) for name, image in images.items()}
        if not np.isfinite(arrays["T1w"]).all() or not np.isfinite(arrays["FLAIR"]).all():
            errors.append(f"{participant_id}: NaN/Inf in MRI")
        if arrays["T1w"].shape != arrays["FLAIR"].shape:
            errors.append(f"{participant_id}: T1w shape {arrays['T1w'].shape} != FLAIR shape {arrays['FLAIR'].shape}")
        if config["data"].get("validate_expected_shape", True) and arrays["T1w"].shape != expected:
            errors.append(f"{participant_id}: expected {expected}, got {arrays['T1w'].shape}")
        if "ROI" in arrays:
            if arrays["ROI"].shape != arrays["FLAIR"].shape:
                errors.append(f"{participant_id}: ROI shape mismatch")
            if not np.isfinite(arrays["ROI"]).all():
                errors.append(f"{participant_id}: NaN/Inf in ROI")
            unique_roi = np.unique(arrays["ROI"])
            if not np.isin(unique_roi, (0, 1)).all():
                errors.append(f"{participant_id}: unsupported ROI values {unique_roi[:20].tolist()}")
            print(f"{participant_id}: ROI unique={np.unique(arrays['ROI'])[:20].tolist()} positive_voxels={(arrays['ROI'] > 0).sum()}")
            for name in ("FLAIR", "ROI"):
                if np.max(np.abs(images["FLAIR"].affine - images[name].affine)) > tolerance:
                    errors.append(f"{participant_id}: affine mismatch for {name}")
        else:
            missing_roi += 1
            print(f"{participant_id}: ROI file missing -> zero mask during training")
    print(f"Subjects in {csv_path}: {len(frame)}")
    print(f"Subjects without ROI: {missing_roi}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FCD cropped NIfTI data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    errors = inspect(config["data"]["train_csv"], config) + inspect(config["data"]["test_csv"], config)
    if errors:
        print("Validation errors:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("Dataset validation passed")


if __name__ == "__main__":
    main()
