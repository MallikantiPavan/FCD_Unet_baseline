from __future__ import annotations

import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FCD3DDataset, load_subject_dataframe
from dataset.augmentation import build_eval_transforms
from models import UNet3D
from utils.config import load_config
from utils.metrics import segmentation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained 3D FCD U-Net")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    config_path = checkpoint_path.with_name("config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    requested = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested)
    model = UNet3D(**{key: config["model"][key] for key in ("in_channels", "out_channels", "base_channels", "num_levels", "norm", "activation")})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    frame = load_subject_dataframe(config["data"]["test_csv"])
    data_root = Path(config["data"]["cropped_root"])
    available_subjects = []
    skipped_subjects = []
    for participant_id in frame["participant_id"]:
        subject_dir = data_root / str(participant_id)
        required_files = (subject_dir / "T1w_brain.nii.gz", subject_dir / "FLAIR_brain.nii.gz")
        if subject_dir.is_dir() and all(path.is_file() for path in required_files):
            available_subjects.append(participant_id)
        else:
            skipped_subjects.append(str(participant_id))
    frame = frame[frame["participant_id"].isin(available_subjects)].reset_index(drop=True)
    print(f"Test subjects in CSV: {len(available_subjects) + len(skipped_subjects)}")
    print(f"Test subjects available: {len(available_subjects)}")
    if skipped_subjects:
        print(f"Skipped missing subjects ({len(skipped_subjects)}): {', '.join(skipped_subjects)}")
    dataset = FCD3DDataset(frame, config, build_eval_transforms())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["training"]["num_workers"])
    output_dir = Path(config["output"]["predictions_root"]) / checkpoint_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {key: 0.0 for key in ("dice", "iou", "precision", "recall")}
    positive_count = 0
    empty_count = 0
    empty_false_positive_count = 0
    logger = logging.getLogger("fcd_unet3d.test")
    for batch in loader:
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        with torch.no_grad():
            logits = model(image)
        metric = segmentation_metrics(logits, mask, config["metrics"]["threshold"])
        positive_count += metric["positive_count"]
        empty_count += metric["empty_count"]
        empty_false_positive_count += metric["empty_false_positive_count"]
        if metric["positive_count"]:
            for key in totals:
                totals[key] += metric[key] * metric["positive_count"]
        participant_id = batch["participant_id"][0]
        prediction = (torch.sigmoid(logits)[0, 0] >= config["metrics"]["threshold"]).cpu().numpy().astype(np.uint8)
        subject_dir = Path(config["data"]["cropped_root"]) / participant_id
        reference = nib.load(str(subject_dir / "FLAIR_brain.nii.gz"))
        nib.save(nib.Nifti1Image(prediction, reference.affine, reference.header), str(output_dir / f"{participant_id}_pred.nii.gz"))
        if config["output"].get("save_probability_maps", True):
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(probability, reference.affine, reference.header), str(output_dir / f"{participant_id}_prob.nii.gz"))
        logger.info("%s | dice %.5f | iou %.5f | precision %.5f | recall %.5f", participant_id, metric["dice"], metric["iou"], metric["precision"], metric["recall"])
    if len(dataset) == 0:
        raise ValueError("Test dataset is empty")
    print(f"Test subjects: {len(dataset)}")
    for key, value in totals.items():
        print(f"Test FCD {key.capitalize()}: {value / positive_count if positive_count else 0.0:.5f}")
    print(f"Healthy-control false-positive rate: {empty_false_positive_count / empty_count if empty_count else 0.0:.5f}")
    print(f"Predictions: {output_dir}")


if __name__ == "__main__":
    main()
