# FCD 3D U-Net

A modular, reproducible 3D binary segmentation pipeline for cropped T1-weighted and FLAIR MRI volumes. It never slices volumes into 2D inputs and uses a genuine 3D U-Net with two input channels and one logit output channel.

## Data

Each subject directory under `data.cropped_root` must contain `T1w_brain.nii.gz` and `FLAIR_brain.nii.gz`. `FLAIR_roi.nii.gz` is optional; when absent, the dataset creates `np.zeros(flair.shape, dtype=np.uint8)` and logs the subject. CSV files must contain `participant_id`; clinical columns are ignored.

The default expected shape is `(159, 190, 160)`. Shapes, NaNs/Infs, and affine compatibility are checked. No reorientation, resize, crop, or padding is performed.

T1w and FLAIR are independently normalized using the fixed constants in `config/config.yaml`, then optionally clipped to `[0, 1]`. No dataset statistics are calculated.

## Splits and reproducibility

`train.csv` is shuffled with the configured seed and split 85/15 by default. `test.csv` is loaded only for overlap checking during training and evaluation. Train, validation, and test IDs are checked for overlap and stored in each run configuration.

## Commands

```bash
pip install -r requirements.txt
python validate_dataset.py --config config/config.yaml
python train.py --config config/config.yaml
python test.py --checkpoint checkpoint/checkpoint_1/best_point.pth
```

Training creates a fresh `checkpoint/checkpoint_N/` directory, never overwriting an earlier run. It contains `best_point.pth`, the exact `config.yaml`, and `train.log`. The checkpoint is selected by validation Dice only.

## Architecture and training

The model uses MONAI's configurable 3D U-Net with InstanceNorm3d by default, encoder downsampling, decoder upsampling, skip concatenation, and a one-channel output head. The model returns logits. Loss is configurable Dice plus BCE-with-logits; metrics threshold sigmoid probabilities and report Dice, IoU, precision, and recall.

Training defaults to batch size 1, AMP on CUDA, AdamW, ReduceLROnPlateau monitored by validation Dice, gradient accumulation support, and configurable early stopping. Validation and test use deterministic transforms only. Training spatial transforms are dictionary transforms applied jointly to image and mask, with nearest interpolation for masks.

## Predictions

`test.py` writes binary `sub-XXXXX_pred.nii.gz` and optional float32 `sub-XXXXX_prob.nii.gz` files under `predictions/checkpoint_N/`. They preserve the FLAIR reference affine and header and retain the original volume shape.

## Troubleshooting

A CUDA request fails clearly when CUDA is unavailable. For out-of-memory errors, reduce `base_channels`, use batch size 1, enable AMP, or increase gradient accumulation; do not alter the anatomical shape. Missing ROI files are allowed, while missing MRI files, shape mismatches, invalid normalization ranges, NaNs, and affine mismatches in strict mode are errors.

## Expected tensor shapes

A sample image is `[2, 159, 190, 160]`, a sample mask is `[1, 159, 190, 160]`, a batch is `[B, 2, 159, 190, 160]`, and model logits are `[B, 1, 159, 190, 160]`.
