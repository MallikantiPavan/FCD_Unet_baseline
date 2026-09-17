#!/bin/bash
#SBATCH --job-name=fcd_smoke
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/projects/vinkle/ez_compass_imaging/code/Unet_Raw/jobs/logs/%x-%j.out
#SBATCH --error=/storage/projects/vinkle/ez_compass_imaging/code/Unet_Raw/jobs/logs/%x-%j.err

source $(conda info --base)/etc/profile.d/conda.sh
conda activate /storage/projects/vinkle/ez_compass_imaging/envs/fcd_env


python train.py --config configs/config.yaml