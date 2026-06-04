#!/bin/bash
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=cc1620_43_ft_slat_img
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 43 — TRELLIS finetune: slat_flow_img (structured-latent, image-cond).
# Budget: 30000 steps (reference_budgets.json — img-cond slat converges faster).
# 1 GPU / --num_gpus 1.  DEPENDS ON data prep (10, 11, 12, 20).
# RESUME: --ckpt latest default — re-submit to resume if needed.
# Wrapped from $QA/runs/trellis_train/COMMANDS.md (Stage 4).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_TRELLIS_TRAIN"
cd "$TRELLIS"

python train.py \
    --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json \
    --output_dir outputs/carcaption3k_1620/slat_flow_img_dit_L_64l8p2_fp16 \
    --data_dir datasets/carcaption3k_1620_train \
    --eval_data_dir datasets/carcaption3k_1620_val \
    --finetune_ckpt ./assets/pretrained_TRELLIS_image_large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \
    --num_gpus 1

echo "Stage 43 (slat_flow_img finetune) complete."
