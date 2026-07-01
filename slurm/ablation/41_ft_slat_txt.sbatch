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
#SBATCH --job-name=cc1620_41_ft_slat_txt
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 41 — TRELLIS finetune: slat_flow_txt (structured-latent, text-cond).
# Budget: 100000 steps (reference_budgets.json).  1 GPU / --num_gpus 1.
# DEPENDS ON data prep (10, 11, 12, 20).
# RESUME: --ckpt latest default — re-submit to resume (may exceed 72h at 100k).
# Wrapped from $QA/runs/trellis_train/COMMANDS.md (Stage 2).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_TRELLIS_TRAIN"
cd "$TRELLIS"

python train.py \
    --config configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json \
    --output_dir outputs/carcaption3k_1620/slat_flow_txt_dit_XL_64l8p2_fp16 \
    --data_dir datasets/carcaption3k_1620_train \
    --eval_data_dir datasets/carcaption3k_1620_val \
    --finetune_ckpt ./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16.safetensors \
    --num_gpus 1

echo "Stage 41 (slat_flow_txt finetune) complete."
