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
#SBATCH --job-name=cc1620_40_ft_ss_txt
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 40 — TRELLIS finetune: ss_flow_txt (sparse-structure, text-cond).
# Budget: 100000 steps (reference_budgets.json).  1 GPU / --num_gpus 1 to MATCH
# the MeshFleet finetune effective batch size.
# DEPENDS ON data prep (10, 11, 12, 20).
#
# RESUME: train.py defaults to --ckpt latest, so re-submitting THIS SAME script
# resumes from the latest checkpoint in output_dir.  100k steps may exceed 72h —
# just resubmit (submit_all.sh wires the dependency; manual resubmit works too).
# Wrapped from $QA/runs/trellis_train/COMMANDS.md (Stage 1).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_TRELLIS_TRAIN"
cd "$TRELLIS"

python train.py \
    --config configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json \
    --output_dir outputs/carcaption3k_1620/ss_flow_txt_dit_XL_16l8_fp16 \
    --data_dir datasets/carcaption3k_1620_train \
    --eval_data_dir datasets/carcaption3k_1620_val \
    --finetune_ckpt ./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors \
    --num_gpus 1

echo "Stage 40 (ss_flow_txt finetune) complete."
