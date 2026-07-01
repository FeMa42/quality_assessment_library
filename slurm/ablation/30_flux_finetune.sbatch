#!/bin/bash
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=cc1620_30_flux
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 30 — FLUX LoRA finetune (8000 steps; see reference_budgets.json).
# Config: ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml
#   (push_to_hub: DamianBoborzi/flux_carcaption3k_1620_lora32 — matches
#    FLUX_LORA_REPO / FLUX_LORA_WEIGHT defaults in env.sh).
# Standalone (does not depend on TRELLIS data prep).
#
# ENV: uses the `ai_toolkit` conda env (ENV_FLUX), built & verified 2026-06-04
# by ./00b_build_ai_toolkit_env.sh (torch 2.6.0+cu126, diffusers@pinned,
# transformers 4.49, opencv-python-headless, bitsandbytes, peft, optimum-quanto;
# snapshot in ai_toolkit_freeze.txt). If the env is missing, run that script once.
# (To run on the HPC instead, use ai-toolkit/start_finetune.slurm as the template:
#  module load anaconda/cuda/cudnn/nccl + HTTP(S) proxy + epyc-gpu partition.)
#
# PREREQUISITES:
#  1. Regenerate the (gitignored) FLUX training dataset:
#       python "$QA/scripts/build_flux_filtered_dataset.py"
#     -> config expects images+captions at $QA/data/ablation/carcaption3k_1620_flux_train
#  2. First run downloads FLUX.1-schnell (~24GB, Apache-2.0, no token) + the
#     ostris/FLUX.1-schnell-training-adapter to the HF cache (needs network;
#     on HPC set HF_HOME + the proxy).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_FLUX"
cd "$AITK"

# Ensure the (gitignored) FLUX training dataset exists before training.
# Uncomment to (re)build it on the fly under the QA env:
# "/home/damian/miniconda3/envs/$ENV_QA/bin/python" "$QA/scripts/build_flux_filtered_dataset.py"

python run.py config/flux_trellis_carcaption3k_1620.yaml

echo "Stage 30 (FLUX LoRA finetune) complete."
echo "CONFIRM the produced LoRA repo/weight and set FLUX_LORA_REPO/FLUX_LORA_WEIGHT in env.sh:"
echo "  FLUX_LORA_REPO  = $FLUX_LORA_REPO"
echo "  FLUX_LORA_WEIGHT= $FLUX_LORA_WEIGHT"
