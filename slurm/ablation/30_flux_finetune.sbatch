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
# !!! IMPORTANT — ENV GAP !!!
# There is NO `ai_toolkit` conda env on THIS cluster (verified via conda env
# list). This job will FAIL until you either:
#   (a) create an `ai_toolkit` env locally and set ENV_FLUX accordingly, OR
#   (b) run the FLUX finetune on the HPC cluster instead, using
#       ai-toolkit/start_finetune.slurm as the template (module load anaconda /
#       cuda / cudnn / nccl + the HTTP(S) proxy). That template hardcodes the
#       HPC paths and the epyc-gpu partition.
#
# PREREQUISITE: regenerate the FLUX training dataset first (it is gitignored):
#   python "$QA/scripts/build_flux_filtered_dataset.py"
# The config expects images+captions at:
#   $QA/data/ablation/carcaption3k_1620_flux_train
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
