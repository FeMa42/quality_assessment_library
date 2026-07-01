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
#SBATCH --job-name=cc1620_11_train_geom
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 11 — TRAIN set geometry chain: voxelize → extract_feature →
# encode_ss_latent → encode_latent, with build_metadata between stages to merge
# flags.  Single GPU is sufficient (these are far cheaper than render).
# DEPENDS ON stage 10 (needs rendered images on disk).
# Wrapped from $QA/runs/trellis_prep/COMMANDS.md (TRAIN set).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_TRELLIS_PREP"
cd "$TRELLIS/dataset_toolkits"

S=carcaption3k_1620_train
D="$TRELLIS/datasets/$S"

python voxelize.py        CarCaption3K1620 --output_dir "$D"
python build_metadata.py  CarCaption3K1620 --output_dir "$D"

python extract_feature.py --output_dir "$D" --batch_size 16
python build_metadata.py  CarCaption3K1620 --output_dir "$D"

python encode_ss_latent.py --output_dir "$D"
python build_metadata.py   CarCaption3K1620 --output_dir "$D"

python encode_latent.py    --output_dir "$D"
python build_metadata.py   CarCaption3K1620 --output_dir "$D"

echo "Stage 11 (train geometry) complete: $D"
