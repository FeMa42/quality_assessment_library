#!/bin/bash
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#SBATCH --job-name=cc1620_10_train_render
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 10 — TRAIN set render (1420 objects, 150 views).
# render ~4.5 min/obj ⇒ ~106 GPU-h on 1 GPU (OVER 72h) ⇒ shard across BOTH L40S.
# Builds TRELLIS metadata first (CPU/instant), then renders rank0+rank1 in
# parallel, then merges the per-rank rendered_*.csv via build_metadata.py.
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

# 1. Build TRELLIS metadata.csv + stage mesh symlinks (QA python; pandas).
#    Runs under ENV_QA because it needs the QA-library pandas stack.
"/home/damian/miniconda3/envs/$ENV_QA/bin/python" \
    "$QA/scripts/build_trellis_metadata.py" \
    --manifest "$QA/manifests/carcaption3k_1620_locked.csv" --split train --out-dir "$D"

python build_metadata.py CarCaption3K1620 --output_dir "$D"

# 2. Render sharded across the 2 L40S (rank0 → GPU0, rank1 → GPU1).
CUDA_VISIBLE_DEVICES=0 python render.py CarCaption3K1620 --output_dir "$D" \
    --num_views 150 --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 python render.py CarCaption3K1620 --output_dir "$D" \
    --num_views 150 --rank 1 --world_size 2 &
wait

# 3. Merge per-rank rendered_0.csv + rendered_1.csv into metadata.csv.
python build_metadata.py CarCaption3K1620 --output_dir "$D"

echo "Stage 10 (train render) complete: $D"
