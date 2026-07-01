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
#SBATCH --job-name=cc1620_12_train_cond
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 12 — TRAIN set conditioning renders (1420 objects, 24 views @ 1024px).
# render_cond ~2.5 min/obj ⇒ ~59 GPU-h on 1 GPU ⇒ shard across BOTH L40S.
# DEPENDS ON stage 10 (needs the train metadata.csv built there).  Independent
# of stage 11, so it can run in parallel with the geometry chain.
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

# render_cond sharded across the 2 L40S (rank0 → GPU0, rank1 → GPU1).
CUDA_VISIBLE_DEVICES=0 python render_cond.py CarCaption3K1620 --output_dir "$D" \
    --num_views 24 --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 python render_cond.py CarCaption3K1620 --output_dir "$D" \
    --num_views 24 --rank 1 --world_size 2 &
wait

# Merge per-rank cond_rendered_0.csv + cond_rendered_1.csv into metadata.csv.
python build_metadata.py CarCaption3K1620 --output_dir "$D"

echo "Stage 12 (train cond render) complete: $D"
