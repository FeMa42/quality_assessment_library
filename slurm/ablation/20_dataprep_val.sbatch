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
#SBATCH --job-name=cc1620_20_val
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 20 — VAL set full pipeline (200 objects).  Standalone (does not depend
# on the train stages).  render / render_cond sharded across BOTH L40S; the
# geometry chain runs on whichever GPU each toolkit picks (cheap on 200 objs).
# 2 GPUs requested only so the sharded render/render_cond steps can use both.
# Wrapped from $QA/runs/trellis_prep/COMMANDS.md (VAL set).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_TRELLIS_PREP"
cd "$TRELLIS/dataset_toolkits"

S=carcaption3k_1620_val
D="$TRELLIS/datasets/$S"

# 1. Build VAL metadata.csv + stage mesh symlinks (QA python).
"/home/damian/miniconda3/envs/$ENV_QA/bin/python" \
    "$QA/scripts/build_trellis_metadata.py" \
    --manifest "$QA/manifests/carcaption3k_1620_locked.csv" --split val --out-dir "$D"

python build_metadata.py CarCaption3K1620 --output_dir "$D"

# 2. render (150v) sharded across 2 GPUs.
CUDA_VISIBLE_DEVICES=0 python render.py CarCaption3K1620 --output_dir "$D" \
    --num_views 150 --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 python render.py CarCaption3K1620 --output_dir "$D" \
    --num_views 150 --rank 1 --world_size 2 &
wait
python build_metadata.py CarCaption3K1620 --output_dir "$D"

# 3. geometry chain.
python voxelize.py        CarCaption3K1620 --output_dir "$D"
python build_metadata.py  CarCaption3K1620 --output_dir "$D"

python extract_feature.py --output_dir "$D" --batch_size 16
python build_metadata.py  CarCaption3K1620 --output_dir "$D"

python encode_ss_latent.py --output_dir "$D"
python build_metadata.py   CarCaption3K1620 --output_dir "$D"

python encode_latent.py    --output_dir "$D"
python build_metadata.py   CarCaption3K1620 --output_dir "$D"

# 4. render_cond (24v) sharded across 2 GPUs.
CUDA_VISIBLE_DEVICES=0 python render_cond.py CarCaption3K1620 --output_dir "$D" \
    --num_views 24 --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 python render_cond.py CarCaption3K1620 --output_dir "$D" \
    --num_views 24 --rank 1 --world_size 2 &
wait
python build_metadata.py CarCaption3K1620 --output_dir "$D"

echo "Stage 20 (val full pipeline) complete: $D"
