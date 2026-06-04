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
#SBATCH --job-name=cc1620_50_gen_eval_p0
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 50 — P0 · FLUX_TRELLIS_CC1620 : generate → assemble → eval.
# FLUX image-to-3D via base TRELLIS (no finetuned ss/slat ckpt), with the
# CarCaption3K-1620 FLUX LoRA.  DEPENDS ON stage 30 (FLUX LoRA).
# Based on $QA/runs/eval/COMMANDS.md (P0).
#
# !!! OPEN ITEMS — CONFIRM BEFORE RUNNING !!!
#  * --prompt_file must be the 232 HELD-OUT objects' prompts (column
#    refined_3d_prompt) in SHA ORDER. METADATA_CSV (meshfleet_test.csv) is the
#    held-out set, but VERIFY its row order corresponds 1:1 to the held-out shas
#    used by the assembly mapping below (see prepare_meshfleet_benchmark.ipynb).
#  * OI-3 assembly i→sha mapping: evaluate_trellis_prompt_following.py writes
#    glb_<sha>/ QA-spec frames; we copy them to <sha>/000.png..011.png in sorted
#    view order. CONFIRM the glb_* dir name carries the sha (it does in the
#    template) — if it carries a numeric index i instead, you must map i→sha via
#    the prompt_file row order.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

VARIANT="FLUX_TRELLIS_CC1620"
RAW_OUT="$TRELLIS/eval_prompt_alignment/output_flux_cc1620"
GEN_FOLDER="$QA/data/ablation/gen/$VARIANT"
METRICS_OUT="$QA/runs/$VARIANT/metrics"

# ---- (1) Generation (ENV_GEN) -------------------------------------------
activate "$ENV_GEN"
cd "$TRELLIS"

python evaluate_trellis_prompt_following.py \
    --output_dir "$RAW_OUT" \
    --use_flux \
    --sample_max_reward_image \
    --flux_lora_repo "$FLUX_LORA_REPO" \
    --flux_lora_weight_name "$FLUX_LORA_WEIGHT" \
    --prompt_file "$METADATA_CSV" \
    --seed 3

# ---- (2) Assembly: glb_<sha>/ → <sha>/000.png..011.png (OI-3) -------------
# Use the QA-spec glb_<sha>/ renders (radius 1.5, elev 90). Source from the
# nested renders/ subdir if present, else the glb_<sha>/ dir directly. Rename to
# zero-padded sorted view order so the eval can match GT↔gen by filename.
mkdir -p "$GEN_FOLDER"
for d in "$RAW_OUT"/glb_*/; do
    [ -d "$d" ] || continue
    sha="${d##*/glb_}"; sha="${sha%/}"
    src="$d"; [ -d "${d}renders" ] && src="${d}renders"
    out="$GEN_FOLDER/$sha"; mkdir -p "$out"
    i=0
    for f in $(ls "$src"/*.png 2>/dev/null | sort); do
        printf -v name "%03d.png" "$i"
        cp "$f" "$out/$name"
        i=$((i+1))
    done
done

# ---- (3) Eval (ENV_QA) ---------------------------------------------------
activate "$ENV_QA"
cd "$QA"

python run_meshfleet_eval.py \
    --config meshfleet_benchmark/benchmark_config.yaml \
    --gt-folder "$GT_FOLDER" \
    --gen-folder "$GEN_FOLDER" \
    --metadata-file "$METADATA_CSV" \
    --output-folder "$METRICS_OUT" \
    --per-object --model-name "$VARIANT" \
    --skip-all-preprocessing

echo "Stage 50 (P0 $VARIANT) complete: $METRICS_OUT"
