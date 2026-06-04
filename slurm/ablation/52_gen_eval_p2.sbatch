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
#SBATCH --job-name=cc1620_52_gen_eval_p2
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 52 — P2 · TRELLIS_IMG_CC1620 : generate → assemble → eval.
# FLUX image-to-3D (--use_flux + CarCaption3K-1620 FLUX LoRA) with the finetuned
# ss_img + slat_img denoisers patched in.
# DEPENDS ON stages 42, 43 (img finetunes) AND 30 (FLUX image source).
# Based on $QA/runs/eval/COMMANDS.md (P2).
#
# NOTE: The FLUX LoRA used here should be held CONSISTENT with the MeshFleet
# TRELLIS_FT comparison so P2 vs the MeshFleet reference is apples-to-apples
# (same FLUX_LORA_REPO/FLUX_LORA_WEIGHT as P0).
#
# Checkpoint selection: highest-step denoiser_step*.pt per finetune ckpts/ dir
# (switch glob to ema_step*.pt for EMA weights — CONFIRM preference).
#
# OI-3 RESOLVED (same as P0/P1): generation outputs are sha-keyed
# (sample_<sha>.glb). Assembly stages them to <sha>/<sha>.glb and renders at the
# GT camera with the canonical render_for_quality_assessment.py ->
# <sha>/000.png..011.png. The held-out 232-object prompt file is built by
# scripts/build_heldout_prompt_file.py.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

VARIANT="TRELLIS_IMG_CC1620"
RAW_OUT="$TRELLIS/eval_prompt_alignment/output_trellis_img_cc1620"
GEN_FOLDER="$QA/data/ablation/gen/$VARIANT"
METRICS_OUT="$QA/runs/$VARIANT/metrics"
PROMPT_FILE="$QA/manifests/meshfleet_heldout_232_prompts.csv"

SS_CKPT_DIR="$TRELLIS/outputs/carcaption3k_1620/ss_flow_img_dit_L_16l8_fp16/ckpts"
SLAT_CKPT_DIR="$TRELLIS/outputs/carcaption3k_1620/slat_flow_img_dit_L_64l8p2_fp16/ckpts"

SS_CKPT="$(ls "$SS_CKPT_DIR"/denoiser_step*.pt 2>/dev/null | sort | tail -n 1)"
SLAT_CKPT="$(ls "$SLAT_CKPT_DIR"/denoiser_step*.pt 2>/dev/null | sort | tail -n 1)"
if [ -z "$SS_CKPT" ] || [ -z "$SLAT_CKPT" ]; then
    echo "ERROR: missing finetuned checkpoints." >&2
    echo "  ss:   $SS_CKPT_DIR/denoiser_step*.pt -> '$SS_CKPT'" >&2
    echo "  slat: $SLAT_CKPT_DIR/denoiser_step*.pt -> '$SLAT_CKPT'" >&2
    exit 1
fi
echo "Using ss_img ckpt:   $SS_CKPT"
echo "Using slat_img ckpt: $SLAT_CKPT"

# ---- (1) Generation (ENV_GEN) -------------------------------------------
activate "$ENV_GEN"
cd "$TRELLIS"

python evaluate_trellis_prompt_following.py \
    --output_dir "$RAW_OUT" \
    --use_flux \
    --sample_max_reward_image \
    --flux_lora_repo "$FLUX_LORA_REPO" \
    --flux_lora_weight_name "$FLUX_LORA_WEIGHT" \
    --prompt_file "$PROMPT_FILE" \
    --seed 3 \
    --ss_flow_checkpoint_path "$SS_CKPT" \
    --slat_flow_checkpoint_path "$SLAT_CKPT"

# ---- (2) Assembly: stage sample_<sha>.glb -> <sha>/<sha>.glb, then render at
#         the GT camera to <sha>/000.png..011.png (canonical, OI-3 RESOLVED) ----
# stage_generated_glbs.py is pure-python (fine in any env, incl. ENV_GEN);
# render_for_quality_assessment.py needs the generation/blender env (ENV_GEN).
python "$QA/scripts/stage_generated_glbs.py" \
    --gen-dir "$RAW_OUT" --staging-dir "$RAW_OUT/_staging"
cd "$TRELLIS" && python render_for_quality_assessment.py \
    --input_dir "$RAW_OUT/_staging" --mesh_output_dir "$GEN_FOLDER"
# -> $GEN_FOLDER/<sha>/000.png..011.png at the GT camera (az 0..330, elev 90, radius 1.5)

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

echo "Stage 52 (P2 $VARIANT) complete: $METRICS_OUT"
