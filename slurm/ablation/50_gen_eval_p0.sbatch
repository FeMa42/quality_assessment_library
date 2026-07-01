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
# OI-3 RESOLVED: generation outputs are sha-keyed (sample_<sha>.glb). Assembly
# stages those GLBs to <sha>/<sha>.glb and renders them at the GT camera with the
# canonical render_for_quality_assessment.py (same renderer as all reference
# methods: az 0..330 step30, elev 90, radius 1.5) -> <sha>/000.png..011.png. No
# more glb_<sha>/renders copy/rename. The held-out 232-object prompt file is built
# by scripts/build_heldout_prompt_file.py (commit / regenerate as needed).
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

VARIANT="FLUX_TRELLIS_CC1620"
RAW_OUT="$TRELLIS/eval_prompt_alignment/output_flux_cc1620"
GEN_FOLDER="$QA/data/ablation/gen/$VARIANT"
METRICS_OUT="$QA/runs/$VARIANT/metrics"
PROMPT_FILE="$QA/manifests/meshfleet_heldout_232_prompts.csv"

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
    --seed 3

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

echo "Stage 50 (P0 $VARIANT) complete: $METRICS_OUT"
