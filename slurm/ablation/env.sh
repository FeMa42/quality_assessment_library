# shellcheck shell=bash
# ---------------------------------------------------------------------------
# env.sh — shared environment for the CarCaption3K-1620 ablation SLURM DAG.
# Sourced by every sbatch in this directory.  Do NOT submit jobs without first
# CONFIRMING the conda env names below — several envs on this cluster are
# either drifted or do not exist (see README.md "Env caveats").
# ---------------------------------------------------------------------------

# --- Conda envs ----------------------------------------------------------
# ONE env runs data-prep + training + generation + eval: `trellis1_cc1620`,
# built by ./00_fix_trellis_env.sh (clone of the known-good `trellis_printability`
# TRELLIS-1 stack + verified fixes: numpy 1.26.4, open3d 0.17.0,
# opencv-python-headless 4.10.0.84, diffusers 0.31.0, lpips, + an ImageReward
# import patch). Verified 2026-06-04 — voxelize, trellis pipelines, FLUX,
# ImageReward, lpips all work; full snapshot in trellis1_cc1620_freeze.txt.
# RUN 00_fix_trellis_env.sh ONCE before submitting if the env does not exist yet.
export ENV_TRELLIS_PREP="${ENV_TRELLIS_PREP:-trellis1_cc1620}"   # render/voxelize/feature/encode
export ENV_TRELLIS_TRAIN="${ENV_TRELLIS_TRAIN:-trellis1_cc1620}" # train.py finetuning
export ENV_GEN="${ENV_GEN:-trellis1_cc1620}"                     # evaluate_trellis_prompt_following.py (FLUX+TRELLIS)
export ENV_QA="${ENV_QA:-trellis1_cc1620}"                       # run_meshfleet_eval.py (working ImageReward + lpips)

# FLUX LoRA *training* (stage 30, ai-toolkit run.py) is the ONE remaining env gap:
# there is NO `ai_toolkit` conda env on this cluster (conda env list: trellis_local,
# trellis2, trellis2_v2, trellis_qa, trellis_printability, hunyuan3d_local,
# prusa_libs, trellis1_cc1620). Create an `ai_toolkit` env, or run the FLUX finetune
# on the HPC per ai-toolkit/start_finetune.slurm (module load + proxy).
export ENV_FLUX="${ENV_FLUX:-ai_toolkit}"

# --- Project roots --------------------------------------------------------
export QA="${QA:-/home/damian/Projects/quality_assessment_library}"
export TRELLIS="${TRELLIS:-/home/damian/Projects/TRELLIS}"
export AITK="${AITK:-/home/damian/Projects/ai-toolkit}"

# --- FLUX LoRA produced by 30_flux_finetune (CONFIRM repo/weight after training).
# Defaults match ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml
# (hf_repo_id: DamianBoborzi/flux_carcaption3k_1620_lora32).  If you keep the LoRA
# local instead of pushing to HF, set FLUX_LORA_REPO to the local output dir, e.g.
#   export FLUX_LORA_REPO="$AITK/output/flux_carcaption3k_1620_lora32"
export FLUX_LORA_REPO="${FLUX_LORA_REPO:-DamianBoborzi/flux_carcaption3k_1620_lora32}"
export FLUX_LORA_WEIGHT="${FLUX_LORA_WEIGHT:-flux_carcaption3k_1620_lora32.safetensors}"

# --- Held-out eval set (DO NOT CHANGE) ------------------------------------
export GT_FOLDER="${GT_FOLDER:-$QA/data/meshfleet/benchmark_data/meshfleet_eval_images}"
export METADATA_CSV="${METADATA_CSV:-$QA/data/meshfleet/meshfleet_test.csv}"

# --- conda activation helper ----------------------------------------------
# Usage: activate "$ENV_TRELLIS_PREP"
activate () {
    # Activate a conda env robustly under `set -euo pipefail`.
    # conda's activate.d hooks (e.g. libblas_mkl_activate.sh does
    # `export CONDA_MKL_INTERFACE_LAYER_BACKUP=${MKL_INTERFACE_LAYER}`) reference
    # UNSET vars, which abort under `set -u`. So disable nounset around the conda
    # calls and restore the caller's previous setting afterward. Also `conda
    # deactivate` first so switching envs mid-script (generation -> eval) is clean.
    export PATH=/home/damian/miniconda3/bin:$PATH
    local _had_u=0; case $- in *u*) _had_u=1;; esac
    set +u
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
    conda activate "$1"
    if [ "$_had_u" = 1 ]; then set -u; fi
}
