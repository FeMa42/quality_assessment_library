# shellcheck shell=bash
# ---------------------------------------------------------------------------
# env.sh — shared environment for the CarCaption3K-1620 ablation SLURM DAG.
# Sourced by every sbatch in this directory.  Do NOT submit jobs without first
# CONFIRMING the conda env names below — several envs on this cluster are
# either drifted or do not exist (see README.md "Env caveats").
# ---------------------------------------------------------------------------

# --- Conda envs — CONFIRM THESE before submitting. ------------------------
# Data pipeline (render / voxelize / extract_feature / encode_*).  trellis_local
# is currently DRIFTED (numpy 2.x + open3d 0.17 segfaults voxelize; wrong utils3d
# lacks .io/.torch).  It MUST be aligned to TRELLIS/requirements_qa.txt
# (numpy 1.26.4 / open3d 0.19.0 / utils3d 0.0.2) via ./00_fix_trellis_env.sh
# before any data-prep job is submitted.
export ENV_TRELLIS_PREP="${ENV_TRELLIS_PREP:-trellis_local}"

# train.py finetuning.  Same dependency constraints as the data pipeline.
export ENV_TRELLIS_TRAIN="${ENV_TRELLIS_TRAIN:-trellis_local}"

# Generation (evaluate_trellis_prompt_following.py): needs trellis + diffusers +
# ImageReward + nvdiffrast.  CONFIRM nvdiffrast (0.3.3) is installed in this env —
# trellis_local currently lacks it (see 00_fix_trellis_env.sh, which installs it).
export ENV_GEN="${ENV_GEN:-trellis_local}"

# FLUX LoRA training.  NOTE: there is NO `ai_toolkit` env on this cluster (verified
# via `conda env list`: trellis_local, trellis2, trellis2_v2, trellis_qa,
# trellis_printability, hunyuan3d_local, prusa_libs).  Either create an
# `ai_toolkit` env here, or run the FLUX finetune on the HPC cluster per
# ai-toolkit/start_finetune.slurm (module load + proxy).  See 30_flux_finetune.sbatch.
export ENV_FLUX="${ENV_FLUX:-ai_toolkit}"

# QA evaluation (run_meshfleet_eval.py).
export ENV_QA="${ENV_QA:-trellis_qa}"

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
