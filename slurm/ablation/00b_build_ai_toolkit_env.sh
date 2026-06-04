#!/bin/bash
# ===========================================================================
# 00b_build_ai_toolkit_env.sh  —  NOT an sbatch.  Run by hand once, on a node
# with network (login node; on HPC set the HTTP(S) proxy first — see below).
#
# Builds the `ai_toolkit` conda env for the FLUX LoRA finetune (stage 30,
# ai-toolkit/run.py). Verified working 2026-06-04 (full snapshot:
# ai_toolkit_freeze.txt). Driver 570.211 supports CUDA 13.1, so the cu126
# torch wheels ai-toolkit pins run fine.
#
# Per ai-toolkit/README: python >3.10, torch 2.6.0 installed from the cu126
# index FIRST, then `pip install -r requirements.txt` (which pulls
# diffusers@<pinned git commit>, transformers 4.49.0, torchao 0.9.0,
# optimum-quanto 0.2.4, bitsandbytes, peft, lycoris-lora, etc.).
#
# ONE fix on top of requirements.txt: requirements lists the GUI build
# `opencv-python`, which needs libGL.so.1 (absent in a pip-only env) and whose
# cv2 import failure cascades into albumentations / controlnet_aux / transformers.
# We replace it with `opencv-python-headless` (server-correct; no libGL needed).
#
# NETWORK NOTE (HPC): export HTTP_PROXY / HTTPS_PROXY per
# ai-toolkit/start_finetune.slurm before running. Direct connection works on the
# local cluster.
#
# Run:  bash slurm/ablation/00b_build_ai_toolkit_env.sh
# ===========================================================================
set -euo pipefail

TARGET_ENV="${1:-ai_toolkit}"
AITK="${AITK:-/home/damian/Projects/ai-toolkit}"

export PATH=/home/damian/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"

echo "=== build conda env '$TARGET_ENV' (python 3.11, pure pip) ==="
if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
    echo "env '$TARGET_ENV' already exists — re-running installs (idempotent)."
else
    conda create -y -n "$TARGET_ENV" python=3.11
fi

set +u; conda activate "$TARGET_ENV"; set -u
echo "active: $CONDA_DEFAULT_ENV | python: $(python --version 2>&1)"

echo "=== 1. torch 2.6.0 / torchvision 0.21.0 (cu126) FIRST ==="
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126

echo "=== 2. ai-toolkit requirements ==="
pip install -r "$AITK/requirements.txt"

echo "=== 3. fix: replace GUI opencv with headless (no libGL) ==="
pip uninstall -y opencv-python opencv-python-headless || true
pip install --no-cache-dir "opencv-python-headless==4.11.0.86"

echo "=== verify ==="
python - <<'PY'
import importlib
mods = ["torch","torchvision","torchao","diffusers","transformers","accelerate",
        "peft","bitsandbytes","optimum.quanto","lycoris","cv2","albumentations",
        "controlnet_aux","safetensors","timm","open_clip"]
for m in mods:
    mod = importlib.import_module(m); print(f"  OK  {m:16s} {getattr(mod,'__version__','?')}")
import torch
print("torch.cuda.is_available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

# ai-toolkit job machinery + our config must load (no training started here).
cd "$AITK"
python - <<'PY'
import sys; sys.argv = ["run.py"]
from toolkit.job import get_job          # imports the trainer stack
import yaml
d = yaml.safe_load(open("config/flux_trellis_carcaption3k_1620.yaml"))
p = d["config"]["process"][0]
print("CONFIG OK:", d["config"]["name"], "| steps", p["train"]["steps"], "| model", p["model"]["name_or_path"])
print("ai_toolkit env READY for: python run.py config/flux_trellis_carcaption3k_1620.yaml")
PY
echo "Done. NOTE: the first run downloads FLUX.1-schnell (~24GB, Apache-2.0, no token)"
echo "and ostris/FLUX.1-schnell-training-adapter to the HF cache."
