#!/bin/bash
# ===========================================================================
# 00_fix_trellis_env.sh  —  NOT an sbatch.  Run by hand once, on a node with
# network (a login node; on HPC set the HTTP(S) proxy first, see below).
#
# Builds the ONE conda env that runs the ENTIRE CarCaption3K-1620 pipeline
# (data prep + TRELLIS training + FLUX/TRELLIS generation + QA eval) by cloning
# the known-good `trellis_printability` (which already has the painful-to-build
# TRELLIS-1 stack: torch 2.4.0, nvdiffrast 0.3.3, spconv 2.3.6,
# flash_attn 2.7.0.post2, xformers, diff_gaussian_rasterization, the correct
# utils3d with .io/.torch) and applying small, VERIFIED fixes.
#
# All fixes verified working 2026-06-04 in env `trellis1_cc1620`
# (full snapshot: trellis1_cc1620_freeze.txt):
#   1. numpy 2.2.6 -> 1.26.4        : open3d 0.17 VoxelGrid voxelize SEGFAULTS
#                                     under numpy 2.x; numpy 1.26.4 fixes it.
#   2. open3d pinned 0.17.0         : keep 0.17 (imports via the env's libX11).
#                                     Do NOT use 0.19 — it needs libGL.so.1 which
#                                     is not present in the env.
#   3. opencv-python-headless 4.10.0.84 : the drifted 4.13 requires numpy>=2;
#                                     4.10.0.84 (the TRELLIS-1 reference pin) is
#                                     numpy-1.26 compatible.
#   4. diffusers 0.38.0 -> 0.31.0   : 0.38 registers flash-attn-3 as a torch
#                                     custom op whose `float | None` annotations
#                                     torch 2.4's infer_schema rejects -> BOTH
#                                     diffusers FluxPipeline AND ImageReward (via
#                                     ReFL.py `from diffusers import ...`) fail to
#                                     import. 0.31.0 still has FluxPipeline and
#                                     predates that registration.
#   5. + lpips                      : eval dependency, not preinstalled.
#   6. ImageReward source patch     : transformers 4.57 MOVED
#                                     apply_chunking_to_forward (+
#                                     find_pruneable_heads_and_indices,
#                                     prune_linear_layer) from
#                                     transformers.modeling_utils to
#                                     transformers.pytorch_utils. Patch
#                                     ImageReward/models/BLIP/med.py accordingly
#                                     (NO transformers downgrade -> FLUX + IR coexist).
#
# NETWORK NOTE (HPC): pip needs outbound HTTPS. On HPC use the proxy from
# ai-toolkit/start_finetune.slurm:
#   export HTTP_PROXY="http://imech:<password>@137.250.175.192:3128"; export HTTPS_PROXY="$HTTP_PROXY"
# On this local cluster a direct connection already works.
#
# Run:  bash slurm/ablation/00_fix_trellis_env.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

SRC_ENV="trellis_printability"          # known-good TRELLIS-1 base (do NOT mutate)
TARGET_ENV="${1:-$ENV_TRELLIS_PREP}"    # default: trellis1_cc1620 (see env.sh)

export PATH=/home/damian/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"

echo "==========================================================================="
echo " Building env '$TARGET_ENV' by cloning '$SRC_ENV' + applying verified fixes"
echo "==========================================================================="

# --- 1. clone (skip if it already exists) ---------------------------------
if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
    echo "env '$TARGET_ENV' already exists — skipping clone, re-applying fixes (idempotent)."
else
    echo "cloning $SRC_ENV -> $TARGET_ENV (a few minutes; hardlinks the pkg cache)..."
    conda create -y --name "$TARGET_ENV" --clone "$SRC_ENV"
fi

set +u; conda activate "$TARGET_ENV"; set -u
echo "active: $CONDA_DEFAULT_ENV | python: $(which python)"

# --- 2-5. pip pins (verified working set) ---------------------------------
# numpy first so open3d/opencv resolve against numpy 1.26.
pip install "numpy==1.26.4"
pip install "open3d==0.17.0" "opencv-python-headless==4.10.0.84" "diffusers==0.31.0" "lpips==0.1.4"

# --- 6. ImageReward import patch (idempotent) -----------------------------
python - <<'PY'
import os, importlib.util
spec = importlib.util.find_spec("ImageReward")
med = os.path.join(os.path.dirname(spec.origin), "models", "BLIP", "med.py")
s = open(med).read()
old = ("from transformers.modeling_utils import (\n"
       "    PreTrainedModel,\n"
       "    apply_chunking_to_forward,\n"
       "    find_pruneable_heads_and_indices,\n"
       "    prune_linear_layer,\n"
       ")")
new = ("from transformers.modeling_utils import (\n"
       "    PreTrainedModel,\n"
       ")\n"
       "from transformers.pytorch_utils import (\n"
       "    apply_chunking_to_forward,\n"
       "    find_pruneable_heads_and_indices,\n"
       "    prune_linear_layer,\n"
       ")")
if old in s:
    open(med, "w").write(s.replace(old, new)); print("patched ImageReward med.py:", med)
elif "from transformers.pytorch_utils import" in s:
    print("ImageReward med.py already patched.")
else:
    raise SystemExit("WARNING: ImageReward med.py import block not in the expected form — patch by hand.")
PY

# --- verify -----------------------------------------------------------------
echo "=== verifying $TARGET_ENV ==="
python - <<'PY'
import importlib
def chk(n, fn):
    try: fn(); print("  OK  ", n)
    except Exception as e: print("  FAIL", n, "->", type(e).__name__, str(e)[:80]); raise
def vox():
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_box()
    o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        m, voxel_size=1/64, min_bound=(-.5,)*3, max_bound=(.5,)*3)
chk("voxelize (open3d+numpy)", vox)
chk("trellis.pipelines",       lambda: importlib.import_module("trellis.pipelines"))
chk("diffusers.FluxPipeline",  lambda: __import__("diffusers").FluxPipeline)
chk("nvdiffrast.torch",        lambda: importlib.import_module("nvdiffrast.torch"))
chk("ImageReward import",      lambda: importlib.import_module("ImageReward"))
chk("lpips",                   lambda: importlib.import_module("lpips"))
chk("utils3d.io/.torch",       lambda: (importlib.import_module("utils3d.io"), importlib.import_module("utils3d.torch")))
print("ALL CHECKS PASSED for", __import__("os").environ.get("CONDA_DEFAULT_ENV"))
PY
echo "Done. '$TARGET_ENV' runs data-prep + training + generation + eval."
echo "NOTE: the FLUX *LoRA finetune* (stage 30, ai-toolkit run.py) uses a SEPARATE"
echo "ai-toolkit env (env.sh ENV_FLUX / 30_flux_finetune.sbatch) — not this one."
