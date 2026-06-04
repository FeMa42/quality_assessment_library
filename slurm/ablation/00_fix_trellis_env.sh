#!/bin/bash
# ===========================================================================
# 00_fix_trellis_env.sh  —  NOT an sbatch.  A documented helper.
#
# Aligns the data-prep / train / generation conda env to
# TRELLIS/requirements_qa.txt so that:
#   * voxelize.py does not SEGFAULT (open3d 0.17 + numpy 2.x crashes
#     VoxelGrid.create_from_triangle_mesh_within_bounds — needs numpy 1.x).
#   * extract_feature.py / encode_*.py find a utils3d that exposes .io / .torch
#     (the PyPI "utils3d" currently installed in trellis_local is an UNRELATED
#     package with only pctodepthimage.py).
#   * generation (evaluate_trellis_prompt_following.py) can import nvdiffrast.
#
# Verified pins from TRELLIS/requirements_qa.txt (2026-06-04):
#   numpy 1.26.4 | open3d 0.19.0 | utils3d 0.0.2 | nvdiffrast 0.3.3
# CONFIRM these against requirements_qa.txt before running — they may change.
#
# NETWORK NOTE: pip installs need outbound HTTPS. On the HPC this requires a
# login node + the HTTP(S) proxy used in ai-toolkit/start_finetune.slurm, e.g.
#   export HTTP_PROXY="http://imech:<password>@137.250.175.192:3128"
#   export HTTPS_PROXY="$HTTP_PROXY"
# On this local cluster a direct connection may already work — try without first.
#
# This script does NOT run automatically. Review, then run by hand:
#   bash slurm/ablation/00_fix_trellis_env.sh
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

TARGET_ENV="${1:-$ENV_TRELLIS_PREP}"
REQ="$TRELLIS/requirements_qa.txt"

echo "==========================================================================="
echo " Fixing conda env: $TARGET_ENV"
echo " Pins source:      $REQ"
echo "==========================================================================="
echo
echo ">> Review the pinned versions BEFORE proceeding:"
grep -iE 'numpy|open3d|utils3d|nvdiffrast' "$REQ" || true
echo
echo ">> If anything above differs from numpy 1.26.4 / open3d 0.19.0 /"
echo "   utils3d 0.0.2 / nvdiffrast 0.3.3, edit this script to match."
echo
read -r -p "Proceed to pip-install into '$TARGET_ENV'? [y/N] " ans
case "$ans" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "Aborted (no changes made)."; exit 0 ;;
esac

activate "$TARGET_ENV"

echo ">> Pinning numpy / open3d ..."
pip install --no-cache-dir "numpy==1.26.4" "open3d==0.19.0"

echo ">> Installing the correct utils3d (the one exposing .io / .torch) ..."
# The TRELLIS-compatible utils3d is pinned to 0.0.2 in requirements_qa.txt.
# If PyPI's 0.0.2 still resolves to the wrong package, install from the upstream
# git ref instead (uncomment the git line and comment the pip pin):
pip install --no-cache-dir "utils3d==0.0.2"
# pip install --no-cache-dir "git+https://github.com/EasternJournalist/utils3d.git"

echo ">> Installing nvdiffrast (needed for generation) ..."
pip install --no-cache-dir "nvdiffrast==0.3.3"

echo
echo ">> Sanity check (import utils3d.io / utils3d.torch, numpy, open3d):"
python - <<'PY'
import numpy, open3d
print("numpy", numpy.__version__)
print("open3d", open3d.__version__)
import utils3d
ok = hasattr(utils3d, "io") and hasattr(utils3d, "torch")
print("utils3d has .io/.torch:", ok)
try:
    import nvdiffrast
    print("nvdiffrast importable:", getattr(nvdiffrast, "__version__", "?"))
except Exception as e:
    print("nvdiffrast import FAILED:", e)
assert ok, "utils3d still lacks .io/.torch — install the correct package (see git line above)."
PY

echo
echo "Done. '$TARGET_ENV' should now satisfy requirements_qa.txt for data-prep,"
echo "training, and generation. Re-run this for ENV_GEN / ENV_TRELLIS_TRAIN if"
echo "they point at a different conda env."
