#!/bin/bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_FLUX"
cd "$AITK"

# Ensure the (gitignored) FLUX training dataset exists before training.
# Uncomment to (re)build it on the fly under the QA env:
# "/home/damian/miniconda3/envs/$ENV_QA/bin/python" "$QA/scripts/build_flux_filtered_dataset.py"

python run.py config/flux_trellis_carcaption3k_1620.yaml

echo "Stage 30 (FLUX LoRA finetune) complete."
echo "CONFIRM the produced LoRA repo/weight and set FLUX_LORA_REPO/FLUX_LORA_WEIGHT in env.sh:"
echo "  FLUX_LORA_REPO  = $FLUX_LORA_REPO"
echo "  FLUX_LORA_WEIGHT= $FLUX_LORA_WEIGHT"
