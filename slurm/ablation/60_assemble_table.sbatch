#!/bin/bash
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cc1620_60_table
#SBATCH --output=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.out
#SBATCH --error=/home/damian/Projects/quality_assessment_library/slurm/ablation/logs/%x-%j.err
# ---------------------------------------------------------------------------
# Stage 60 — assemble the final ablation comparison table (CPU-only, no GPU).
# Merges the reference per-object metrics with the three CC1620 variant metrics.
# DEPENDS ON stages 50, 51, 52.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

activate "$ENV_QA"
cd "$QA"

# The second glob expands to the per-variant per_object_metrics_*.csv files
# written by run_meshfleet_eval.py --per-object (one per P0/P1/P2 metrics dir).
python "$QA/scripts/assemble_ablation_table.py" \
    --inputs \
        "$QA/analysis/track_b/out/per_object_metrics_all.csv" \
        "$QA"/runs/*/metrics/per_object_metrics_*.csv

echo "Stage 60 (assemble table) complete."
