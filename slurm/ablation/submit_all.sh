#!/bin/bash
# ===========================================================================
# submit_all.sh — submit the full CarCaption3K-1620 ablation SLURM DAG.
#
# Dependency graph (afterok = run only if the dependency completes successfully):
#
#   data prep:
#     10 train_render        (standalone)
#     11 train_geom          afterok:10
#     12 train_cond          afterok:10
#     20 val                 (standalone)
#   DATAPREP_DONE = afterok:11:12:20
#
#   30 flux                  (standalone)
#
#   finetunes:
#     40 ft_ss_txt           afterok DATAPREP_DONE
#     41 ft_slat_txt         afterok DATAPREP_DONE
#     42 ft_ss_img           afterok DATAPREP_DONE
#     43 ft_slat_img         afterok DATAPREP_DONE
#
#   gen+eval:
#     50 p0                  afterok:30
#     51 p1                  afterok:40:41
#     52 p2                  afterok:42:43:30
#
#   60 table                 afterok:50:51:52
#
# BEFORE RUNNING: confirm the conda envs in env.sh (run ./00_fix_trellis_env.sh
# first; create/handle the ai_toolkit FLUX env per 30_flux_finetune.sbatch).
# This script ONLY submits — it does not validate envs.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/logs"

submit () {
    # submit <script> [extra sbatch args...]; prints + returns the job id.
    local script="$1"; shift
    local jid
    jid="$(sbatch --parsable "$@" "$HERE/$script")"
    echo "  submitted $script -> job $jid (deps: $*)" >&2
    echo "$jid"
}

echo "Submitting CarCaption3K-1620 ablation DAG..." >&2

# --- data prep -----------------------------------------------------------
J10="$(submit 10_dataprep_train_render.sbatch)"
J11="$(submit 11_dataprep_train_geom.sbatch  --dependency=afterok:"$J10")"
J12="$(submit 12_dataprep_train_cond.sbatch  --dependency=afterok:"$J10")"
J20="$(submit 20_dataprep_val.sbatch)"

DATAPREP_DONE="afterok:${J11}:${J12}:${J20}"

# --- FLUX finetune (standalone) ------------------------------------------
J30="$(submit 30_flux_finetune.sbatch)"

# --- TRELLIS finetunes (all gated on data prep) --------------------------
J40="$(submit 40_ft_ss_txt.sbatch    --dependency="$DATAPREP_DONE")"
J41="$(submit 41_ft_slat_txt.sbatch  --dependency="$DATAPREP_DONE")"
J42="$(submit 42_ft_ss_img.sbatch    --dependency="$DATAPREP_DONE")"
J43="$(submit 43_ft_slat_img.sbatch  --dependency="$DATAPREP_DONE")"

# --- generation + eval ---------------------------------------------------
J50="$(submit 50_gen_eval_p0.sbatch  --dependency=afterok:"$J30")"
J51="$(submit 51_gen_eval_p1.sbatch  --dependency=afterok:"${J40}:${J41}")"
J52="$(submit 52_gen_eval_p2.sbatch  --dependency=afterok:"${J42}:${J43}:${J30}")"

# --- final table ---------------------------------------------------------
J60="$(submit 60_assemble_table.sbatch --dependency=afterok:"${J50}:${J51}:${J52}")"

cat >&2 <<EOF

===========================================================================
Submitted job IDs:
  10 train_render : $J10
  11 train_geom   : $J11   (afterok:$J10)
  12 train_cond   : $J12   (afterok:$J10)
  20 val          : $J20
  30 flux         : $J30
  40 ft_ss_txt    : $J40   ($DATAPREP_DONE)
  41 ft_slat_txt  : $J41   ($DATAPREP_DONE)
  42 ft_ss_img    : $J42   ($DATAPREP_DONE)
  43 ft_slat_img  : $J43   ($DATAPREP_DONE)
  50 p0           : $J50   (afterok:$J30)
  51 p1           : $J51   (afterok:$J40:$J41)
  52 p2           : $J52   (afterok:$J42:$J43:$J30)
  60 table        : $J60   (afterok:$J50:$J51:$J52)

DAG:
  10 ─┬─> 11 ─┐
      └─> 12 ─┼─> {40,41,42,43} ─┬─> 51 (40,41) ┐
  20 ───────┘                    └─> 52 (42,43,30) ┤
  30 ──────────────────────────────> 50 (30) ─────┼─> 60
                                                   ┘
===========================================================================
EOF
