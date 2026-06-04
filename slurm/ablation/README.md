# CarCaption3K-1620 Controlled Ablation — SLURM DAG

Queue-able set of sbatch scripts that run the entire CarCaption3K-1620 ablation
end-to-end: TRELLIS data prep → FLUX LoRA finetune → 4 TRELLIS finetunes →
generation + assembly + eval for the three variants (P0/P1/P2) → final table.

All scripts source [`env.sh`](./env.sh) and wrap the exact commands documented in
`$QA/runs/trellis_prep/COMMANDS.md`, `$QA/runs/trellis_train/COMMANDS.md`, and
`$QA/runs/eval/COMMANDS.md`. Budgets come from `$QA/runs/reference_budgets.json`.

## How to run (one paragraph)

1. Build the one working conda env: `bash 00_fix_trellis_env.sh` (clones the
   known-good `trellis_printability` → `trellis1_cc1620` and applies verified
   fixes: numpy 1.26.4, open3d 0.17.0, opencv-headless 4.10.0.84, diffusers 0.31.0,
   lpips, ImageReward import patch). Already created here & verified 2026-06-04.
2. Handle the FLUX *LoRA-training* env gap (stage 30, see caveats) and confirm the
   env vars in `env.sh`.
3. Submit the whole DAG: `bash submit_all.sh`. It prints the job IDs + the
   dependency graph. Re-submit any 100k-step finetune if it hits the 72h wall
   (it resumes via `--ckpt latest`).

## Environment variables (`env.sh`)

| Var | Default | Notes |
|-----|---------|-------|
| `ENV_TRELLIS_PREP` | `trellis1_cc1620` | data pipeline (built by `00_fix_trellis_env.sh`) |
| `ENV_TRELLIS_TRAIN` | `trellis1_cc1620` | `train.py` finetuning |
| `ENV_GEN` | `trellis1_cc1620` | generation; trellis + diffusers 0.31 + ImageReward + nvdiffrast (all verified) |
| `ENV_FLUX` | `ai_toolkit` | FLUX LoRA **training** only; **NO such env on this cluster — create it or run on HPC** |
| `ENV_QA` | `trellis1_cc1620` | `run_meshfleet_eval.py` (working ImageReward + lpips) |
| `QA` / `TRELLIS` / `AITK` | project roots | |
| `FLUX_LORA_REPO` | `DamianBoborzi/flux_carcaption3k_1620_lora32` | **confirm post-training** |
| `FLUX_LORA_WEIGHT` | `flux_carcaption3k_1620_lora32.safetensors` | **confirm post-training** |
| `GT_FOLDER` | `$QA/.../meshfleet_eval_images` | held-out eval set — DO NOT CHANGE |
| `METADATA_CSV` | `$QA/data/meshfleet/meshfleet_test.csv` | held-out metadata — DO NOT CHANGE |

All vars use `${VAR:-default}` so you can override from the environment without
editing the file.

## Env caveats (READ before submitting)

- **Data-prep / training / generation / eval env is RESOLVED.** `trellis1_cc1620`
  (built & verified 2026-06-04 by `00_fix_trellis_env.sh`) runs all of them:
  voxelize (open3d 0.17 + numpy 1.26.4), TRELLIS train + pipelines, FLUX+TRELLIS
  generation (diffusers 0.31.0), and eval (ImageReward import-patched + lpips).
  `trellis_local` was abandoned (drifted). Full version snapshot:
  `trellis1_cc1620_freeze.txt`. **Why these pins:** numpy 1.26.4 (open3d voxelize
  segfaults on numpy 2.x); open3d 0.17 (0.19 needs a missing `libGL.so.1`);
  diffusers 0.31 (0.38 registers flash-attn-3 as a torch custom op that torch
  2.4's `infer_schema` rejects, breaking FluxPipeline AND ImageReward); the
  ImageReward `med.py` patch (transformers 4.57 moved `apply_chunking_to_forward`
  to `transformers.pytorch_utils`).
- **ONE remaining gap — the FLUX *LoRA finetune* (stage 30).** There is no
  `ai_toolkit` conda env on this cluster (conda env list: `trellis_local,
  trellis2, trellis2_v2, trellis_qa, trellis_printability, hunyuan3d_local,
  prusa_libs, trellis1_cc1620`). Create an `ai_toolkit` env **or** run the FLUX
  finetune on the HPC per `ai-toolkit/start_finetune.slurm` (module load + proxy).
  Note: FLUX *generation* (stage 50, via diffusers FluxPipeline) works in
  `trellis1_cc1620`; only the LoRA *training* needs the ai-toolkit env.

## Submit order + DAG

```
data prep:
  10 train_render  (gpu:2, standalone)   render 1420 obj × 150v, sharded 2 GPUs
  11 train_geom    (gpu:1, afterok:10)   voxelize→feat→ss_latent→latent
  12 train_cond    (gpu:2, afterok:10)   render_cond 1420 obj × 24v, sharded
  20 val           (gpu:2, standalone)   full pipeline, 200 obj
  DATAPREP_DONE = afterok:11:12:20

30 flux            (gpu:1, standalone)   FLUX LoRA, 8000 steps

finetunes (each gpu:1, --num_gpus 1, afterok DATAPREP_DONE):
  40 ft_ss_txt     100k steps
  41 ft_slat_txt   100k steps
  42 ft_ss_img     100k steps
  43 ft_slat_img    30k steps

gen+eval (each gpu:1):
  50 p0 FLUX_TRELLIS_CC1620   afterok:30
  51 p1 TRELLIS_TXT_CC1620    afterok:40:41
  52 p2 TRELLIS_IMG_CC1620    afterok:42:43:30

60 table  (no GPU)  afterok:50:51:52
```

```
10 ─┬─> 11 ─┐
    └─> 12 ─┼─> {40,41,42,43} ─┬─> 51 (40,41) ┐
20 ───────┘                    └─> 52 (42,43,30) ┤
30 ──────────────────────────────> 50 (30) ─────┼─> 60
                                                 ┘
```

## Walltime / sharding rationale

- **render ≈ 4.5 min/obj × 1420 ≈ 106 GPU-h on 1 GPU — over the 72h wall.** So
  `render` (stage 10) and `render_cond` (stage 12) request `--gres=gpu:2` and run
  rank0 on GPU0 + rank1 on GPU1 with `--world_size 2`, then `build_metadata.py`
  merges the per-rank CSVs. The geometry chain (stage 11) is cheap → 1 GPU.
- **TRELLIS finetunes use 1 GPU / `--num_gpus 1`** to match the MeshFleet runs'
  effective batch size (the reference runs used the train.py default on a single
  effective batch). Do not bump to 2 GPUs without re-tuning LR/batch.
- **100k-step finetunes may exceed 72h.** `train.py` defaults to `--ckpt latest`,
  so simply **re-submit the same script** to resume from the latest checkpoint in
  its `output_dir`. (`submit_all.sh` wires the first submission; resume is manual
  or via a follow-up `sbatch` of the same file.)

## Open items

- **OI-1 (FLUX LoRA pin) — handled via env vars.** The generation script now takes
  `--flux_lora_repo` / `--flux_lora_weight_name`; `env.sh` passes
  `FLUX_LORA_REPO` / `FLUX_LORA_WEIGHT`. **Confirm the actual repo/weight after
  stage 30 finishes** and update `env.sh` if it differs from the default.
- **OI-2 (ss_flow checkpoint) — handled.** The script now accepts
  `--ss_flow_checkpoint_path` (and loads `.pt`), so P1/P2 pass both finetuned
  stages directly. The gen scripts glob the highest-step `denoiser_step*.pt`.
- **OI-3 (assembly) — RESOLVED.** The generation script already keys outputs by
  sha (`sample_<sha>.glb`, `glb_<sha>/`), so there is no `i→sha` mapping to do.
  Assembly now stages `sample_<sha>.glb → <sha>/<sha>.glb` (`stage_generated_glbs.py`)
  and renders at the GT camera with the canonical `render_for_quality_assessment.py`
  (the same renderer used for every reference method: az 0–330 step 30, elevation 90,
  radius 1.5) → `data/ablation/gen/<VARIANT>/<sha>/000.png..011.png`. The fragile
  `glb_<sha>/renders` copy is no longer used. The `--prompt_file` is the committed
  `manifests/meshfleet_heldout_232_prompts.csv` (exactly the 232 held-out objects,
  sha-sorted, built by `scripts/build_heldout_prompt_file.py`).
- **EMA vs raw checkpoint.** The gen scripts use the raw `denoiser_step*.pt`. To
  use EMA weights, switch the glob to your trainer's EMA filename
  (e.g. `ema_step*.pt`). Confirm which is preferred.
- **FLUX env / dataset.** Stage 30 needs an `ai_toolkit` env (missing here) and
  the gitignored FLUX dataset at
  `$QA/data/ablation/carcaption3k_1620_flux_train`, rebuildable via
  `$QA/scripts/build_flux_filtered_dataset.py`.

## Files

| File | Purpose |
|------|---------|
| `env.sh` | shared env vars + `activate()` helper (sourced by all) |
| `00_fix_trellis_env.sh` | build `trellis1_cc1620` (clone `trellis_printability` + verified fixes); run by hand once |
| `trellis1_cc1620_freeze.txt` | `pip freeze` snapshot of the verified working env |
| `10_dataprep_train_render.sbatch` | train metadata + render (sharded 2 GPU) |
| `11_dataprep_train_geom.sbatch` | voxelize → feature → ss_latent → latent |
| `12_dataprep_train_cond.sbatch` | render_cond (sharded 2 GPU) |
| `20_dataprep_val.sbatch` | full val pipeline (200 obj) |
| `30_flux_finetune.sbatch` | FLUX LoRA finetune |
| `40_ft_ss_txt.sbatch` … `43_ft_slat_img.sbatch` | 4 TRELLIS finetunes |
| `50_gen_eval_p0.sbatch` … `52_gen_eval_p2.sbatch` | gen → assemble → eval per variant |
| `60_assemble_table.sbatch` | final comparison table |
| `submit_all.sh` | submit the whole DAG with dependencies |
