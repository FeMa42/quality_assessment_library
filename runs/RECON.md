# RECON — MeshFleet budget recovery + generation-pipeline location (Phase 0)

CarCaption3K-1620 controlled ablation. Read-only recon across all repos; only
`runs/reference_budgets.json` and this file were written. Date: 2026-06-03.

Repo roots:
- `QA = /home/damian/Projects/quality_assessment_library`
- `AITK = /home/damian/Projects/ai-toolkit`
- `TRELLIS = /home/damian/Projects/TRELLIS`
- `CCD = /home/damian/Projects/datasets/CarCaptionData`

## 1. Recovered TRELLIS per-family budgets

All four controller-suggested sweep dirs exist and ARE MeshFleet finetunes
(`finetune_ckpt` points at the pretrained TRELLIS txt/image checkpoints; data
defaults to `meshfleetxl_train`/`meshfleetxl_test`). Source of truth =
`<sweep_dir>/command.txt` + `<sweep_dir>/config.json` under `$TRELLIS`. Each
realized-step count was cross-checked against the highest `denoiser_step*.pt` in
`ckpts/` and the line count of `log.txt` (one JSON line per step).

| Family            | max_steps | realized ckpt          | base_config | finetune_ckpt (rel to $TRELLIS) | lr | sweep_dir |
| ----------------- | --------- | ---------------------- | ----------- | -------------------------------- | -- | --------- |
| trellis_slat_txt  | 100000    | denoiser_step0100000.pt | configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_finetune.json | assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432.../ckpts/slat_flow_txt_dit_XL_64l8p2_fp16.safetensors | 6.93e-05 | outputs/sweeps_slat_flow_txt/sweep_lcqjobfr_20250629_230408 |
| trellis_ss_txt    | 100000    | denoiser_step0100000.pt | configs/generation/ss_flow_txt_dit_XL_16l8_fp16_finetune.json | assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432.../ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors | 5.00e-05 | outputs/sweeps_ss_flow_txt/sweep_wqt27r19_20250704_034000 |
| trellis_ss_img    | 100000    | denoiser_step0100000.pt | configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune.json | assets/pretrained_TRELLIS_image_large/snapshots/25e0d31f.../ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors | 1.07e-05 | outputs/sweeps_ss_flow_img/sweep_x3va3sda_20250705_011646 |
| trellis_slat_img  | 30000     | denoiser_step0030000.pt | configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune.json | assets/pretrained_TRELLIS_image_large/snapshots/25e0d31f.../ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors | 2.39e-05 | outputs/sweeps_slat_flow_img/sweep_uoxlv9op_20250627_151924 |

These MATCH the controller's prior recon exactly:
slat_txt=100k, ss_txt=100k, ss_img=100k, slat_img=30k. No corrections needed.

All four `finetune_ckpt` paths were verified present on disk (`ls` EXISTS for
each). All four configs use `batch_size_per_gpu=2` and `min_aesthetic_score=4.5`.

`data_dir` / `eval_data_dir` / `num_gpus` were NOT passed on the command line in
any `command.txt`, so `train.py` defaults applied:
- `--data_dir` default = `./datasets/meshfleetxl_train`
- `--eval_data_dir` default = `./datasets/meshfleetxl_test`
- `--num_gpus` default = `-1` (= all GPUs on the training node; not pinned).
(Source: `$TRELLIS/train.py` lines 132-133, 141, 147.) Recorded as `null` in the
JSON with a note. The `command.txt` invokes a now-absent `sweep_train.py`
wrapper; the surviving `sweep_helper.py` shows the same `python train.py
--config ... --output_dir ... --finetune_ckpt ...` call shape.

## 2. Recovered FLUX budget

Source: `$AITK/config/train_lora_flux_schnell_32_3.yaml`.

- steps: **8000** (matches controller guess)
- lr: 1e-4
- LoRA: `linear=32`, `linear_alpha=32` (rank 32)
- batch_size: 4, gradient_accumulation_steps: 1
- optimizer: adamw8bit, noise_scheduler: flowmatch, linear_timesteps: true, EMA decay 0.99
- base model: `black-forest-labs/FLUX.1-schnell` (+ schnell training adapter), 8-bit quantize
- resolutions: [512, 768, 1024]
- train folder: `/hpc/gpfs2/scratch/u/boborzda/meshfleet_aitk_train` (HPC scratch, NOT on this machine)
- pushes to HF repo: `DamianBoborzi/flux_schnell_MeshFleet_newPrompts_lora32_6`

## 3. Object counts

| Source | Count |
| ------ | ----- |
| `meshfleetxl_train/metadata.csv` (minus header) | **2229** objects |
| `meshfleetxl_test/metadata.csv` (minus header)  | **394** objects |
| FLUX local train folder | unknown (HPC scratch path, not present locally) |
| CCD `CarCaptionData_ai_toolkit/*.txt` | 2633 caption files |
| CCD `car_meshes_trellis_aesthetics_65/` | 2648 dirs |
| Paper / curated CarCaption3K | 1620 |

## 4. GATE A — object-count conflict (RESOLVED by controller)

- Conflict: paper/curated subset = **1620** objects, but `meshfleetxl_train` holds
  **2229** objects (and a 394-object test split).
- **Resolution: N = 1620 (deliberate, controller-locked).** We knowingly do NOT
  use all 2229 MeshFleet train objects; we match the paper's curated 1620 so the
  CarCaption3K ablation is size-matched.
- **Locked split: train = 1420 / val = 200** (~87.7 / 12.3). MeshFleet itself used
  pre-split webdataset dirs at ~85/15 (2229 / 394); the ablation uses its own
  1420/200 split.

## 5. GATE B — aesthetic-score policy (intent recorded; configs edited in a later phase)

- MeshFleet's original filter: `min_aesthetic_score = 4.5` in ALL FOUR TRELLIS
  finetune base configs. This threshold is on the **LAION aesthetic scale**
  (metadata column `aesthetic_score`, range ~0-8; max observed in the test set
  = 7.9), NOT a 0-1 scale.
- **Ablation intent (controller GATE B default):** store the **0-1 trellis
  aesthetic score** and set `min_aesthetic_score = 0.0` in later configs so that
  **no selected object is dropped**. (No config is edited in Phase 0 — this is
  recorded intent only.)

## 6. Chosen subset (carried into later phases)

- N = **1620**, train = **1420**, val = **200**.
- `min_aesthetic_score = 0.0` against a stored 0-1 trellis aesthetic score.

---

# Generation pipelines (Task 0.2)

Three generation families feed the MeshFleet benchmark. The single generation
entrypoint for both image- and text-conditioned TRELLIS is
`$TRELLIS/evaluate_trellis_prompt_following.py`; the QA-spec 12-view renders are
produced either by that script's `get_images_from_glb(..., elevations=[90]*12)`
call or by the standalone `$TRELLIS/render_for_quality_assessment.py`. The
benchmark folder that `QA/meshfleet_benchmark/benchmark_config.yaml` points at is
assembled by `$TRELLIS/prepare_meshfleet_benchmark.ipynb`.

## A. FLUX -> TRELLIS (image-conditioned) — the `generated_flux_lora_maxR` row

- **Entrypoint:** `$TRELLIS/evaluate_trellis_prompt_following.py` run with
  `--use_flux --sample_max_reward_image`.
  Example invocation (commented in `$TRELLIS/start_3.slurm` / `start.slurm`):
  `python evaluate_trellis_prompt_following.py --output_dir "eval_prompt_alignment/output_flux_lora_maxR" --use_flux --sample_max_reward_image --prompt_file "datasets/meshfleet_benchmark/meshfleet_test.csv" --seed 3`
  (finetuned-TRELLIS variant adds `--finetuned_trellis_path assets/pretrained_TRELLIS_image_large_finetuned/snapshots/25e0d31f...`).
- **Class:** `FluxSchnellTrellisImageTo3DPipeline` (lines 60-122).
- **FLUX LoRA loaded from:** HF `DamianBoborzi/FLUX.1-schnell_meshleet`,
  `weight_name='flux_schnell_meshfleet_lora32.safetensors'` (line 80). NOTE the
  version mismatch flagged in section 7 below.
- **TRELLIS checkpoint:** base `JeffreyXiang/TRELLIS-image-large`, OR a finetuned
  folder via `--finetuned_trellis_path` (loads whole pipeline) and/or a single
  finetuned slat denoiser via `--slat_flow_checkpoint_path` (loads
  `models['slat_flow_model']` state dict).
- **Flow per object:** FLUX-schnell generates 6 images
  (`guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256`), the
  ImageReward reward model picks the max-reward image, which conditions the
  TRELLIS image-to-3D pipeline
  (`sparse_structure_sampler steps=24 cfg=7.5`, `slat_sampler steps=24 cfg=3`).
  FLUX prompt = refined_3d_prompt + "\nHigh Quality Render of 3/4 front view of
  the 3D object, studio lighting, clean background."
- **Prompts:** read from `--prompt_file` CSV with columns `sha256`,
  `refined_3d_prompt` (e.g. `datasets/meshfleet_benchmark/meshfleet_test.csv`).
- **Raw output layout (per sha `i`):** `gaussian_<i>/000.png..011.png` (12 gaussian
  renders, radius=2, fov=40, pitch~0), `glb_<i>/` (12 GLB renders,
  `elevations=[90]*12`), `sample_<i>.glb`, `sample_<i>.jpg` (the chosen FLUX
  image), `sample_<i>.mp4`, `metrics_<i>.json`, plus aggregate `metrics.csv` /
  `metrics_mean_std.csv`.

## B. TRELLIS-txt (text-conditioned)

- **Entrypoint:** SAME script `evaluate_trellis_prompt_following.py` run WITHOUT
  `--use_flux` -> selects `WrappedllTrellisTextTo3DPipeline` (lines 23-57).
- **TRELLIS checkpoint:** base `JeffreyXiang/TRELLIS-text-xlarge`, OR finetuned
  folder via `--finetuned_trellis_path`, OR a single finetuned slat denoiser via
  `--slat_flow_checkpoint_path`.
- **Flow:** prompt -> TRELLIS text-to-3D (`sparse_structure steps=24 cfg=7.5`,
  `slat steps=24 cfg=3`). Same output layout as A but without `sample_<i>.jpg`.

## C. TRELLIS-img (image-conditioned, NON-FLUX input)

- There is no separate "TRELLIS-img with a non-FLUX image source" generation
  script in this repo. The image-conditioned TRELLIS path is exercised only via
  the FLUX-fed pipeline (family A). The base `TrellisImageTo3DPipeline` is also
  used directly in `$TRELLIS/example.py` / `example_multi_image.py` for ad-hoc
  single-image inference, but those are demos, not the benchmark generators. See
  blocker note in section 7.

## D. 12-view QA-spec rendering (used to build `<sha>/000.png..011.png`)

- **Script:** `$TRELLIS/render_for_quality_assessment.py`
  (`get_quality_assessment_camera_params`, lines 33-42):
  - azimuths = `[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]`
  - elevations (TRELLIS pitch) = `[90]*12`
  - num_frames = 12, fixed radius = **1.5**
  (rendered via `get_images_from_glb` -> `render_glb_frames` ->
  `generate_views_from_angles(..., fixed_radius=1.5)` in
  `trellis/utils/eval_utils.py`). Output renamed to `000.png .. 011.png` per sha.
- This **matches** the QA library spec in CLAUDE.md (azimuth 0-330 step 30,
  elevation 90, distance ~1.5). NOTE: the inline `glb_<i>/` renders produced by
  `evaluate_trellis_prompt_following.py` pass `elevations=[90]*12` but rely on
  `render_glb_frames`'s default azimuths (`torch.linspace(0,360,13)[:-1]` =
  0,30,...,330) and `fixed_radius=1.5` — also spec-compliant. The
  `gaussian_<i>/` renders use a different camera (radius=2, fov=40, pitch~0) and
  are NOT the QA-spec views.

## E. Benchmark folder assembly

- `$TRELLIS/prepare_meshfleet_benchmark.ipynb` (cells ~915-940) copies
  `prompt_alignment_images/output_flux_lora_maxR/<obj_id>` into
  `prompt_alignment_images_testset/generated_flux_lora_maxResults/<obj_id>` and
  the aligned GT into `.../ground_truth/<obj_id>`.
- `QA/meshfleet_benchmark/benchmark_config.yaml` `generated_folder` =
  `/mnt/damian/Projects/TRELLIS/prompt_alignment_images_testset/generated_flux_lora_maxR`
  — i.e. it consumes the output of family A after the notebook assembly step.
- Evaluation of the assembled folders is driven by
  `QA/run_meshfleet_eval.py` per `QA/analysis/track_b/out/RUN_LOG.md` (the
  `TRELLIS` and `TRELLIS_FT` rows use pre-generated folders under
  `data/meshfleet/benchmark_data/scaled/<method-folder>`; evaluation itself does
  no generation).

## 7. Blockers / concerns

1. **TRELLIS-img standalone generator: not a distinct script (minor).** Image
   conditioning is only realized through the FLUX-fed pipeline (family A). If a
   later phase wants TRELLIS image-to-3D from a non-FLUX image source for the
   ablation, it must add a small driver around `TrellisImageTo3DPipeline`
   (pattern available in `example.py` / `example_multi_image.py`). Not a hard
   blocker for matching MeshFleet, since MeshFleet's own image row is also
   FLUX-fed.
2. **FLUX LoRA version ambiguity (flag).** The training config pushes to
   `DamianBoborzi/flux_schnell_MeshFleet_newPrompts_lora32_6` (8000 steps), but
   the generation script hard-codes the older `DamianBoborzi/FLUX.1-schnell_meshleet`
   / `flux_schnell_meshfleet_lora32.safetensors` (with `_newPrompts_lora32_3` and
   `CarCaption3K_lora32_1` commented out). For the ablation, treat the 8000-step
   `lora32` recipe as the authoritative FLUX budget, and pin the exact LoRA repo
   in a later phase.
3. **num_gpus not recorded in any TRELLIS run** (train.py default -1 = all GPUs).
   To reproduce wall-clock-comparable budgets, a later phase must decide a fixed
   GPU count; step counts (recorded above) are the reproducible budget anchor.
