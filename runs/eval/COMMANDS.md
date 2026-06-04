# CarCaption3K-1620 Controlled Ablation — Generate → Eval Command Templates

Generated: 2026-06-04 (Phase 5).
These are **templates**. Generation requires the trained checkpoints to exist and
runs under the TRELLIS conda env. Evaluation runs under the `trellis_qa` conda env.

---

## GT layout contract (verified 2026-06-04)

```
$ ls data/meshfleet/benchmark_data/meshfleet_eval_images | head -1 \
    | xargs -I{} ls data/meshfleet/benchmark_data/meshfleet_eval_images/{}
000.png  001.png  002.png  003.png  004.png  005.png
006.png  007.png  008.png  009.png  010.png  011.png
metadata.json
```

Each GT object directory contains exactly **12 PNG views** (`000.png`–`011.png`,
azimuths 0–330 step 30, elevation 90, radius 1.5) plus a `metadata.json`.
Total: **232 objects × 12 views**.

**Prompt file (held-out 232):** generation reads `sha256` + `refined_3d_prompt`
from `manifests/meshfleet_heldout_232_prompts.csv`, built by
`scripts/build_heldout_prompt_file.py` for exactly the 232 GT object dirs
(sorted by sha, all prompts non-empty). Regenerate after any GT/source change:

```bash
/home/damian/miniconda3/envs/trellis_qa/bin/python scripts/build_heldout_prompt_file.py
```

---

## Preprocessing flags (from `analysis/track_b/out/RUN_LOG.md`)

All reference rows (TRELLIS, TRELLIS_FT, Hunyuan3D_20, Hunyuan3D_21,
Instantmesh) were evaluated with:

```
--skip-all-preprocessing
```

The reference data lived in `data/meshfleet/benchmark_data/scaled/<method-folder>`,
which had already been background-removed, view-aligned, and scale-equalized in
a prior preprocessing pass. The CC1620 ablation variants MUST use the same flag
so the eval pipeline does not re-run rembg on already-RGBA inputs (this caused
a 26 % geometric-metric drift in an earlier run — see RUN_LOG §"Cross-comparison
vs legacy per-viewpoint outputs"). If the generated images have NOT been
preprocessed through the QA-library preprocessing pipeline (background removal →
view alignment → scale equalization), **remove `--skip-all-preprocessing`** and
let the pipeline run all stages.

> **Assembly (RESOLVED — OI-3):** generation writes sha-keyed
> `sample_<sha>.glb`. Assembly stages those GLBs to `<sha>/<sha>.glb` and renders
> them at the GT camera with the canonical `render_for_quality_assessment.py`
> (the same renderer used for every reference method: az 0–330 step 30,
> elevation 90, radius 1.5) → `<sha>/000.png..011.png`. The old
> `gaussian_<sha>/` (radius 2, non-QA-spec) and `glb_<sha>/renders` copy paths
> are NOT used. See the OI-3 section below.

---

## Open items (must resolve before generation)

### OI-1 · FLUX LoRA repo pin (MUST FIX before running P0/P2)

`evaluate_trellis_prompt_following.py` line 80 hard-codes:

```python
self.flux_pipe.load_lora_weights(
    'DamianBoborzi/FLUX.1-schnell_meshleet',
    weight_name='flux_schnell_meshfleet_lora32.safetensors')
```

For the CarCaption3K-1620 ablation this **must** be changed to load from the
newly trained LoRA repo (placeholder path — update when training is complete):

```python
self.flux_pipe.load_lora_weights(
    'ai-toolkit/output/flux_carcaption3k_1620_lora32',   # LOCAL path
    weight_name='<weight_file>.safetensors')
# OR if pushed to HF:
# self.flux_pipe.load_lora_weights(
#     'DamianBoborzi/flux_carcaption3k_1620_lora32',
#     weight_name='flux_carcaption3k_1620_lora32.safetensors')
```

The training config (`$AITK/config/train_lora_flux_schnell_32_3.yaml`) pushes
to `DamianBoborzi/flux_schnell_MeshFleet_newPrompts_lora32_6`. Confirm the
actual output location before generation.

### OI-2 · ss_flow stage checkpoint loading (MUST FIX for P1/P2)

The generation script currently only accepts `--slat_flow_checkpoint_path` for
the SLAT denoiser (line 134). There is **no CLI flag** for the SS (sparse
structure) flow checkpoint. To load both finetuned stages for CC1620, either:

- Add `--ss_flow_checkpoint_path` to the script's argparser and a corresponding
  `self.pipeline.models['ss_flow_model'].load_state_dict(...)` call (mirrors
  the existing slat pattern), OR
- Assemble a full finetuned pipeline folder (as `--finetuned_trellis_path`
  expects `from_pretrained()`-compatible layout) containing both stage weights.

The SS-flow checkpoint paths (100k-step finetuned) are:
- txt: `$TRELLIS/outputs/sweeps_ss_flow_txt/sweep_wqt27r19_20250704_034000/ckpts/denoiser_step0100000.pt`
- img: `$TRELLIS/outputs/sweeps_ss_flow_img/sweep_x3va3sda_20250705_011646/ckpts/denoiser_step0100000.pt`

### OI-3 · GLB → QA-render assembly (RESOLVED)

**RESOLVED.** Generation outputs are already sha-keyed: per object it writes
`sample_<sha>.glb` to `--output_dir`. Assembly no longer copies/renames the
fragile `glb_<sha>/` (or `glb_<sha>/renders/`) frames (blender-internal naming,
loaded in unordered fashion). Instead it uses the **canonical renderer** — the
same `render_for_quality_assessment.py` used for every reference method — so the
generated 3D is rendered at the GT camera (az 0–330 step 30, elevation 90,
radius 1.5), apples-to-apples with the reference rows.

Two steps (used verbatim in the P0/P1/P2 blocks below):

```bash
# 1) Stage sample_<sha>.glb -> <staging>/<sha>/<sha>.glb (pure-python; any env)
python /home/damian/Projects/quality_assessment_library/scripts/stage_generated_glbs.py \
    --gen-dir "$RAW_OUT" --staging-dir "$RAW_OUT/_staging"

# 2) Render each staged GLB at the GT camera -> $GEN_FOLDER/<sha>/000.png..011.png
#    (runs under the generation/blender env, NOT trellis_qa)
cd /home/damian/Projects/TRELLIS && python render_for_quality_assessment.py \
    --input_dir "$RAW_OUT/_staging" --mesh_output_dir "$GEN_FOLDER"
```

`render_for_quality_assessment.py` writes zero-padded `{i:03d}.png` at the
QA-spec camera and skips objects already rendered. The eval then matches GT↔gen
by viewpoint filename directly. The old `gaussian_<sha>/` (radius 2, fov 40 —
non-QA-spec) and `glb_<sha>/renders` copy paths are no longer used.

---

## P0 · FLUX_TRELLIS_CC1620

**Status:** Requires trained `flux_carcaption3k_1620_lora32` LoRA (OI-1).
**Generation env:** TRELLIS conda env (with FLUX + TRELLIS inference deps).

### Generation

```bash
# PREREQUISITE: Fix OI-1 (FLUX LoRA path) in evaluate_trellis_prompt_following.py
# PREREQUISITE: flux_carcaption3k_1620_lora32 LoRA trained and accessible

cd /home/damian/Projects/TRELLIS

python evaluate_trellis_prompt_following.py \
    --output_dir "eval_prompt_alignment/output_flux_cc1620" \
    --use_flux \
    --sample_max_reward_image \
    --prompt_file "/home/damian/Projects/quality_assessment_library/manifests/meshfleet_heldout_232_prompts.csv" \
    --seed 3
```

Notes:
- `--use_flux` selects `FluxSchnellTrellisImageTo3DPipeline` (image-to-3D via FLUX).
- No `--finetuned_trellis_path` → uses base `JeffreyXiang/TRELLIS-image-large`.
- No `--slat_flow_checkpoint_path` → base TRELLIS SLAT weights (P0 tests only
  the FLUX LoRA contribution; add finetuned TRELLIS stages if desired).
- FLUX LoRA is loaded via `use_lora=True` (triggered by absence of
  `--use_flux_base_model`). **OI-1 must be fixed first.**
- Raw output per sha: `sample_<sha>.glb` (the GLB consumed by assembly), plus
  `gaussian_<sha>/` and `glb_<sha>/` render dirs (not used by assembly).

### Assembly (stage sample_<sha>.glb → canonical GT-camera render)

```bash
# Stage the sha-keyed GLBs, then render at the GT camera with the canonical
# renderer (same as all reference methods) -> $GEN_FOLDER/<sha>/000.png..011.png.
RAW_OUT="/home/damian/Projects/TRELLIS/eval_prompt_alignment/output_flux_cc1620"
GEN_FOLDER="/home/damian/Projects/quality_assessment_library/data/ablation/gen/FLUX_TRELLIS_CC1620"

# (1) sample_<sha>.glb -> <sha>/<sha>.glb (pure-python; any env)
python /home/damian/Projects/quality_assessment_library/scripts/stage_generated_glbs.py \
    --gen-dir "$RAW_OUT" --staging-dir "$RAW_OUT/_staging"

# (2) render at GT camera (gen/blender env, NOT trellis_qa)
cd /home/damian/Projects/TRELLIS && python render_for_quality_assessment.py \
    --input_dir "$RAW_OUT/_staging" --mesh_output_dir "$GEN_FOLDER"
# -> $GEN_FOLDER/<sha>/000.png..011.png (az 0..330 step30, elev 90, radius 1.5)
```

### Eval (trellis_qa env)

```bash
cd /home/damian/Projects/quality_assessment_library

/home/damian/miniconda3/envs/trellis_qa/bin/python run_meshfleet_eval.py \
    --config meshfleet_benchmark/benchmark_config.yaml \
    --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
    --gen-folder data/ablation/gen/FLUX_TRELLIS_CC1620 \
    --metadata-file data/meshfleet/meshfleet_test.csv \
    --output-folder runs/FLUX_TRELLIS_CC1620/metrics \
    --per-object --model-name FLUX_TRELLIS_CC1620 \
    --skip-all-preprocessing
```

> **Preprocessing note:** Use `--skip-all-preprocessing` if the assembled images
> have already been background-removed and scaled. Remove this flag if running
> on raw renders that need the full preprocessing pipeline.

---

## P1 · TRELLIS_TXT_CC1620

**Status:** Requires both finetuned TRELLIS-txt stages (OI-2 for ss_flow).
**Generation env:** TRELLIS conda env.

Finetuned checkpoint paths:
- SLAT-txt (100k steps): `outputs/sweeps_slat_flow_txt/sweep_lcqjobfr_20250629_230408/ckpts/denoiser_step0100000.pt`
- SS-txt (100k steps): `outputs/sweeps_ss_flow_txt/sweep_wqt27r19_20250704_034000/ckpts/denoiser_step0100000.pt`

### Generation

```bash
# PREREQUISITE: Fix OI-2 (add --ss_flow_checkpoint_path to the generation script)
# Both SS-txt and SLAT-txt 100k-step checkpoints must exist (verified present).

cd /home/damian/Projects/TRELLIS

python evaluate_trellis_prompt_following.py \
    --output_dir "eval_prompt_alignment/output_trellis_txt_cc1620" \
    --prompt_file "/home/damian/Projects/quality_assessment_library/manifests/meshfleet_heldout_232_prompts.csv" \
    --seed 3 \
    --slat_flow_checkpoint_path \
        "outputs/sweeps_slat_flow_txt/sweep_lcqjobfr_20250629_230408/ckpts/denoiser_step0100000.pt" \
    --ss_flow_checkpoint_path \
        "outputs/sweeps_ss_flow_txt/sweep_wqt27r19_20250704_034000/ckpts/denoiser_step0100000.pt"
```

Notes:
- No `--use_flux` → selects `WrappedllTrellisTextTo3DPipeline` (text-to-3D).
- No `--finetuned_trellis_path` → base `JeffreyXiang/TRELLIS-text-xlarge` is
  loaded; both finetuned stage weights are then patched in via load_state_dict.
- `--ss_flow_checkpoint_path` **does not exist yet** in the script — see OI-2.
- `.pt` checkpoints: load with `torch.load(..., map_location='cpu')` (not
  safetensors); the script currently uses `load_file` (safetensors). The loader
  may need to be updated for `.pt` format. Alternatively convert to safetensors
  before passing.

### Assembly (stage sample_<sha>.glb → canonical GT-camera render)

```bash
RAW_OUT="/home/damian/Projects/TRELLIS/eval_prompt_alignment/output_trellis_txt_cc1620"
GEN_FOLDER="/home/damian/Projects/quality_assessment_library/data/ablation/gen/TRELLIS_TXT_CC1620"

# (1) sample_<sha>.glb -> <sha>/<sha>.glb (pure-python; any env)
python /home/damian/Projects/quality_assessment_library/scripts/stage_generated_glbs.py \
    --gen-dir "$RAW_OUT" --staging-dir "$RAW_OUT/_staging"

# (2) render at GT camera (gen/blender env, NOT trellis_qa)
cd /home/damian/Projects/TRELLIS && python render_for_quality_assessment.py \
    --input_dir "$RAW_OUT/_staging" --mesh_output_dir "$GEN_FOLDER"
# -> $GEN_FOLDER/<sha>/000.png..011.png (az 0..330 step30, elev 90, radius 1.5)
```

### Eval (trellis_qa env)

```bash
cd /home/damian/Projects/quality_assessment_library

/home/damian/miniconda3/envs/trellis_qa/bin/python run_meshfleet_eval.py \
    --config meshfleet_benchmark/benchmark_config.yaml \
    --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
    --gen-folder data/ablation/gen/TRELLIS_TXT_CC1620 \
    --metadata-file data/meshfleet/meshfleet_test.csv \
    --output-folder runs/TRELLIS_TXT_CC1620/metrics \
    --per-object --model-name TRELLIS_TXT_CC1620 \
    --skip-all-preprocessing
```

---

## P2 · TRELLIS_IMG_CC1620

**Status:** Requires trained `flux_carcaption3k_1620_lora32` LoRA (OI-1) AND
both finetuned TRELLIS-img stages (OI-2 for ss_flow).
**Generation env:** TRELLIS conda env.

Finetuned checkpoint paths:
- SLAT-img (30k steps): `outputs/sweeps_slat_flow_img/sweep_uoxlv9op_20250627_151924/ckpts/denoiser_step0030000.pt`
- SS-img (100k steps): `outputs/sweeps_ss_flow_img/sweep_x3va3sda_20250705_011646/ckpts/denoiser_step0100000.pt`

Note: The image-conditioned TRELLIS path is exercised only via the FLUX-fed
pipeline (there is no separate non-FLUX image generation script — see RECON.md
§C and §7.1). P2 therefore uses the same `--use_flux` flag as P0 but with
finetuned TRELLIS-img stages patched in.

### Generation

```bash
# PREREQUISITE: Fix OI-1 (FLUX LoRA path) in evaluate_trellis_prompt_following.py
# PREREQUISITE: Fix OI-2 (add --ss_flow_checkpoint_path to the generation script)
# All four finetuned stage checkpoints must exist (verified present).

cd /home/damian/Projects/TRELLIS

python evaluate_trellis_prompt_following.py \
    --output_dir "eval_prompt_alignment/output_trellis_img_cc1620" \
    --use_flux \
    --sample_max_reward_image \
    --prompt_file "/home/damian/Projects/quality_assessment_library/manifests/meshfleet_heldout_232_prompts.csv" \
    --seed 3 \
    --slat_flow_checkpoint_path \
        "outputs/sweeps_slat_flow_img/sweep_uoxlv9op_20250627_151924/ckpts/denoiser_step0030000.pt" \
    --ss_flow_checkpoint_path \
        "outputs/sweeps_ss_flow_img/sweep_x3va3sda_20250705_011646/ckpts/denoiser_step0100000.pt"
```

Notes:
- `--use_flux` selects `FluxSchnellTrellisImageTo3DPipeline`.
- No `--finetuned_trellis_path` → base `JeffreyXiang/TRELLIS-image-large`;
  both finetuned-img stage weights patched in.
- OI-1 (FLUX LoRA pin) and OI-2 (ss_flow CLI arg) must both be fixed.
- `.pt` checkpoint loader note applies (same as P1).

### Assembly (stage sample_<sha>.glb → canonical GT-camera render)

```bash
RAW_OUT="/home/damian/Projects/TRELLIS/eval_prompt_alignment/output_trellis_img_cc1620"
GEN_FOLDER="/home/damian/Projects/quality_assessment_library/data/ablation/gen/TRELLIS_IMG_CC1620"

# (1) sample_<sha>.glb -> <sha>/<sha>.glb (pure-python; any env)
python /home/damian/Projects/quality_assessment_library/scripts/stage_generated_glbs.py \
    --gen-dir "$RAW_OUT" --staging-dir "$RAW_OUT/_staging"

# (2) render at GT camera (gen/blender env, NOT trellis_qa)
cd /home/damian/Projects/TRELLIS && python render_for_quality_assessment.py \
    --input_dir "$RAW_OUT/_staging" --mesh_output_dir "$GEN_FOLDER"
# -> $GEN_FOLDER/<sha>/000.png..011.png (az 0..330 step30, elev 90, radius 1.5)
```

### Eval (trellis_qa env)

```bash
cd /home/damian/Projects/quality_assessment_library

/home/damian/miniconda3/envs/trellis_qa/bin/python run_meshfleet_eval.py \
    --config meshfleet_benchmark/benchmark_config.yaml \
    --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
    --gen-folder data/ablation/gen/TRELLIS_IMG_CC1620 \
    --metadata-file data/meshfleet/meshfleet_test.csv \
    --output-folder runs/TRELLIS_IMG_CC1620/metrics \
    --per-object --model-name TRELLIS_IMG_CC1620 \
    --skip-all-preprocessing
```

---

## Summary of open items

| ID   | Affects | Action required |
| ---- | ------- | --------------- |
| OI-1 | P0, P2  | Edit `evaluate_trellis_prompt_following.py` line 80 to load `flux_carcaption3k_1620_lora32` LoRA instead of the old MeshFleet LoRA. Confirm local vs HF path after training completes. |
| OI-2 | P1, P2  | Add `--ss_flow_checkpoint_path` arg to the generation script (argparser + `pipeline.models['ss_flow_model'].load_state_dict(...)` call). Also check that `.pt` checkpoint format is handled (script currently uses `load_file` for safetensors). |
| OI-3 | All     | **RESOLVED.** Outputs are sha-keyed (`sample_<sha>.glb`). Assembly stages them to `<sha>/<sha>.glb` (`scripts/stage_generated_glbs.py`) and renders at the GT camera with the canonical `render_for_quality_assessment.py` (same renderer as all reference methods) → `<sha>/000.png..011.png`. Old `glb_<sha>/`/`gaussian_<sha>/` copy paths removed. |
