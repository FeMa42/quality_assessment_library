# Spec — CarCaption3K-1620 Controlled Ablation (autonomous prep harness)

- Date: 2026-06-03
- Branch context: `meshflee_benchmark`
- Status: **design approved (verbal), pending written review**
- Source brief: `carcaption3k_controlled_ablation_handoff.md` (this repo)
- Companion (next artifact): an executable implementation plan produced via `writing-plans`.

---

## 1. Problem & goal

The BMVC manuscript compares MeshFleet finetuning against caption-filtered CarCaption
subsets. The existing CarCaption3K row is **confounded** by data source *and* object count
*and* step budget *and* effective epochs. The rebuttal-prep experiment removes the size /
budget confound:

> Build one deterministic **1,620-object** subset of CarCaption ("CarCaption3K-1620"),
> reuse it across every model family, match each MeshFleet run's budget, and evaluate on the
> **same MeshFleet held-out test set**. This isolates the effect of MeshFleet's curation from
> the effect of dataset scale.

Because both datasets are exactly 1,620 objects, **matching steps == matching epochs**, so
budget-matching reduces to "use MeshFleet's actual step counts."

This spec covers an **autonomous prep harness**: an agent builds the locked subset, all
per-family adapters, configs, smoke tests, and a sanity gate, then **emits ready-to-run
launch commands**. The agent does **not** fire the heavy multi-GPU jobs — the human launches
rendering / training / inference / eval.

## 2. Locked decisions

| Decision | Choice | Consequence |
| --- | --- | --- |
| Model families in scope | **FLUX→TRELLIS (P0) + TRELLIS-Txt (P1) + TRELLIS-image (P2)** | SV3D (P3) deferred unless Table 3 is directly attacked. |
| Autonomy boundary | **Prep + dry-run; human launches GPUs** | Terminal state = "built, smoke-tested, sanity-gate green, commands emitted". |
| Sampling pool | **`car_meshes_trellis_aesthetics_65`** (≈2,648 objects) | Sample 1,620 from the aesthetic≥0.65 pool, intersected with caption/render/readability. |
| Budget source | **Recover MeshFleet's actual step counts from run logs** | If TRELLIS budgets cannot be recovered → STOP and ask. |

**Baked defaults (override if wrong):**
- `aesthetic_score` for the TRELLIS metadata is sourced from MeshFleet's own method if
  recoverable in Phase 0; otherwise from the trellis-aesthetic predictor that produced the
  `_65` split. Membership in `_65` guarantees score ≥ 0.65.
- FLUX is finetuned as a **LoRA** (matching MeshFleet); TRELLIS uses the finetune recipe in
  the committed configs.
- Assets are **symlinked**, not copied, when staging the 1,620 objects (saves tens of GB).

## 2b. Post-extraction addendum (verbatim findings that refine the design)

- **TRELLIS finetune = 4 runs per the MeshFleet recipe, not 2.** MeshFleet finetuned **both** flow stages
  (sparse-structure *and* structured-latent) for each of txt and img. Recovered realized budgets
  (from `TRELLIS/outputs/sweeps_*/.../command.txt` + highest `denoiser_step*.pt`):
  `ss_flow_txt`=**100k**, `slat_flow_txt`=**100k**, `ss_flow_img`=**100k**, `slat_flow_img`=**30k**.
  FLUX=**8k** (`train_lora_flux_schnell_32_3.yaml`, `lr=1e-4`, lora32). The plan creates 4 TRELLIS
  configs; generation/inference must load both finetuned stages.
- **GATE A — object-count conflict.** `meshfleetxl_train` has **2,229** rows and `meshfleetxl_test`
  **394** — so MeshFleet's actual TRELLIS run was *not* 1,620 objects. The handoff says match the
  actual run, but also mandates 1,620. This is surfaced to the user in Phase 0 (default: keep 1,620).
- **TRELLIS train/val = separate directories**, each with its own `metadata.csv` (no split column).
  `train.py` takes `--data_dir` and `--eval_data_dir`. `keep_ckpt=1`, so checkpoint selection = final step.
- **TRELLIS metadata schema (mirror `meshfleetxl_test`)**: `sha256, file_identifier, aesthetic_score,
  captions(JSON list), local_path(./raw/<sha>.<ext>), rendered, voxelized, num_voxels, cond_rendered,
  feature_dinov2_vitl14_reg, ss_latent_ss_enc_conv3d_16l8_fp16,
  latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16, eval_rendered`.
- **FLUX dataset format**: a folder of `<sha>.png` + `<sha>.txt` (caption_ext=txt) is read directly via
  `folder_path`; we symlink the 1,620 pairs from `CarCaptionData_ai_toolkit`.
- **Eval gen-folder contract**: `<gen>/<sha>/000.png..011.png`, matched by viewpoint filename to the GT.
  Reference rows `TRELLIS` and `TRELLIS_FT` already exist in `analysis/track_b/out/per_object_metrics_all.csv`.
- **Custom subset module required**: TRELLIS stage scripts import `datasets.<SUBSET>`; we add
  `dataset_toolkits/datasets/CarCaption3K1620.py` (get_metadata reads the prebuilt `metadata.csv`,
  download is a no-op since meshes are pre-staged).
- **Render non-determinism** lives at `render.py:34` / `render_cond.py:34` (`np.random.rand()` offset);
  the plan pins a per-object seed.

## 3. Non-goals (hard constraints)

- Do **not** change the MeshFleet held-out evaluation set (232 objects × 12 views).
- Do **not** resample CarCaption per model — one canonical manifest for every run.
- Do **not** tune hyperparameters uniquely for CarCaption3K-1620.
- Do **not** replace any existing reported table row until the controlled row is validated.
- Do **not** launch GPU jobs autonomously (per the autonomy decision).
- Do **not** do Arena Round 2 or new zero-shot baselines.

## 4. Architecture

### 4.1 Single source of truth → per-family adapters

```
manifests/carcaption3k_1620_locked.csv      (canonical, never resampled)
        │
        ├─► FLUX adapter      → filtered ai-toolkit dataset (JSON or symlink folder)
        ├─► TRELLIS adapter   → TRELLIS metadata.csv + custom subset module
        └─► eval is shared    → run_meshfleet_eval.py on the fixed 232-object set
```

### 4.2 Cross-repo, additive layout

| Repo | New / changed artifacts |
| --- | --- |
| `quality_assessment_library/` (orchestration home) | `scripts/build_carcaption3k_1620_subset.py`, `scripts/build_flux_filtered_dataset.py`, `scripts/build_trellis_metadata.py`, `scripts/validate_ablation_ready.py`, `scripts/assemble_ablation_table.py`, `manifests/`, `configs/ablations/`, `runs/`, `tables/` |
| `ai-toolkit/` | `config/flux_trellis_carcaption3k_1620.yaml` (clone of MeshFleet FLUX config) + filtered dataset (JSON/symlink folder) |
| `TRELLIS/` | `dataset_toolkits/datasets/CarCaption3K1620.py` (custom subset module), a pipeline driver script, `configs/generation/{slat_flow_txt_dit_XL...,slat_flow_img_dit_L...}_carcaption3k_1620.json` |

Nothing in the existing MeshFleet training/eval code is rewritten; everything is added
alongside.

## 5. Canonical data model

### 5.1 Locked manifest — `manifests/carcaption3k_1620_locked.csv`

Columns (per handoff §3):

| column | meaning |
| --- | --- |
| `object_id` | stable key = the CarCaption SHA256 filename stem |
| `sha256` | same value (CarCaption keys *are* sha256); document that they coincide |
| `source_dataset` | constant `CarCaption3K` |
| `asset_path` | absolute path to the mesh (in `car_meshes_trellis_aesthetics_65`) |
| `render_path` | absolute path to the FLUX training image `CarCaptionData_ai_toolkit/<sha>.png` (the asset FLUX trains on; TRELLIS regenerates its own renders in Phase 3) |
| `caption` | caption text from `CarCaptionData_ai_toolkit/<sha>.txt` |
| `split` | `train` / `val` mirroring the recovered MeshFleet protocol |
| `selection_seed` | `20260603` |
| `selection_notes` | inclusion/exclusion/fallback reason |

Plus `manifests/carcaption3k_1620_locked_summary.json`: counts before/after each filter,
seed, paths used, timestamp (passed in — not generated in-process), and the manifest SHA256.

### 5.2 TRELLIS metadata — `metadata.csv` (indexed by `sha256`)

Mirror `TRELLIS/datasets/meshfleetxl_train/metadata.csv`:

```
sha256 (index), file_identifier, aesthetic_score, captions (JSON list[str]),
local_path, rendered(bool), voxelized(bool), num_voxels(int), cond_rendered(bool),
feature_dinov2_vitl14_reg(bool),
ss_latent_ss_enc_conv3d_16l8_fp16(bool),
latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16(bool)
```

`local_path` is relative to the pipeline `--output_dir`; boolean flags start `False` and are
flipped by the pipeline stages.

### 5.3 FLUX dataset (ai-toolkit)

ai-toolkit consumes either a folder of `<name>.png` + `<name>.txt` pairs (`folder_path`) or a
`dataset_path` JSON mapping `{image_path: caption}`. The filtered dataset contains exactly the
1,620 selected objects' preview PNG + caption, sourced from
`CarCaptionData/CarCaptionData_ai_toolkit/`.

### 5.4 Evaluation target (fixed, do not modify)

- GT images: `data/meshfleet/benchmark_data/meshfleet_eval_images/` — 232 objects × 12 views
  (`000.png`..`011.png`; azimuths 0–330°, elevation 90°).
- Metadata / prompts: `data/meshfleet/meshfleet_test.csv` (prompt column `refined_3d_prompt`).
- Metrics: MSE, CLIP-S_img, LPIPS, SSIM, PSNR (appearance); geometric image metrics; CLIPScore_txt
  + ImageReward (prompt); vehicle dims; FID/KID/IS (model-level).

## 6. Phased design

Each phase ends with a **hard gate**; a failed gate STOPs the agent and surfaces the issue.

### Phase 0 — Reconnaissance & budget recovery
**Goal:** establish the authoritative MeshFleet recipes the matched runs must copy.

Recover and record into `runs/reference_budgets.json` + `runs/RECON.md`:
- **FLUX**: MeshFleet config (`ai-toolkit/config/train_lora_flux_schnell_32_3.yaml`), actual
  step count (~8k), lora rank, buckets, sampler, sample/checkpoint cadence, dataset path.
- **TRELLIS-txt & img**: the *real* finetune budgets. Search the MeshFleet TRELLIS run output
  dirs for `command.txt` / `config.json` (the trainer writes these). Record `max_steps`,
  `batch_size_per_gpu`, `batch_split`, `lr`, `i_log/i_sample/i_save`, `keep_ckpt`,
  `min_aesthetic_score`, `finetune_ckpt`, `num_gpus`.
- **Split**: the real MeshFleet train/val(/test) split (resolves the 1420/200 vs 1169/200/251
  ambiguity from the handoff).
- **Aesthetic score provenance & threshold**: how MeshFleet populated `aesthetic_score` and on
  what scale; whether the trellis aesthetic predictor is available locally. **Phase 0 must
  output the concrete `min_aesthetic_score` value (and score scale) the CarCaption3K-1620
  TRELLIS configs will use**, chosen against the recovered MeshFleet metadata so the filter does
  not empty the set. (Per user: the agent resolves the exact threshold here, not the spec.)
- **Templates**: the existing meshfleet TRELLIS subset module + `meshfleetxl_train/test`
  layout to mirror; blender path (`/home/damian/Projects/Diffus3D/blender-3.2.2-linux-x64/blender`).
- **GPU/env**: available GPUs, CUDA, conda/venv envs per repo, free disk on the target volume.

**Gate:** `reference_budgets.json` has concrete step counts for FLUX, TRELLIS-txt, TRELLIS-img.
**STOP** if TRELLIS budgets are unrecoverable.

### Phase 1 — Lock the subset
**Build `scripts/build_carcaption3k_1620_subset.py`:**
1. Enumerate meshes in `car_meshes_trellis_aesthetics_65`.
2. Strict inclusion filter (handoff §3): mesh readable; caption `.txt` exists & non-empty;
   preview `.png` exists (the FLUX training image); sha unique (no multi-ID dupes); **not** in
   the MeshFleet held-out test set (check sha overlap against `meshfleet_test.csv`). The
   multi-view render zip is recorded if present but is **not** a hard requirement (TRELLIS
   regenerates renders in Phase 3).
3. Sort candidates by sha256 (stable key) before sampling.
4. Sample exactly **1,620** with `seed=20260603`.
5. Deterministic train/val split mirroring Phase 0's recovered MeshFleet protocol.
6. Write `manifests/carcaption3k_1620_locked.csv` + `_summary.json`.

**Gate:** exactly 1,620 unique rows; all `asset_path`/`render_path`/`caption` resolve; sha unique;
split counts match protocol. **STOP** if <1,620 pass the strict filter (handoff §3).

### Phase 2 — FLUX (P0) prep
1. `scripts/build_flux_filtered_dataset.py`: from the manifest, materialize the filtered
   ai-toolkit dataset (symlinked PNG+TXT folder, or `{path: caption}` JSON) of 1,620 objects.
2. `ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml`: clone MeshFleet FLUX config; change
   **only** dataset path, training name/output, HF repo id, and `steps` = recovered FLUX budget.
   Keep batch/lr/buckets/lora/sampler/cadence identical.
3. Dry-run: instantiate the dataloader, pull a few batches, assert 1,620 items found and
   captions present. Delete any stale `.aitk_size.json` first.

**Outputs/commands emitted:** FLUX train cmd; FLUX-inference → TRELLIS(zero-shot) → eval chain.
**Gate:** dataloader dry-run loads 1,620 items with captions.

### Phase 3 — TRELLIS data pipeline (de-risking phase)
1. `dataset_toolkits/datasets/CarCaption3K1620.py`: custom subset module mirroring the meshfleet
   one — `get_metadata()` returns a DataFrame indexed by sha256 with `local_path` (→ aesthetics_65
   mesh), `captions` (JSON list from the `.txt`), `aesthetic_score`, `file_identifier`.
2. `scripts/build_trellis_metadata.py`: produce the initial `metadata.csv` from the locked
   manifest (all stage flags `False`).
3. Pipeline driver running stages in order with `build_metadata` between each:
   `render.py` (150 views, 512px) → `voxelize.py` (64³) → `extract_feature.py` (dinov2_vitl14_reg)
   → `encode_ss_latent.py` → `encode_latent.py` (slat) → `render_cond.py` (24 views, 1024px).
   Pin the render offset seed (otherwise renders are non-deterministic).
4. **Smoke test on 2–3 objects end-to-end** and verify outputs match the trainer-expected
   schema (`latents/<model>/<sha>.npz`, `renders_cond/<sha>/transforms.json`, metadata flags).
5. **Mesh-format check:** confirm glb/fbx/obj/gltf/blend all render via blender; flag any
   unsupported format and record how many of the 1,620 it affects.

**Outputs/commands emitted:** full-1,620 pipeline commands (rank/world_size parameterized) +
disk estimate. **Gate:** smoke chain produces valid latents + cond renders for the sample objects;
no unsupported-format objects silently dropped.

### Phase 4 — TRELLIS finetune configs (txt + img)
1. Clone `configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_finetune.json` and
   `..._img_dit_L_64l8p2_fp16_finetune.json` → `*_carcaption3k_1620.json`. Set `data_dir`,
   `max_steps` = recovered MeshFleet budgets, `finetune_ckpt`, output dirs; reconcile
   `min_aesthetic_score` so our `aesthetic_score` values don't filter the set to empty.
2. Short dry-run training step on the Phase-3 smoke dataset (or the `_test` config) to confirm
   dataloader + model + checkpoint loading work on CarCaption3K-1620 data.

**Outputs/commands emitted:** `train.py` launch cmds for txt and img.
**Gate:** dry-run training step completes and writes a checkpoint without shape/key errors.

### Phase 5 — Evaluation wiring
1. Per-family generation → eval against the **unchanged** 232-object set:
   ```
   python run_meshfleet_eval.py \
     --config meshfleet_benchmark/benchmark_config.yaml \
     --gen-folder <generated_outputs> \
     --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
     --metadata-file data/meshfleet/meshfleet_test.csv \
     --output-folder runs/<run_id>/metrics --per-object --model-name <variant>
   ```
2. `scripts/assemble_ablation_table.py`: build `tables/carcaption3k_1620_controlled_ablation_summary.{csv,md}`
   with rows base / MeshFleet-reference / CarCaption3K-1620-matched per family, pulling existing
   reference numbers from `analysis/track_b/out/`.

**Gate:** table assembler runs on a stub and produces the expected columns
(Family, Variant, Train objects, Budget, MSE, CLIP-S_img, CLIPScore_txt, ImageReward, 3D metrics, Notes).

### Phase 6 — Run logging, sanity gate, deliverables
1. `scripts/validate_ablation_ready.py` encodes handoff §7; must pass before "ready to launch":
   - manifest has exactly 1,620 rows; sha unique;
   - all asset/render/caption paths exist; manifest hash recorded;
   - train/val split matches protocol;
   - the same manifest path is referenced by every ablation config;
   - eval manifest == existing MeshFleet held-out set;
   - a dataloader dry-run batch works for each family.
2. Per-run scaffolding: `runs/<run_id>/{RUN_LOG.md, config_resolved.{yaml,json}, manifest_sha256.txt, metrics.json, metrics.csv}`.

**Deliverables (handoff §8):** the manifest + summary, `configs/ablations/*`, `runs/*/RUN_LOG.md`,
`runs/*/metrics.*`, `tables/carcaption3k_1620_controlled_ablation_summary.{csv,md}`, and the exact
train/eval commands.

## 7. Environment reference (verified during recon)

**CarCaptionData** `/home/damian/Projects/datasets/CarCaptionData`
- `CarCaptionData_meshes` — 3,064 meshes, sha256-named (GLB 2012 / FBX 487 / OBJ 313 / GLTF 101 / BLEND 101)
- `CarCaptionData_ai_toolkit` — 2,633 `<sha>.txt` + `<sha>.png` (FLUX training format)
- `CarCaptionData_renders` — 2,633 `<sha>.zip`
- `car_meshes_trellis_aesthetics_65` — **2,648 meshes (sampling pool, score ≥ 0.65)**

**ai-toolkit** `/home/damian/Projects/ai-toolkit`
- MeshFleet FLUX config: `config/train_lora_flux_schnell_32_3.yaml` (8k steps, bs4, lr 8e-5, bf16,
  buckets [512,768,1024], lora32, sample_every 500, caption_dropout 0.05, EMA 0.99, adamw8bit,
  assistant_lora `ostris/FLUX.1-schnell-training-adapter`, cache_latents_to_disk)
- Loader: `toolkit/data_loader.py` (folder or `{path:caption}` JSON); entry `run.py <config.yaml>`
- MeshFleet dataset path (HPC): `/hpc/gpfs2/scratch/u/boborzda/meshfleet_aitk_train`

**TRELLIS** `/home/damian/Projects/TRELLIS`
- Pipeline: `dataset_toolkits/{render(150v/512px),voxelize(64³),extract_feature(dinov2_vitl14_reg),encode_ss_latent,encode_latent(slat),render_cond(24v/1024px),build_metadata}.py`
- Templates: `datasets/meshfleetxl_{train,test}/metadata.csv`
- Train: `python train.py --config <json> --output_dir <out> --data_dir <root> --finetune_ckpt <safetensors> --num_gpus N`
- txt config: `configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_finetune.json`
  (ElasticSLatFlowModel 1280ch/28blk, lr 1e-4, bs4 × batch_split4, i_log500/i_sample10k/i_save10k,
  keep_ckpt1, min_aesthetic_score 4.5, text_cond `openai/clip-vit-large-patch14`)
- img config: `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune.json`
  (ElasticSLatFlowModel 1024ch/24blk, lr 5e-5, bs4 × batch_split2, i_log100/i_sample2k/i_save2k,
  image_cond `dinov2_vitl14_reg`, image_size 518)
- blender: `/home/damian/Projects/Diffus3D/blender-3.2.2-linux-x64/blender`

**quality_assessment_library** (this repo)
- Eval entry: `run_meshfleet_eval.py`; config `meshfleet_benchmark/benchmark_config.yaml` + `config_meshfleet.json`
- Manifests: `data/meshfleet/meshfleet_test.csv` (2,137), `meshfleet_train.csv` (10,715)
- Held-out eval: `data/meshfleet/benchmark_data/meshfleet_eval_images/` (232×12)
- Prior ablation outputs: `analysis/track_b/out/per_object_metrics_all.csv`, `analysis/track_b/out/RUN_LOG.md`

## 8. Risks & open items the agent must resolve

1. **TRELLIS real budgets** (not the 1M/100k config defaults) — recover from run logs; STOP if absent.
2. **Real MeshFleet split** — recover; mirror exactly.
3. **`aesthetic_score` provenance** — needed so the TRELLIS `min_aesthetic_score` filter doesn't
   empty the set; reconcile threshold vs values.
4. **Mesh-format coverage** — FBX/OBJ/GLTF/BLEND must render in blender; quantify any drop.
5. **Disk** — TRELLIS renders (150v) + features + latents for 1,620 objects are large; estimate
   before launch; symlink source assets.
6. **Non-deterministic render offset** — pin the seed for reproducibility.
7. **HPC vs local** — the MeshFleet FLUX data path is on `/hpc` scratch; the agent prepares
   commands for the target environment but does not assume the HPC dataset is reachable locally.
8. **CarCaption "3K" vs on-disk counts** — paper says 3,326; disk has 3,064 meshes / 2,633
   captioned / 2,648 aesthetic≥0.65. Sampling 1,620 from the aesthetic≥0.65 pool is unaffected,
   but document the discrepancy in the summary.

## 9. Success criteria

The harness is "done" when `scripts/validate_ablation_ready.py` passes and the agent has emitted,
for each of FLUX / TRELLIS-txt / TRELLIS-img:
- a config referencing the single locked manifest hash,
- a smoke-tested data path,
- and copy-pasteable train → generate → eval commands,

with the controlled rows ready to drop into
`tables/carcaption3k_1620_controlled_ablation_summary.md` once the human-launched runs finish.
