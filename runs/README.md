# runs/ — CarCaption3K-1620 Controlled Ablation Run Directory

This directory holds per-run logs, resolved configs, and metric outputs for the
CarCaption3K-1620 controlled ablation. Each GPU-launched run produces a
subdirectory with a fixed set of required artifacts.

---

## Per-run artifact contract

Every run directory `runs/<run_id>/` **must** contain the following files before
it is considered complete and hand-offable:

| File | Description |
| ---- | ----------- |
| `RUN_LOG.md` | Human-written log following the template below |
| `config_resolved.yaml` or `config_resolved.json` | Full resolved config (no defaults omitted) used for this run |
| `manifest_sha256.txt` | Copy of `manifests/carcaption3k_1620_locked.sha256` — proves which locked manifest drove this run |
| `metrics.json` | Per-object and aggregate metrics output (JSON) |
| `metrics.csv` | Per-object metrics in tabular form (CSV) |
| `COMMANDS.md` | Exact commands used for generation, assembly, and eval (copy from `runs/eval/COMMANDS.md` P-section and fill placeholders) |

---

## RUN_LOG.md template

```markdown
# RUN_LOG — <run_id>

## Identity
- run_id: <run_id>
- variant: <FLUX_TRELLIS_CC1620 | TRELLIS_TXT_CC1620 | TRELLIS_IMG_CC1620>
- date_started: YYYY-MM-DD HH:MM (timezone)
- date_completed: YYYY-MM-DD HH:MM (timezone)

## Hardware
- GPU type: <e.g. L40S>
- GPU count: <N>
- Node: <hostname or cluster job ID>

## Wall-clock
- Generation: Xh Ym
- Assembly: Xm
- Eval: Xh Ym
- Total: Xh Ym

## Code commits
- QA repo (quality_assessment_library): <git rev-parse HEAD>
- TRELLIS repo: <git rev-parse HEAD>
- ai-toolkit repo: <git rev-parse HEAD>  # if FLUX LoRA training

## Manifest
- manifest_sha256: <contents of manifest_sha256.txt>
- manifest_csv: manifests/carcaption3k_1620_locked.csv
- N objects in manifest: 1620

## Exact command
```bash
# Generation (run from $TRELLIS):
<paste exact command>

# Assembly (run from $QA):
<paste exact command>

# Eval (run from $QA, trellis_qa env):
<paste exact command>
```

## Failures / restarts
- <describe any failures, restarts, or skipped objects here; "none" if clean>

## Open items resolved
- OI-1 FLUX LoRA pin: <resolved / n/a> — actual LoRA repo: <path or HF repo>
- OI-2 ss_flow_checkpoint_path: <resolved / n/a>
- OI-3 glb_<sha> assembly: <resolved / n/a>

## Notes
- <any other relevant notes>
```

---

## Deliverables index (handoff §8)

The following artifacts must be present and hand-off-ready at the conclusion of
the CarCaption3K-1620 ablation:

### Manifest + provenance
- `manifests/carcaption3k_1620_locked.csv` — 1620-row locked manifest
- `manifests/carcaption3k_1620_locked_summary.json` — subset-build summary
- `manifests/carcaption3k_1620_locked.sha256` — manifest content hash

### Training configs
- `ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml` — FLUX LoRA training config
- `TRELLIS/dataset_toolkits/datasets/CarCaption3K1620.py` — TRELLIS dataset class
- `TRELLIS/configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json` — TRELLIS SLAT-txt CC1620 config
- `TRELLIS/configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json` — TRELLIS SS-txt CC1620 config
- `TRELLIS/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json` — TRELLIS SLAT-img CC1620 config
- `TRELLIS/configs/generation/ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json` — TRELLIS SS-img CC1620 config

### Run provenance
- `runs/reference_budgets.json` — recovered MeshFleet budgets (Phase 0)
- `runs/RECON.md` — full pipeline recon + readiness report (Phase 0 + Phase 6)
- `runs/eval/COMMANDS.md` — generate → eval command templates for all 3 variants
- `runs/trellis_prep/COMMANDS.md` — TRELLIS data-prep pipeline commands
- `runs/flux_carcaption3k_1620/COMMANDS.md` — FLUX LoRA training commands
- `runs/flux_carcaption3k_1620/manifest_sha256.txt` — manifest hash used by FLUX run
- `runs/trellis_txt_cc1620/manifest_sha256.txt` — manifest hash used by TRELLIS-txt run
- `runs/trellis_img_cc1620/manifest_sha256.txt` — manifest hash used by TRELLIS-img run

### Results
- `tables/carcaption3k_1620_controlled_ablation_summary.csv` — ablation results table
- `tables/carcaption3k_1620_controlled_ablation_summary.md` — ablation results table (Markdown)

---

## Human-launched GPU steps (still to run)

The following steps require GPU access and must be launched by a human operator.
All command templates are in `runs/eval/COMMANDS.md` and `runs/trellis_prep/COMMANDS.md`.

1. **TRELLIS data prep** (train 1420 + val 200 objects): fix trellis_local env first
   (see `runs/trellis_prep/COMMANDS.md` §"CRITICAL — environment caveat").
2. **FLUX LoRA finetune** (8000 steps, rank 32): `ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml`
3. **TRELLIS-txt finetune** (SLAT 100k + SS 100k steps): after fix OI-2
4. **TRELLIS-img finetune** (SLAT 30k + SS 100k steps): after fix OI-2
5. **Generation P0** (FLUX_TRELLIS_CC1620): after OI-1 fix
6. **Generation P1** (TRELLIS_TXT_CC1620): after OI-2 fix
7. **Generation P2** (TRELLIS_IMG_CC1620): after OI-1 + OI-2 fix
8. **Eval** (all 3 variants): `runs/eval/COMMANDS.md` Eval sections

### Open items before generation
| ID   | Affects | Action required |
| ---- | ------- | --------------- |
| OI-1 | P0, P2  | Pin FLUX LoRA path in `evaluate_trellis_prompt_following.py` to the newly trained `flux_carcaption3k_1620_lora32` LoRA. |
| OI-2 | P1, P2  | Add `--ss_flow_checkpoint_path` CLI arg to the generation script; handle `.pt` checkpoint format. |
| OI-3 | All     | Assembly step must copy `glb_<sha>/` PNG views (not `gaussian_<sha>/`) into `data/ablation/gen/<variant>/<sha>/`. |
| ENV  | Data prep | Fix `trellis_local` env (numpy 1.26.4, open3d 0.19.0, correct utils3d) before running the full prep pipeline. |
