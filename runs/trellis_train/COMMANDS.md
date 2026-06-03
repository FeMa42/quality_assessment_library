# CarCaption3K-1620 — TRELLIS finetune train commands (Task 4.2)

All commands are run from `$TRELLIS` = `/home/damian/Projects/TRELLIS`.

TRELLIS training environment: **`trellis_local`**
(`/home/damian/miniconda3/envs/trellis_local/bin/python`).
If this env has been replaced or renamed, substitute the correct env name wherever
`trellis_local` appears below.

---

## PREREQUISITES

### 1. Data pipeline must be complete first

The TRELLIS dataset-preparation pipeline (documented in
`$QA/runs/trellis_prep/COMMANDS.md`) must have run to completion over all
1620 CarCaption3K objects before any training command is executed. Specifically,
the following directories must exist under `$TRELLIS/datasets/`:

```
datasets/carcaption3k_1620_train/
  latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16/<sha>.npz   (1420 files)
  ss_latents/ss_enc_conv3d_16l8_fp16/<sha>.npz                     (1420 files)
  renders_cond/<sha>/transforms.json                                (1420 dirs)
  metadata.csv  (all 1420 rows; flags rendered, voxelized,
                 feature_dinov2_vitl14_reg, ss_latent_*, latent_*, cond_rendered = True)

datasets/carcaption3k_1620_val/
  (same structure, 200 objects)
```

The split is train=1420 / val=200 (~87.7/12.3), controller-locked to N=1620.

### 2. Environment caveats

The `trellis_local` env must satisfy the same dependency constraints as the
data-prep pipeline (see `$QA/requirements_qa.txt` and RECON.md for details):

- numpy == 1.26.4  (numpy 2.x breaks open3d)
- open3d == 0.19.0
- utils3d pinned to the TRELLIS-compatible fork (not PyPI)

These are the same constraints used during smoke-test verification (2026-06-03).

---

## DRY-RUN FIRST (mandatory before full training)

Before committing GPU-hours, run each command with `--tryrun` appended and a
throwaway output dir to validate:
1. The dataloader finds non-zero instances after metadata filtering.
   If it reports 0 instances, `min_aesthetic_score` or the metadata flags are wrong.
2. The `--finetune_ckpt` loads without shape/key errors.

**`--tryrun` requires real latents on disk** — it cannot be run before the data
pipeline is complete.

Example dry-run for one stage (adapt path for each stage):

```bash
cd /home/damian/Projects/TRELLIS
/home/damian/miniconda3/envs/trellis_local/bin/python train.py \
  --config configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json \
  --output_dir outputs/_tryrun_ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620 \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt ./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors \
  --num_gpus 2 \
  --tryrun
```

Run the dry-run for all 4 stages before submitting any full-budget job.

---

## RUN ORDER

For each conditioning family (txt / img), BOTH stages (ss + slat) must be
finetuned independently. Generation/inference uses both finetuned denoisers
together. Recommended order:

1. **ss_txt** and **ss_img** first (either order; they are independent).
2. **slat_txt** and **slat_img** after ss is done (or in parallel on separate nodes).

Budgets:
- `ss_flow_txt`  → 100 000 steps
- `slat_flow_txt` → 100 000 steps
- `ss_flow_img`  → 100 000 steps
- `slat_flow_img` → 30 000 steps  (shorter; img-conditioned slat converges faster)

---

## TRAIN COMMANDS

### Stage 1 — ss_flow_txt (sparse-structure, text-conditioned)

Checkpoint verified on disk: 1 976 327 512 bytes

```bash
cd /home/damian/Projects/TRELLIS
/home/damian/miniconda3/envs/trellis_local/bin/python train.py \
  --config configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json \
  --output_dir outputs/carcaption3k_1620/ss_flow_txt_dit_XL_16l8_fp16 \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt ./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors \
  --num_gpus 2
```

Config: `configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json`
max_steps: 100 000 | min_aesthetic_score: 0.0 | wandb name: ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620

---

### Stage 2 — slat_flow_txt (structured-latent, text-conditioned)

Checkpoint verified on disk: 2 151 142 688 bytes

```bash
cd /home/damian/Projects/TRELLIS
/home/damian/miniconda3/envs/trellis_local/bin/python train.py \
  --config configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json \
  --output_dir outputs/carcaption3k_1620/slat_flow_txt_dit_XL_64l8p2_fp16 \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt ./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16.safetensors \
  --num_gpus 2
```

Config: `configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json`
max_steps: 100 000 | min_aesthetic_score: 0.0 | wandb name: slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620

---

### Stage 3 — ss_flow_img (sparse-structure, image-conditioned)

Checkpoint verified on disk: 1 130 770 840 bytes

```bash
cd /home/damian/Projects/TRELLIS
/home/damian/miniconda3/envs/trellis_local/bin/python train.py \
  --config configs/generation/ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json \
  --output_dir outputs/carcaption3k_1620/ss_flow_img_dit_L_16l8_fp16 \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt ./assets/pretrained_TRELLIS_image_large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors \
  --num_gpus 2
```

Config: `configs/generation/ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json`
max_steps: 100 000 | min_aesthetic_score: 0.0 | wandb name: ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620

---

### Stage 4 — slat_flow_img (structured-latent, image-conditioned)

Checkpoint verified on disk: 1 203 755 136 bytes

```bash
cd /home/damian/Projects/TRELLIS
/home/damian/miniconda3/envs/trellis_local/bin/python train.py \
  --config configs/generation/slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json \
  --output_dir outputs/carcaption3k_1620/slat_flow_img_dit_L_64l8p2_fp16 \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt ./assets/pretrained_TRELLIS_image_large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors \
  --num_gpus 2
```

Config: `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json`
max_steps: 30 000 | min_aesthetic_score: 0.0 | wandb name: slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620

---

## CHECKPOINT PATHS SUMMARY

| Stage        | finetune_ckpt (relative to $TRELLIS)                                                                                                          | Exists on disk |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| ss_flow_txt  | `./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/ss_flow_txt_dit_XL_16l8_fp16.safetensors`       | YES (1.9 GB)   |
| slat_flow_txt| `./assets/pretrained_TRELLIS_txt_xl/snapshots/e0b00432b8e3a8ecee0df806ab1df9f7281f2be4/ckpts/slat_flow_txt_dit_XL_64l8p2_fp16.safetensors`   | YES (2.0 GB)   |
| ss_flow_img  | `./assets/pretrained_TRELLIS_image_large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors`    | YES (1.1 GB)   |
| slat_flow_img| `./assets/pretrained_TRELLIS_image_large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors`| YES (1.1 GB)   |

All 4 checkpoint files confirmed present on disk (verified 2026-06-04).

---

## CONFIG CHANGE SUMMARY (vs base finetune configs)

The `_carcaption3k_1620` configs differ from their `_finetune` bases ONLY in:

| Key                                    | Base value       | CarCaption3K-1620 value                        |
|----------------------------------------|------------------|------------------------------------------------|
| `dataset.args.min_aesthetic_score`     | 4.5              | 0.0  (Gate B: pass all 1620 curated objects)   |
| `eval_dataset.args.min_aesthetic_score`| 4.5              | 0.0  (same)                                    |
| `trainer.args.max_steps`               | varies (see note)| matched budget from reference_budgets.json     |
| `trainer.args.wandb_config.name`       | stage name only  | stage name + `_carcaption3k_1620`              |

Note on base `max_steps`: the `_finetune` base configs had `max_steps=1000000` for
txt stages and `max_steps=100000` for img stages — these are training-script
defaults, not the realized MeshFleet budgets. The realized budgets (recovered from
sweep command.txt + config.json) are in `reference_budgets.json` and are set here.

Note on `ss_flow_img` wandb_config: the base `ss_flow_img_dit_L_16l8_fp16_finetune.json`
had no `wandb_config` key at all. A minimal one (name only) was added to the clone
so runs are trackable. All other wandb fields (project, entity, tags, notes) are
inherited from the existing base configs or will fall back to train.py defaults.
