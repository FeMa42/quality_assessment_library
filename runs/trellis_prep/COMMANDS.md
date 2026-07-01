# CarCaption3K-1620 — TRELLIS data-prep pipeline (Task 3.4)

## SMOKE verified (2026-06-03)

End-to-end smoke test on **one object per mesh format = 4 objects** completed
successfully. The full chain (metadata builder 3.1 + subset module 3.2 +
render-determinism edit 3.3 + all dataset_toolkits stages) produces every
artifact the TRELLIS trainer reads.

Smoke objects (sha[:8] / ext):
- `0038d24d` .glb
- `0044b9fb` .obj
- `0062d77c` .gltf
- `009fe33d` .fbx

### Per-format outcome — ALL 4 FORMATS PASS

| sha[:8] | ext  | rendered (150v) | mesh.ply | voxelized | dinov2 feat | ss_latent | slat latent | cond render (24v) |
|---------|------|-----------------|----------|-----------|-------------|-----------|-------------|-------------------|
| 0038d24d| glb  | yes             | 5.4 MB   | 10630 vox | yes         | yes       | yes         | yes               |
| 0044b9fb| obj  | yes             | 5.1 MB   | 11356 vox | yes         | yes       | yes         | yes               |
| 0062d77c| gltf | yes             | 5.6 MB   | 11562 vox | yes         | yes       | yes         | yes               |
| 009fe33d| fbx  | yes             | 14.7 MB  | 30535 vox | yes         | yes       | yes         | yes               |

Final metadata flags (all 4/4): rendered, voxelized, feature_dinov2_vitl14_reg,
ss_latent_ss_enc_conv3d_16l8_fp16,
latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16, cond_rendered.

Trainer-layout check: `latents/<model>/<sha>.npz` (4), `ss_latents/<model>/<sha>.npz` (4),
`renders_cond/<sha>/transforms.json` (4). Latent shapes verified:
slat `{feats:(N,8), coords:(N,3)}`, ss `{mean:(8,16,16,16)}`,
feature `{indices:(N,3), patchtokens:(N,1024)}`.

**Conclusion: NO mesh format fails to render. The ~466 non-glb objects
(obj 235 / fbx 155 / gltf 76) are all expected to process.** No controller
decision needed on format dropping.

### Disk estimate (measured on 4 objects = 314 MB ≈ 83.5 MB/object)

Full-1620 extrapolation (× 1620/4):

| subdir        | full-1620 |
|---------------|-----------|
| renders (150v, intermediate) | ~57 GB |
| renders_cond (24v @ 1024px)  | ~23 GB |
| features (dinov2 patchtokens)| ~44 GB |
| latents (slat)               | ~0.8 GB |
| ss_latents                   | ~0.2 GB |
| voxels                       | ~0.3 GB |
| **TOTAL (all artifacts)**    | **~132 GB** |

Note: `renders/` (~57 GB) and `features/` (~44 GB) are intermediates consumed by
voxelize/extract_feature/encode_latent. After prep the trainer only needs
latents + ss_latents + renders_cond + voxels + metadata (~24 GB). Prune
`renders/` once latents are built to reclaim ~57 GB.

### CRITICAL — environment caveat (blocked under `trellis_local`)

The smoke test had to run `voxelize`, `extract_feature`, `encode_ss_latent`,
`encode_latent` under the `trellis_printability` conda env (and a manual
voxelize via `hunyuan3d_local`), NOT `trellis_local`, because **`trellis_local`
has drifted from its pinned config** (`TRELLIS/requirements_qa.txt`:
numpy 1.26.4, open3d 0.19.0). Two regressions found:

1. **open3d 0.17.0 + numpy 2.2.6** → `VoxelGrid.create_from_triangle_mesh_within_bounds()`
   SEGFAULTS (exit 139) on every mesh, including a known-good MeshFleet mesh.
   Works fine under numpy 1.x (verified in `hunyuan3d_local`: open3d 0.18.0 / numpy 1.24.4).
2. **wrong `utils3d` package** in `trellis_local`: the installed `utils3d` lacks
   `.io` and `.torch` (it's an unrelated package with only `pctodepthimage.py`),
   so `extract_feature.py`/`encode_*.py` (which call `utils3d.io.read_ply`,
   `utils3d.torch.*`) would fail. The correct `utils3d` is present in
   `trellis_printability` / `trellis_qa` / `trellis2`.

**Before the full run, fix `trellis_local`** to match `requirements_qa.txt`
(downgrade numpy to 1.26.4, install open3d 0.19.0, and install the correct
`utils3d` with `.io`/`.torch`). OR run the full prep under `trellis_printability`
for all stages (it has correct utils3d; voxelize works there ONLY if numpy is
also 1.x — it currently has numpy 2.2.6 so voxelize would segfault there too).
Cleanest: align ONE env to `requirements_qa.txt` and use it for all stages.
The commands below assume that fixed env at the `$TP` path.

---

## Full-1620 prep commands (do NOT run blindly — fix env first)

```bash
TP=/home/damian/miniconda3/envs/trellis_local/bin/python   # <-- must match requirements_qa.txt (numpy 1.26.4 / open3d 0.19.0 / correct utils3d)
QA=/home/damian/Projects/quality_assessment_library
TRELLIS=/home/damian/Projects/TRELLIS
cd "$TRELLIS/dataset_toolkits"
```

### TRAIN set (1420 objects)

```bash
S=carcaption3k_1620_train
D="$TRELLIS/datasets/$S"

# 1. Build TRELLIS metadata.csv + stage mesh symlinks (QA python; pandas)
/home/damian/miniconda3/envs/trellis_qa/bin/python \
  "$QA/scripts/build_trellis_metadata.py" \
  --manifest "$QA/manifests/carcaption3k_1620_locked.csv" --split train --out-dir "$D"

# 2. Pipeline stages (run one at a time; build_metadata after each merges flags)
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP render.py          CarCaption3K1620 --output_dir "$D" --num_views 150 --rank 0 --world_size 1
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP voxelize.py        CarCaption3K1620 --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP extract_feature.py --output_dir "$D" --batch_size 16
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP encode_ss_latent.py --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP encode_latent.py    --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP render_cond.py     CarCaption3K1620 --output_dir "$D" --num_views 24 --rank 0 --world_size 1
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
```

### VAL set (200 objects)

```bash
S=carcaption3k_1620_val
D="$TRELLIS/datasets/$S"

/home/damian/miniconda3/envs/trellis_qa/bin/python \
  "$QA/scripts/build_trellis_metadata.py" \
  --manifest "$QA/manifests/carcaption3k_1620_locked.csv" --split val --out-dir "$D"

$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP render.py          CarCaption3K1620 --output_dir "$D" --num_views 150 --rank 0 --world_size 1
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP voxelize.py        CarCaption3K1620 --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP extract_feature.py --output_dir "$D" --batch_size 16
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP encode_ss_latent.py --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP encode_latent.py    --output_dir "$D"
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
$TP render_cond.py     CarCaption3K1620 --output_dir "$D" --num_views 24 --rank 0 --world_size 1
$TP build_metadata.py  CarCaption3K1620 --output_dir "$D"
```

### Sharding render / render_cond across the 2× L40S

`render.py` and `render_cond.py` accept `--rank`/`--world_size` and slice the
metadata into `[len*rank/world : len*(rank+1)/world]`. Run one process per GPU,
then run `build_metadata.py` once to merge the per-rank `rendered_<rank>.csv` /
`cond_rendered_<rank>.csv` files:

```bash
# render across 2 GPUs (same for render_cond with --num_views 24)
CUDA_VISIBLE_DEVICES=0 $TP render.py CarCaption3K1620 --output_dir "$D" --num_views 150 --rank 0 --world_size 2 &
CUDA_VISIBLE_DEVICES=1 $TP render.py CarCaption3K1620 --output_dir "$D" --num_views 150 --rank 1 --world_size 2 &
wait
$TP build_metadata.py CarCaption3K1620 --output_dir "$D"   # merges rendered_0.csv + rendered_1.csv
```

`extract_feature.py` / `encode_ss_latent.py` / `encode_latent.py` also accept
`--rank/--world_size` for GPU sharding (one process per GPU, then build_metadata
to merge).

### Timing observed (smoke, GPU 0 only)
- render 150v × 4 obj @ 512px CYCLES, 4 workers: ~18 min (~4.5 min/obj)
- render_cond 24v × 4 obj @ 1024px, 4 workers: ~10 min
- extract_feature 4 obj: ~2.2 min; encode_ss_latent 4 obj: ~3 s; encode_latent 4 obj: ~40 s
- Render dominates wall time → shard across both GPUs for the full run.
