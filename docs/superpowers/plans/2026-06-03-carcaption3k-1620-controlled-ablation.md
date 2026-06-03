# CarCaption3K-1620 Controlled Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous prep harness that produces a deterministic 1,620-object CarCaption subset and all per-family adapters/configs (FLUX + TRELLIS txt & img), smoke-tested and sanity-gated, then emits ready-to-run launch commands for size- and budget-matched ablation runs evaluated on the unchanged MeshFleet held-out test set.

**Architecture:** One canonical locked manifest (`manifests/carcaption3k_1620_locked.csv`) is the single source of truth. Thin per-family adapters reshape it into (a) an ai-toolkit FLUX dataset (symlinked PNG+TXT), and (b) TRELLIS `metadata.csv` + a custom `dataset_toolkits` subset module fed through the TRELLIS data pipeline. Evaluation is shared and fixed: `run_meshfleet_eval.py` on the 232-object held-out set. New code is added across three repos; nothing existing is rewritten.

**Tech Stack:** Python 3, pandas, pytest; ai-toolkit (FLUX LoRA, `sd_trainer`); TRELLIS (`dataset_toolkits` + `train.py`, flow-matching); blender-3.2.2 (rendering); the existing `quality_assessment_library` eval pipeline.

**Companion spec:** `docs/superpowers/specs/2026-06-03-carcaption3k-1620-controlled-ablation-design.md`

---

## Conventions used by every task

- **Repo roots** (absolute):
  - `QA=/home/damian/Projects/quality_assessment_library` (orchestration home — run commands from here unless stated)
  - `AITK=/home/damian/Projects/ai-toolkit`
  - `TRELLIS=/home/damian/Projects/TRELLIS`
  - `CCD=/home/damian/Projects/datasets/CarCaptionData`
  - `BLENDER=/home/damian/Projects/Diffus3D/blender-3.2.2-linux-x64/blender`
- **Sampling pool:** `$CCD/car_meshes_trellis_aesthetics_65` (aesthetic≥0.65, ≈2,648 meshes).
- **Captions/previews:** `$CCD/CarCaptionData_ai_toolkit/<sha>.txt` and `<sha>.png`.
- **Determinism:** seed `20260603` everywhere a random choice is made.
- **Tests:** `python -m pytest tests/ablations/<file>::<test> -v`, run from `$QA`.
- **Commit cadence:** commit after each task's tests pass. Branch: work on `meshflee_benchmark` (already checked out). Use `git add <listed files>` only — the working tree has unrelated changes; never `git add -A`.
- **The agent does NOT launch GPU jobs.** Tasks that would run rendering/training/inference instead *emit* the exact command into `runs/<run_id>/COMMANDS.md` and stop.

## ⚠️ Two decision gates that STOP the agent (resolve before the dependent phase)

- **GATE A (object count).** `TRELLIS/datasets/meshfleetxl_train/metadata.csv` = **2,229** rows; `meshfleetxl_test` = **394**. The paper/handoff says MeshFleet = 1,620. These disagree, so "size-matched" is ambiguous. **Phase 0 must surface both numbers and get the user's choice** (keep 1,620 per the handoff, or match the real per-family MeshFleet counts). Do not build the manifest (Phase 1) until resolved. Default if the user is unreachable: **1,620** (the handoff's explicit, repeated requirement), documented as a deliberate size choice.
- **GATE B (aesthetic scale).** TRELLIS configs filter `aesthetic_score ≥ 4.5` on a ~0–10 scale; the `_65` pool uses a 0–1 scorer. Phase 0 must output the concrete `aesthetic_score` source + the `min_aesthetic_score` the ablation configs will use, chosen so the filter **never drops a selected object** (the manifest is the source of truth). Default: set `min_aesthetic_score = 0.0` in the ablation configs and store the 0–1 trellis score in metadata, so all 1,620 pass.

---

## File structure (what gets created/modified)

**In `$QA` (this repo):**
- Create `scripts/build_carcaption3k_1620_subset.py` — manifest builder (pure functions + CLI).
- Create `scripts/build_flux_filtered_dataset.py` — FLUX symlink-dataset adapter.
- Create `scripts/build_trellis_metadata.py` — TRELLIS `metadata.csv` builder + mesh staging.
- Create `scripts/validate_ablation_ready.py` — sanity gate (handoff §7).
- Create `scripts/assemble_ablation_table.py` — results table assembler.
- Create tests under `tests/ablations/`.
- Create `manifests/`, `configs/ablations/`, `runs/`, `tables/` (outputs).
- Create `runs/reference_budgets.json`, `runs/RECON.md` (Phase 0 artifacts).

**In `$AITK`:**
- Create `config/flux_trellis_carcaption3k_1620.yaml` (clone of MeshFleet FLUX config).

**In `$TRELLIS`:**
- Create `dataset_toolkits/datasets/CarCaption3K1620.py` (custom subset module).
- Create `configs/generation/{ss_flow_txt,slat_flow_txt,ss_flow_img,slat_flow_img}_dit_*_carcaption3k_1620.json` (clones with edited budgets/data dirs).

---

## Phase 0 — Reconnaissance & budget recovery

### Task 0.1: Recover MeshFleet per-family budgets and counts

**Files:**
- Create: `runs/reference_budgets.json`
- Create: `runs/RECON.md`

- [ ] **Step 1: Read the recovered TRELLIS run logs**

Run (from `$TRELLIS`):
```bash
for d in outputs/sweeps_slat_flow_txt/sweep_lcqjobfr_20250629_230408 \
         outputs/sweeps_ss_flow_txt/sweep_wqt27r19_20250704_034000 \
         outputs/sweeps_ss_flow_img/sweep_x3va3sda_20250705_011646 \
         outputs/sweeps_slat_flow_img/sweep_uoxlv9op_20250627_151924; do
  echo "=== $d ==="
  cat "$d/command.txt" 2>/dev/null
  grep -o '"max_steps"[^,]*' "$d/config.json" 2>/dev/null | head -1
  ls "$d/ckpts/" 2>/dev/null | grep denoiser_step | tail -1
done
```
Expected: each prints a `train.py …` command, its `max_steps`, and the highest `denoiser_step*.pt`.
Record the `--data_dir`, `--eval_data_dir`, `--finetune_ckpt`, `--num_gpus`, and `max_steps` from each `command.txt`. If any `command.txt` is missing, derive the realized budget from the highest `denoiser_step*.pt` filename.

- [ ] **Step 2: Recover FLUX budget and object count**

Run:
```bash
grep -E 'steps|folder_path|hf_repo_id|linear:' "$AITK/config/train_lora_flux_schnell_32_3.yaml"
# object counts actually used:
wc -l "$TRELLIS/datasets/meshfleetxl_train/metadata.csv" "$TRELLIS/datasets/meshfleetxl_test/metadata.csv"
ls "$CCD/car_meshes_trellis_aesthetics_65" | wc -l
ls "$CCD/CarCaptionData_ai_toolkit"/*.txt | wc -l
```
Expected: FLUX `steps: 8000`; meshfleetxl train=2230 lines (2229 objects), test=395 (394). Note the FLUX MeshFleet object count is on HPC (`/hpc/.../meshfleet_aitk_train`) and not countable locally — record as "unknown locally; confirm with user".

- [ ] **Step 3: Determine the base finetune checkpoints**

From each `command.txt`, copy the `--finetune_ckpt` path. Verify it exists:
```bash
# example — use the real paths from command.txt:
ls -la <finetune_ckpt_path_for_each_stage>
```
Expected: the four base TRELLIS denoiser checkpoints (ss-txt, slat-txt, ss-img, slat-img) exist. If a path points at an HF cache id rather than a file, record the id.

- [ ] **Step 4: Write `runs/reference_budgets.json`**

Write the recovered facts (fill the real values from Steps 1–3):
```json
{
  "flux":            {"steps": 8000, "config": "config/train_lora_flux_schnell_32_3.yaml", "lr": 1e-4, "lora_linear": 32, "object_count_local": null},
  "trellis_ss_txt":  {"max_steps": 100000, "data_dir": "<from command.txt>", "eval_data_dir": "<...>", "finetune_ckpt": "<...>", "num_gpus": null},
  "trellis_slat_txt":{"max_steps": 100000, "data_dir": "<...>", "eval_data_dir": "<...>", "finetune_ckpt": "<...>", "num_gpus": null},
  "trellis_ss_img":  {"max_steps": 100000, "data_dir": "<...>", "eval_data_dir": "<...>", "finetune_ckpt": "<...>", "num_gpus": null},
  "trellis_slat_img":{"max_steps": 30000,  "data_dir": "<...>", "eval_data_dir": "<...>", "finetune_ckpt": "<...>", "num_gpus": null},
  "meshfleet_counts": {"trellis_train": 2229, "trellis_val": 394, "flux_local": "unknown", "paper_curated": 1620},
  "split_policy": {"source": "separate dirs (meshfleetxl_train / meshfleetxl_test)", "ratio_note": "2229/394 ≈ 85/15"}
}
```

- [ ] **Step 5: Resolve GATE A and GATE B; write `runs/RECON.md`**

In `runs/RECON.md` document: the four recovered TRELLIS budgets and their source paths; the FLUX budget; the object-count conflict (2,229 vs 1,620) with an explicit **GATE A** question for the user; the aesthetic-scale finding with the **GATE B** chosen value (default `min_aesthetic_score=0.0`, store 0–1 trellis score). State the chosen subset size (`N`) and split (`train`/`val`) to be used downstream.

- [ ] **Step 6: STOP for GATE A confirmation**

Surface GATE A to the user. Do not proceed to Phase 1 until the subset size `N` and split are confirmed (default `N=1620`, `train=1420`, `val=200`).

- [ ] **Step 7: Commit**

```bash
git add runs/reference_budgets.json runs/RECON.md
git commit -m "chore(ablation): recover MeshFleet budgets and document gates A/B"
```

### Task 0.2: Locate the MeshFleet generation/inference pipelines

The eval (Phase 5) compares generated outputs to the held-out GT. To emit *matched* generation commands, find the scripts MeshFleet used to (a) run FLUX inference + zero-shot TRELLIS into 12-view PNGs, and (b) run finetuned TRELLIS (ss+slat) inference. `benchmark_config.yaml` already references a `generated_flux_lora_maxR` folder, so the FLUX+TRELLIS generation pipeline exists.

**Files:**
- Append to: `runs/RECON.md`

- [ ] **Step 1: Find the generation scripts**

Run:
```bash
grep -rIl "prompt_alignment_images\|generated_flux\|flux_lora\|render_views\|000.png" \
  "$TRELLIS" "$QA" 2>/dev/null | grep -vE "/(outputs|datasets|node_modules)/" | head -40
ls "$TRELLIS" | grep -iE "infer|generate|sample|pipeline|demo"
cat "$QA/analysis/track_b/out/RUN_LOG.md" 2>/dev/null | sed -n '1,80p'
```
Expected: identify (1) the FLUX→TRELLIS generation entrypoint that writes `<sha>/000.png..011.png`, (2) the TRELLIS image/text inference entrypoint, and (3) the exact preprocessing/eval flags used for the existing `TRELLIS`/`TRELLIS_FT` rows (from `RUN_LOG.md`).

- [ ] **Step 2: Record the generation entrypoints**

Append to `runs/RECON.md` a "Generation pipelines" section: for FLUX+TRELLIS, TRELLIS-txt, TRELLIS-img — the script path, how it loads a finetuned checkpoint/LoRA, the output folder layout, and the render/camera settings (must match the 12 views: azimuth 0–330°, elevation 90°). If a generation script cannot be found for a family, **flag it** as a blocker the user must resolve before that family's row can be produced.

- [ ] **Step 3: Commit**

```bash
git add runs/RECON.md
git commit -m "docs(ablation): locate MeshFleet generation/inference pipelines"
```

---

## Phase 1 — Lock the canonical subset

### Task 1.1: Manifest builder — candidate discovery & filtering (TDD)

**Files:**
- Create: `scripts/build_carcaption3k_1620_subset.py`
- Test: `tests/ablations/test_build_subset.py`

- [ ] **Step 1: Write failing tests for discovery/filter/sample**

```python
# tests/ablations/test_build_subset.py
import os, json, csv, importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "build_subset",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_carcaption3k_1620_subset.py"))
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)

def _make_pool(tmp_path, shas, with_caption=True, with_preview=True, ext=".glb"):
    mesh = tmp_path / "meshes"; aitk = tmp_path / "aitk"; ren = tmp_path / "ren"
    for d in (mesh, aitk, ren): d.mkdir(exist_ok=True)
    for s in shas:
        (mesh / f"{s}{ext}").write_text("m")
        if with_caption: (aitk / f"{s}.txt").write_text(f"a car {s}")
        if with_preview: (aitk / f"{s}.png").write_bytes(b"\x89PNG")
        (ren / f"{s}.zip").write_bytes(b"PK")
    return str(mesh), str(aitk), str(ren)

def test_discover_finds_complete_objects(tmp_path):
    shas = [f"{i:064x}" for i in range(5)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    assert {c["sha256"] for c in cands} == set(shas)
    assert all(c["caption"].startswith("a car") for c in cands)

def test_filter_excludes_missing_caption_and_overlap(tmp_path):
    shas = [f"{i:064x}" for i in range(5)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    # remove caption for one object
    os.remove(os.path.join(aitk, f"{shas[0]}.txt"))
    cands = bs.discover_candidates(mesh, aitk, ren)
    included, excluded = bs.filter_candidates(cands, meshfleet_test_shas={shas[1]})
    inc = {c["sha256"] for c in included}
    assert shas[0] not in inc           # missing caption excluded
    assert shas[1] not in inc           # meshfleet overlap excluded
    assert len(inc) == 3

def test_sample_is_deterministic_and_sized(tmp_path):
    shas = [f"{i:064x}" for i in range(100)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    inc, _ = bs.filter_candidates(cands, meshfleet_test_shas=set())
    a = [c["sha256"] for c in bs.sample_locked(inc, n=10, seed=20260603)]
    b = [c["sha256"] for c in bs.sample_locked(inc, n=10, seed=20260603)]
    assert a == b and len(a) == 10

def test_split_counts(tmp_path):
    shas = [f"{i:064x}" for i in range(20)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    inc, _ = bs.filter_candidates(cands, meshfleet_test_shas=set())
    sel = bs.sample_locked(inc, n=20, seed=20260603)
    rows = bs.assign_splits(sel, n_val=5, seed=20260603)
    splits = [r["split"] for r in rows]
    assert splits.count("val") == 5 and splits.count("train") == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ablations/test_build_subset.py -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement the builder**

```python
# scripts/build_carcaption3k_1620_subset.py
"""Build the locked CarCaption3K-1620 subset manifest (handoff §3, §5.1)."""
import argparse, csv, glob, hashlib, json, os, random

MESH_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".blend")

def discover_candidates(mesh_dir, aitk_dir, renders_dir):
    """One record per mesh that has a sibling caption+preview. Caption text is read."""
    cands = []
    for path in sorted(glob.glob(os.path.join(mesh_dir, "*"))):
        ext = os.path.splitext(path)[1].lower()
        if ext not in MESH_EXTS:
            continue
        sha = os.path.splitext(os.path.basename(path))[0]
        cap_path = os.path.join(aitk_dir, f"{sha}.txt")
        png_path = os.path.join(aitk_dir, f"{sha}.png")
        zip_path = os.path.join(renders_dir, f"{sha}.zip")
        caption = ""
        if os.path.exists(cap_path):
            with open(cap_path, encoding="utf-8") as f:
                caption = f.read().strip()
        cands.append({
            "sha256": sha, "asset_path": os.path.abspath(path), "mesh_ext": ext,
            "caption_path": cap_path, "preview_path": png_path,
            "render_path": zip_path if os.path.exists(zip_path) else "",
            "caption": caption,
        })
    return cands

def filter_candidates(cands, meshfleet_test_shas):
    """Inclusion rules: mesh readable + caption non-empty + preview exists + not meshfleet-test + unique."""
    included, excluded, seen = [], [], set()
    for c in cands:
        reasons = []
        if not os.path.exists(c["asset_path"]): reasons.append("mesh_missing")
        if not c["caption"]: reasons.append("no_caption")
        if not os.path.exists(c["preview_path"]): reasons.append("no_preview")
        if c["sha256"] in meshfleet_test_shas: reasons.append("meshfleet_test_overlap")
        if c["sha256"] in seen: reasons.append("duplicate")
        seen.add(c["sha256"])
        if reasons:
            excluded.append({**c, "selection_notes": ";".join(reasons)})
        else:
            included.append(c)
    return included, excluded

def sample_locked(included, n, seed):
    """Sort by sha256 (stable key), then deterministically sample n with seed."""
    pop = sorted(included, key=lambda c: c["sha256"])
    if len(pop) < n:
        raise SystemExit(f"FAIL: only {len(pop)} candidates pass strict filter (<{n}). Stop before training (handoff §3).")
    rng = random.Random(seed)
    return rng.sample(pop, n)

def assign_splits(selected, n_val, seed):
    """Deterministic val holdout within the locked set."""
    pop = sorted(selected, key=lambda c: c["sha256"])
    rng = random.Random(seed + 1)
    val = set(c["sha256"] for c in rng.sample(pop, n_val))
    rows = []
    for c in pop:
        rows.append({**c, "split": "val" if c["sha256"] in val else "train"})
    return rows

def load_meshfleet_test_shas(meshfleet_csv):
    shas = set()
    if meshfleet_csv and os.path.exists(meshfleet_csv):
        with open(meshfleet_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("sha256"):
                    shas.add(row["sha256"].strip())
    return shas

def write_manifest(rows, out_csv, seed):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = ["object_id", "sha256", "source_dataset", "asset_path", "render_path",
            "caption", "split", "selection_seed", "selection_notes"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({
                "object_id": r["sha256"], "sha256": r["sha256"], "source_dataset": "CarCaption3K",
                "asset_path": r["asset_path"], "render_path": r.get("render_path", ""),
                "caption": r["caption"], "split": r["split"],
                "selection_seed": seed, "selection_notes": r.get("selection_notes", ""),
            })

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="/home/damian/Projects/datasets/CarCaptionData/car_meshes_trellis_aesthetics_65")
    ap.add_argument("--aitk-dir", default="/home/damian/Projects/datasets/CarCaptionData/CarCaptionData_ai_toolkit")
    ap.add_argument("--renders-dir", default="/home/damian/Projects/datasets/CarCaptionData/CarCaptionData_renders")
    ap.add_argument("--meshfleet-test-csv", default="data/meshfleet/meshfleet_test.csv")
    ap.add_argument("--n", type=int, default=1620)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--out-csv", default="manifests/carcaption3k_1620_locked.csv")
    ap.add_argument("--out-summary", default="manifests/carcaption3k_1620_locked_summary.json")
    ap.add_argument("--timestamp", default="", help="ISO timestamp passed in (scripts cannot call clock here)")
    a = ap.parse_args()

    cands = discover_candidates(a.pool_dir, a.aitk_dir, a.renders_dir)
    mf = load_meshfleet_test_shas(a.meshfleet_test_csv)
    included, excluded = filter_candidates(cands, mf)
    selected = sample_locked(included, a.n, a.seed)
    rows = assign_splits(selected, a.n_val, a.seed)
    write_manifest(rows, a.out_csv, a.seed)

    summary = {
        "seed": a.seed, "n": a.n, "n_val": a.n_val,
        "counts": {
            "pool_meshes": len(cands), "after_strict_filter": len(included),
            "excluded": len(excluded), "selected": len(rows),
            "train": sum(r["split"] == "train" for r in rows),
            "val": sum(r["split"] == "val" for r in rows),
        },
        "mesh_ext_breakdown": {ext: sum(1 for r in rows if r["asset_path"].lower().endswith(ext)) for ext in MESH_EXTS},
        "paths": {"pool": a.pool_dir, "aitk": a.aitk_dir, "renders": a.renders_dir, "meshfleet_test": a.meshfleet_test_csv},
        "timestamp": a.timestamp,
        "manifest_sha256": sha256_of_file(a.out_csv),
    }
    os.makedirs(os.path.dirname(a.out_summary), exist_ok=True)
    with open(a.out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ablations/test_build_subset.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_carcaption3k_1620_subset.py tests/ablations/test_build_subset.py
git commit -m "feat(ablation): deterministic CarCaption3K-1620 subset builder"
```

### Task 1.2: Generate the locked manifest

**Files:**
- Create: `manifests/carcaption3k_1620_locked.csv`
- Create: `manifests/carcaption3k_1620_locked_summary.json`

- [ ] **Step 1: Run the builder (use N/split confirmed in Phase 0)**

Run (from `$QA`, pass a timestamp since scripts can't read the clock):
```bash
python scripts/build_carcaption3k_1620_subset.py \
  --n 1620 --n-val 200 --seed 20260603 \
  --timestamp "$(date -Iseconds)"
```
Expected: prints summary JSON with `selected: 1620`, `train: 1420`, `val: 200`, a `mesh_ext_breakdown`, and a `manifest_sha256`.

- [ ] **Step 2: Verify the gate**

Run:
```bash
python - <<'PY'
import csv
rows=list(csv.DictReader(open("manifests/carcaption3k_1620_locked.csv")))
shas=[r["sha256"] for r in rows]
assert len(rows)==1620, len(rows)
assert len(set(shas))==1620, "non-unique sha"
import os
assert all(os.path.exists(r["asset_path"]) for r in rows), "missing asset"
print("OK", len(rows), "unique, all assets exist")
PY
```
Expected: `OK 1620 unique, all assets exist`. If <1,620 candidates passed (builder raised SystemExit), **STOP** and report (handoff §3).

- [ ] **Step 3: Record the manifest hash and commit**

```bash
sha256sum manifests/carcaption3k_1620_locked.csv | awk '{print $1}' > manifests/carcaption3k_1620_locked.sha256
git add manifests/carcaption3k_1620_locked.csv manifests/carcaption3k_1620_locked_summary.json manifests/carcaption3k_1620_locked.sha256
git commit -m "feat(ablation): lock CarCaption3K-1620 manifest (seed 20260603)"
```

---

## Phase 2 — FLUX (P0) prep

### Task 2.1: FLUX dataset adapter (TDD)

**Files:**
- Create: `scripts/build_flux_filtered_dataset.py`
- Test: `tests/ablations/test_flux_adapter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ablations/test_flux_adapter.py
import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "flux_adapter",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_flux_filtered_dataset.py"))
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)

def _manifest(tmp_path, shas, splits):
    src = tmp_path / "aitk"; src.mkdir()
    for s in shas:
        (src / f"{s}.png").write_bytes(b"\x89PNG"); (src / f"{s}.txt").write_text("cap "+s)
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256","split"]); w.writeheader()
        for s, sp in zip(shas, splits): w.writerow({"sha256": s, "split": sp})
    return str(p), str(src)

def test_builds_symlinks_for_selected_split(tmp_path):
    shas=[f"{i:064x}" for i in range(4)]
    man, src = _manifest(tmp_path, shas, ["train","train","val","train"])
    out = tmp_path / "flux_ds"
    n = fa.build_symlink_dataset(man, src, str(out), splits=("train",))
    assert n == 3
    assert os.path.exists(out / f"{shas[0]}.png") and os.path.exists(out / f"{shas[0]}.txt")
    assert not os.path.exists(out / f"{shas[2]}.png")   # val excluded
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/ablations/test_flux_adapter.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# scripts/build_flux_filtered_dataset.py
"""Materialize a filtered ai-toolkit FLUX dataset (symlinked <sha>.png + <sha>.txt) from the locked manifest."""
import argparse, csv, os

def build_symlink_dataset(manifest_csv, src_aitk_dir, out_dir, splits=("train", "val")):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") and row["split"] not in splits:
                continue
            sha = row["sha256"]
            for ext in (".png", ".txt"):
                src = os.path.join(src_aitk_dir, f"{sha}{ext}")
                dst = os.path.join(out_dir, f"{sha}{ext}")
                if not os.path.exists(src):
                    raise SystemExit(f"FAIL: missing FLUX source {src}")
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(src), dst)
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/carcaption3k_1620_locked.csv")
    ap.add_argument("--src", default="/home/damian/Projects/datasets/CarCaptionData/CarCaptionData_ai_toolkit")
    ap.add_argument("--out", default="data/ablation/carcaption3k_1620_flux_train")
    ap.add_argument("--splits", default="train,val", help="comma list; FLUX uses all 1620 by default")
    a = ap.parse_args()
    n = build_symlink_dataset(a.manifest, a.src, a.out, tuple(a.splits.split(",")))
    print(f"OK linked {n} FLUX objects into {a.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/ablations/test_flux_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Build the real FLUX dataset and commit**

Run (FLUX uses the full 1,620 — see GATE A note; pass both splits):
```bash
python scripts/build_flux_filtered_dataset.py --splits train,val
ls data/ablation/carcaption3k_1620_flux_train/*.png | wc -l   # expect 1620
```
Expected: `OK linked 1620 …`; 1620 png symlinks.
```bash
git add scripts/build_flux_filtered_dataset.py tests/ablations/test_flux_adapter.py
git commit -m "feat(ablation): FLUX filtered-dataset adapter"
```

### Task 2.2: FLUX ablation config (clone + edit)

**Files:**
- Create: `$AITK/config/flux_trellis_carcaption3k_1620.yaml`

- [ ] **Step 1: Clone the MeshFleet FLUX config**

Run:
```bash
cp "$AITK/config/train_lora_flux_schnell_32_3.yaml" "$AITK/config/flux_trellis_carcaption3k_1620.yaml"
```

- [ ] **Step 2: Edit only the ablation-specific keys**

Edit `$AITK/config/flux_trellis_carcaption3k_1620.yaml` so exactly these change (everything else — `steps: 8000`, `batch_size: 4`, `lr: 1e-4`, `linear: 32`, resolution buckets, sampler — stays identical to MeshFleet):
- `config.name` → `"flux_carcaption3k_1620_lora32"`
- `save.hf_repo_id` → `"DamianBoborzi/flux_carcaption3k_1620_lora32"`
- `datasets[0].folder_path` → `"/home/damian/Projects/quality_assessment_library/data/ablation/carcaption3k_1620_flux_train"`

- [ ] **Step 3: Verify the diff is minimal**

Run:
```bash
diff "$AITK/config/train_lora_flux_schnell_32_3.yaml" "$AITK/config/flux_trellis_carcaption3k_1620.yaml"
```
Expected: only the three lines above differ. Confirm `steps: 8000` and `lr: 1e-4` are unchanged.

- [ ] **Step 4: Dataloader dry-run (no training)**

Run (delete any stale size cache first):
```bash
rm -f data/ablation/carcaption3k_1620_flux_train/.aitk_size.json
cd "$AITK" && python - <<'PY'
import os
d="/home/damian/Projects/quality_assessment_library/data/ablation/carcaption3k_1620_flux_train"
pngs=[f for f in os.listdir(d) if f.endswith(".png")]
txts=[f for f in os.listdir(d) if f.endswith(".txt")]
assert len(pngs)==len(txts)==1620, (len(pngs), len(txts))
# every png has a sibling caption with content
for p in pngs[:50]:
    t=os.path.join(d, p[:-4]+".txt"); assert os.path.getsize(t)>0
print("OK FLUX dataset has 1620 png+txt pairs with non-empty captions")
PY
```
Expected: `OK FLUX dataset has 1620 png+txt pairs …`.

- [ ] **Step 5: Emit the FLUX launch command (do NOT run)**

Append to `$QA/runs/flux_carcaption3k_1620/COMMANDS.md` (create dir):
```text
# FLUX finetune (run on a GPU box, ~24GB+ VRAM)
cd /home/damian/Projects/ai-toolkit
python run.py config/flux_trellis_carcaption3k_1620.yaml
# Output LoRA -> ai-toolkit/output/flux_carcaption3k_1620_lora32/
```

- [ ] **Step 6: Commit**

```bash
cd "$QA"
git add runs/flux_carcaption3k_1620/COMMANDS.md
git -C "$AITK" add config/flux_trellis_carcaption3k_1620.yaml
git commit -m "feat(ablation): FLUX CarCaption3K-1620 config + launch command"
# Note: ai-toolkit is a separate repo; commit there separately if it is version-controlled.
```

---

## Phase 3 — TRELLIS data pipeline (de-risking phase)

### Task 3.1: TRELLIS `metadata.csv` builder + mesh staging (TDD)

**Files:**
- Create: `scripts/build_trellis_metadata.py`
- Test: `tests/ablations/test_trellis_metadata.py`

The output mirrors `TRELLIS/datasets/meshfleetxl_test/metadata.csv` (13 columns; `captions` is a JSON list; `local_path` = `./raw/<sha>.<ext>`; boolean flags start False).

- [ ] **Step 1: Write failing test**

```python
# tests/ablations/test_trellis_metadata.py
import os, csv, json, importlib.util
spec = importlib.util.spec_from_file_location(
    "trellis_meta",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_trellis_metadata.py"))
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

def _manifest(tmp_path):
    shas=[f"{i:064x}" for i in range(3)]
    meshes=tmp_path/"meshes"; meshes.mkdir()
    rows=[]
    for i,s in enumerate(shas):
        ext=".glb"
        (meshes/f"{s}{ext}").write_text("m")
        rows.append({"sha256":s,"asset_path":str(meshes/f"{s}{ext}"),
                     "caption":f'A car number {i}',"split":"train" if i<2 else "val"})
    p=tmp_path/"m.csv"
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sha256","asset_path","caption","split"]); w.writeheader(); w.writerows(rows)
    return str(p), shas

def test_metadata_schema_and_caption_json(tmp_path):
    man, shas = _manifest(tmp_path)
    out=tmp_path/"trellis_train"
    df = tm.build_metadata_df(man, split="train")
    cols=list(df.columns)
    for c in ["sha256","file_identifier","aesthetic_score","captions","local_path",
              "rendered","voxelized","num_voxels","cond_rendered",
              "feature_dinov2_vitl14_reg","ss_latent_ss_enc_conv3d_16l8_fp16",
              "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"]:
        assert c in cols, c
    assert len(df)==2   # only train rows
    cap=json.loads(df.iloc[0]["captions"]); assert isinstance(cap,list) and cap[0].startswith("A car")
    assert df.iloc[0]["local_path"].startswith("./raw/") and df.iloc[0]["local_path"].endswith(".glb")
    assert bool(df.iloc[0]["rendered"]) is False

def test_stage_meshes_symlinks(tmp_path):
    man, shas = _manifest(tmp_path)
    out=tmp_path/"trellis_train"
    df = tm.build_metadata_df(man, split="train")
    tm.stage_meshes(df, str(out))
    assert os.path.islink(out/"raw"/f"{shas[0]}.glb")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/ablations/test_trellis_metadata.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# scripts/build_trellis_metadata.py
"""Build a TRELLIS-format metadata.csv (mirrors meshfleetxl_test schema) and stage meshes for a split."""
import argparse, csv, json, os
import pandas as pd

BOOL_COLS = ["rendered","voxelized","cond_rendered","feature_dinov2_vitl14_reg",
             "ss_latent_ss_enc_conv3d_16l8_fp16",
             "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16","eval_rendered"]
COLS = ["sha256","file_identifier","aesthetic_score","captions","local_path",
        "rendered","voxelized","num_voxels","cond_rendered",
        "feature_dinov2_vitl14_reg","ss_latent_ss_enc_conv3d_16l8_fp16",
        "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16","eval_rendered"]

def build_metadata_df(manifest_csv, split, aesthetic_score=0.0):
    recs = []
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") and row["split"] != split:
                continue
            sha = row["sha256"]; ext = os.path.splitext(row["asset_path"])[1].lower()
            recs.append({
                "sha256": sha,
                "file_identifier": row.get("asset_path", sha),
                "aesthetic_score": aesthetic_score,
                "captions": json.dumps([row["caption"]]),
                "local_path": f"./raw/{sha}{ext}",
                "num_voxels": "",
                **{c: False for c in BOOL_COLS},
            })
    df = pd.DataFrame.from_records(recs)[COLS]
    return df

def stage_meshes(df, out_dir, manifest_csv=None):
    """Symlink each mesh into <out_dir>/raw/<sha>.<ext>. asset_path resolved from manifest."""
    raw = os.path.join(out_dir, "raw"); os.makedirs(raw, exist_ok=True)
    src_by_sha = {}
    if manifest_csv:
        with open(manifest_csv, newline="") as f:
            for row in csv.DictReader(f):
                src_by_sha[row["sha256"]] = row["asset_path"]
    for _, r in df.iterrows():
        sha = r["sha256"]; dst = os.path.join(out_dir, r["local_path"].lstrip("./"))
        src = src_by_sha.get(sha)
        if src is None:   # test path: derive from local_path sibling if present
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/carcaption3k_1620_locked.csv")
    ap.add_argument("--split", required=True, choices=["train","val"])
    ap.add_argument("--out-dir", required=True, help="e.g. /home/damian/Projects/TRELLIS/datasets/carcaption3k_1620_train")
    ap.add_argument("--aesthetic-score", type=float, default=0.0, help="GATE B: value stored so min_aesthetic_score=0 passes all")
    a = ap.parse_args()
    df = build_metadata_df(a.manifest, a.split, a.aesthetic_score)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, "metadata.csv"), index=False)
    stage_meshes(df, a.out_dir, a.manifest)
    print(f"OK wrote {len(df)} rows to {a.out_dir}/metadata.csv and staged meshes")

if __name__ == "__main__":
    main()
```

> **Note on `stage_meshes` in the test vs CLI:** the unit test calls `stage_meshes(df, out)` without a manifest; pass `manifest_csv` in production (the CLI does). Update the test's `stage_meshes` call to `tm.stage_meshes(df, str(out), man)` so the symlink source resolves.

- [ ] **Step 4: Fix the test call and run to pass**

Edit `test_stage_meshes_symlinks` to call `tm.stage_meshes(df, str(out), man)`.
Run: `python -m pytest tests/ablations/test_trellis_metadata.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Build the train/val TRELLIS datasets**

Run:
```bash
python scripts/build_trellis_metadata.py --split train --out-dir "$TRELLIS/datasets/carcaption3k_1620_train"
python scripts/build_trellis_metadata.py --split val   --out-dir "$TRELLIS/datasets/carcaption3k_1620_val"
wc -l "$TRELLIS/datasets/carcaption3k_1620_train/metadata.csv"   # expect 1421 (1420 + header)
ls "$TRELLIS/datasets/carcaption3k_1620_train/raw" | wc -l        # expect 1420 symlinks
```
Expected: train metadata 1421 lines, 1420 raw symlinks; val 201 lines / 200 symlinks.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_trellis_metadata.py tests/ablations/test_trellis_metadata.py
git commit -m "feat(ablation): TRELLIS metadata builder + mesh staging"
```

### Task 3.2: TRELLIS custom subset module

**Files:**
- Create: `$TRELLIS/dataset_toolkits/datasets/CarCaption3K1620.py`

This module makes `importlib.import_module('datasets.CarCaption3K1620')` work for the stage scripts (render/voxelize/render_cond/build_metadata take a SUBSET positional). Because we pre-build `metadata.csv` and pre-stage meshes, `download` is a no-op and `get_metadata` reads the pre-built file.

- [ ] **Step 1: Read the real call sites before writing**

Run and read fully (the verbatim CLI is known; confirm how each calls the module):
```bash
sed -n '1,60p' "$TRELLIS/dataset_toolkits/build_metadata.py"
sed -n '1,80p' "$TRELLIS/dataset_toolkits/render.py"
grep -n "get_metadata\|foreach_instance\|download\|metadata.csv" "$TRELLIS/dataset_toolkits/"*.py
```
Expected: confirms whether `get_metadata(**kwargs)` receives `output_dir` and whether stages read `<output_dir>/metadata.csv` directly. Adjust the module below if a call site differs.

- [ ] **Step 2: Write the module**

```python
# TRELLIS/dataset_toolkits/datasets/CarCaption3K1620.py
"""Custom TRELLIS subset for the CarCaption3K-1620 controlled ablation.

Meshes are pre-staged under <output_dir>/raw and metadata.csv is pre-built by
quality_assessment_library/scripts/build_trellis_metadata.py. So get_metadata reads
the existing metadata.csv and download is a no-op.
"""
import os
import argparse
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    # No extra args; metadata.csv lives in --output_dir.
    pass


def get_metadata(output_dir=None, **kwargs):
    assert output_dir, "CarCaption3K1620 requires --output_dir containing a prebuilt metadata.csv"
    meta_path = os.path.join(output_dir, "metadata.csv")
    assert os.path.exists(meta_path), f"missing {meta_path} (run build_trellis_metadata.py first)"
    return pd.read_csv(meta_path)


def download(metadata, output_dir, **kwargs):
    # Meshes already symlinked under raw/ by build_trellis_metadata.py.
    return metadata[["sha256", "local_path"]].copy()


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects"):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    records = []
    metadata = metadata.to_dict("records")
    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(total=len(metadata), desc=desc) as pbar:
        def worker(m):
            try:
                file = os.path.join(output_dir, m["local_path"])
                rec = func(file, m["sha256"])
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                print(f"Error processing {m['sha256']}: {e}")
            finally:
                pbar.update()
        ex.map(worker, metadata)
        ex.shutdown(wait=True)
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 3: Verify it imports**

Run:
```bash
cd "$TRELLIS/dataset_toolkits" && python -c "import importlib; m=importlib.import_module('datasets.CarCaption3K1620'); print('OK', [f for f in dir(m) if not f.startswith('_')])"
```
Expected: `OK ['add_args', 'download', 'foreach_instance', 'get_metadata', ...]`.

- [ ] **Step 4: Commit (in TRELLIS repo)**

```bash
git -C "$TRELLIS" add dataset_toolkits/datasets/CarCaption3K1620.py
git -C "$TRELLIS" commit -m "feat: CarCaption3K-1620 ablation subset module"
```

### Task 3.3: Pin render determinism

**Files:**
- Modify: `$TRELLIS/dataset_toolkits/render.py` (the `offset` line)
- Modify: `$TRELLIS/dataset_toolkits/render_cond.py` (the `offset` line)

- [ ] **Step 1: Make the Hammersley offset seedable**

In both files, replace the line `offset = (np.random.rand(), np.random.rand())` inside `_render`/`_render_cond` with a per-object deterministic offset:
```python
    import hashlib
    _seed = int(hashlib.sha256(sha256.encode()).hexdigest(), 16) % (2**32)
    _rng = np.random.RandomState(_seed)
    offset = (_rng.rand(), _rng.rand())
```
This makes renders reproducible (same per-object camera offset every run) without changing view count or distribution. Document the change in `runs/RECON.md`.

- [ ] **Step 2: Verify syntax**

Run: `cd "$TRELLIS/dataset_toolkits" && python -c "import ast; ast.parse(open('render.py').read()); ast.parse(open('render_cond.py').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Commit (in TRELLIS repo)**

```bash
git -C "$TRELLIS" add dataset_toolkits/render.py dataset_toolkits/render_cond.py
git -C "$TRELLIS" commit -m "feat: deterministic per-object render offset for ablation reproducibility"
```

### Task 3.4: End-to-end smoke test on 2–3 objects (critical de-risk)

**Files:**
- Create: `$TRELLIS/datasets/carcaption3k_1620_smoke/` (temporary, gitignored)

- [ ] **Step 1: Build a 3-object smoke dataset (mixed formats if present)**

Run:
```bash
python - <<'PY'
import csv, json, os, subprocess
rows=list(csv.DictReader(open("manifests/carcaption3k_1620_locked.csv")))
# pick up to 3 objects covering distinct mesh extensions
seen=set(); pick=[]
for r in sorted(rows, key=lambda r:r["sha256"]):
    ext=os.path.splitext(r["asset_path"])[1].lower()
    if ext not in seen: seen.add(ext); pick.append(r)
    if len(pick)>=3: break
out="manifests/_smoke.csv"
with open(out,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader()
    for r in pick: w.writerow(r)
print("smoke objects:", [(p["sha256"][:8], os.path.splitext(p["asset_path"])[1]) for p in pick])
PY
python scripts/build_trellis_metadata.py --manifest manifests/_smoke.csv --split train \
  --out-dir "$TRELLIS/datasets/carcaption3k_1620_smoke"
```
Expected: prints the chosen objects + extensions; stages them.

- [ ] **Step 2: Run the full pipeline on the smoke set (this DOES use GPU/blender — it is tiny)**

> **Autonomy note:** This 3-object smoke run is the de-risking exception to "no GPU jobs" — it is a sanity dry-run, not training. First check a GPU + blender are usable: `nvidia-smi && "$BLENDER" --version`. If either is unavailable in the agent's environment, **do not run** — instead write these exact stage commands into `runs/trellis_prep/COMMANDS.md` under a "SMOKE (run first)" heading, mark the Phase 3 gate "pending human smoke run", and continue with the rest of the prep.

Run each stage from `$TRELLIS/dataset_toolkits` (the agent may run this small smoke job; it is not a full training job):
```bash
cd "$TRELLIS/dataset_toolkits"
S=carcaption3k_1620_smoke
D="$TRELLIS/datasets/$S"
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python render.py        CarCaption3K1620 --output_dir "$D" --num_views 150 --rank 0 --world_size 1
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python voxelize.py      CarCaption3K1620 --output_dir "$D"
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python extract_feature.py --output_dir "$D" --batch_size 16
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python encode_ss_latent.py --output_dir "$D"
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python encode_latent.py    --output_dir "$D"
python build_metadata.py CarCaption3K1620 --output_dir "$D"
python render_cond.py   CarCaption3K1620 --output_dir "$D" --num_views 24 --rank 0 --world_size 1
python build_metadata.py CarCaption3K1620 --output_dir "$D"
```
Expected: each stage completes; `metadata.csv` flags flip to True per stage.

- [ ] **Step 3: Verify the outputs match the trainer's expected layout**

Run:
```bash
D="$TRELLIS/datasets/carcaption3k_1620_smoke"
python - <<PY
import os, glob, pandas as pd
D="$D"
df=pd.read_csv(os.path.join(D,"metadata.csv"))
for col in ["rendered","voxelized","feature_dinov2_vitl14_reg","ss_latent_ss_enc_conv3d_16l8_fp16","latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16","cond_rendered"]:
    print(col, df[col].sum(), "/", len(df))
assert glob.glob(os.path.join(D,"latents","*","*.npz")), "no slat latents"
assert glob.glob(os.path.join(D,"ss_latents","*","*.npz")), "no ss latents"
assert glob.glob(os.path.join(D,"renders_cond","*","transforms.json")), "no cond renders"
print("OK smoke pipeline produced latents + cond renders")
PY
```
Expected: all flags == len(df); `OK smoke pipeline produced latents + cond renders`.
**If any mesh format failed to render**, record which extension and how many of the 1,620 it affects; decide (with the user) whether to drop those objects (and re-lock the manifest) or convert them to `.glb`.

- [ ] **Step 4: Record disk usage + extrapolate, then clean up**

Run:
```bash
du -sh "$TRELLIS/datasets/carcaption3k_1620_smoke"
echo "Extrapolate × (1620/3) for the full prep disk estimate; record in runs/RECON.md"
rm -rf "$TRELLIS/datasets/carcaption3k_1620_smoke" manifests/_smoke.csv
```
Expected: a size figure; note the ~540× extrapolation (150-view renders dominate) in `runs/RECON.md`.

- [ ] **Step 5: Emit the full-1,620 pipeline commands (do NOT run)**

Write `$QA/runs/trellis_prep/COMMANDS.md` with the same stage sequence as Step 2 but with `--output_dir "$TRELLIS/datasets/carcaption3k_1620_train"` and `…_val`, plus a note: use `--rank/--world_size` to shard `render`/`render_cond` across GPUs, and run `build_metadata.py` after each sharded stage to merge. Commit:
```bash
git add runs/trellis_prep/COMMANDS.md runs/RECON.md
git commit -m "docs(ablation): TRELLIS smoke verified; emit full prep commands + disk estimate"
```

---

## Phase 4 — TRELLIS finetune configs (ss + slat, txt + img)

### Task 4.1: Clone the four finetune configs with matched budgets

**Files:**
- Create: `$TRELLIS/configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json`
- Create: `$TRELLIS/configs/generation/ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json`
- Create: `$TRELLIS/configs/generation/slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json`
- Create: `$TRELLIS/configs/generation/ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json`

- [ ] **Step 1: Clone each committed finetune config**

Run:
```bash
cd "$TRELLIS/configs/generation"
cp slat_flow_txt_dit_XL_64l8p2_fp16_finetune.json slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json
cp ss_flow_txt_dit_XL_16l8_fp16_finetune.json    ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json
cp slat_flow_img_dit_L_64l8p2_fp16_finetune.json  slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json
cp ss_flow_img_dit_L_16l8_fp16_finetune.json     ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json
```

- [ ] **Step 2: Edit budgets + aesthetic filter (GATE B) in each clone**

In each clone set `trainer.args.max_steps` to the **recovered MeshFleet budget** (from `runs/reference_budgets.json`): `slat_txt=100000`, `ss_txt=100000`, `ss_img=100000`, `slat_img=30000`. In **both** `dataset.args` and `eval_dataset.args` set `"min_aesthetic_score": 0.0` (GATE B default, so the metadata's stored score never drops a selected object). Leave lr, batch_size_per_gpu, batch_split, i_log/i_sample/i_save, keep_ckpt, normalization, model args unchanged. Adjust the `wandb_config.name` to include `_carcaption3k_1620`.

Run (per file; example for slat_txt):
```bash
python - <<'PY'
import json
for fn, steps in [
  ("slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json", 100000),
  ("ss_flow_txt_dit_XL_16l8_fp16_carcaption3k_1620.json", 100000),
  ("ss_flow_img_dit_L_16l8_fp16_carcaption3k_1620.json", 100000),
  ("slat_flow_img_dit_L_64l8p2_fp16_carcaption3k_1620.json", 30000),
]:
    p=f"/home/damian/Projects/TRELLIS/configs/generation/{fn}"
    c=json.load(open(p))
    c["trainer"]["args"]["max_steps"]=steps
    c["trainer"]["args"].setdefault("wandb_config",{})["name"]=fn.replace(".json","")
    for k in ("dataset","eval_dataset"):
        if k in c: c[k]["args"]["min_aesthetic_score"]=0.0
    json.dump(c, open(p,"w"), indent=4)
    print("edited", fn, "max_steps=", steps)
PY
```
Expected: prints four "edited …" lines.

- [ ] **Step 3: Verify the edits**

Run:
```bash
cd "$TRELLIS/configs/generation"
for f in *_carcaption3k_1620.json; do
  echo "$f"; python -c "import json;c=json.load(open('$f'));print('  max_steps',c['trainer']['args']['max_steps'],'min_aes',c['dataset']['args']['min_aesthetic_score'])"
done
```
Expected: each shows the matched `max_steps` and `min_aes 0.0`.

- [ ] **Step 4: Commit (TRELLIS repo)**

```bash
git -C "$TRELLIS" add configs/generation/*_carcaption3k_1620.json
git -C "$TRELLIS" commit -m "feat: CarCaption3K-1620 matched finetune configs (ss+slat, txt+img)"
```

### Task 4.2: Tryrun the dataloader against the smoke/real dataset

> Requires Phase 3 outputs. Use `--tryrun` (train.py supports it) so no training happens.

- [ ] **Step 1: Tryrun each config (CPU/quick)**

Run (once the train/val prep exists; for an early check, point `--data_dir`/`--eval_data_dir` at the smoke dataset rebuilt with all flags True):
```bash
cd "$TRELLIS"
python train.py --config configs/generation/slat_flow_txt_dit_XL_64l8p2_fp16_carcaption3k_1620.json \
  --output_dir outputs/_tryrun_slat_txt \
  --data_dir datasets/carcaption3k_1620_train \
  --eval_data_dir datasets/carcaption3k_1620_val \
  --finetune_ckpt "<slat_txt base ckpt from reference_budgets.json>" \
  --num_gpus 1 --tryrun
```
Expected: dataset loads, filter stats print (non-zero instances after `min_aesthetic_score=0`), model + finetune checkpoint load without shape/key errors, exits before the training loop. Repeat for the other three configs. **If instances filter to 0**, GATE B value is wrong — fix `min_aesthetic_score`/stored score and re-run.

- [ ] **Step 2: Emit the four training launch commands (do NOT run full training)**

Write `$QA/runs/trellis_train/COMMANDS.md` with the four `train.py` commands (drop `--tryrun`, set the real `--num_gpus`, point at the full `datasets/carcaption3k_1620_train|val`, and each stage's recovered `--finetune_ckpt`). Note the run order: prep data (Phase 3) → finetune ss + slat per family → inference uses both finetuned stages.

- [ ] **Step 3: Commit**

```bash
cd "$QA"; git add runs/trellis_train/COMMANDS.md
git commit -m "docs(ablation): emit TRELLIS finetune launch commands"
```

---

## Phase 5 — Evaluation wiring

### Task 5.1: Per-family generate→eval command templates

**Files:**
- Create: `$QA/runs/eval/COMMANDS.md`

The eval is fixed: `--gen-folder <X>/<sha>/000.png..011.png` vs the 232-object GT. The reference rows (`TRELLIS`, `TRELLIS_FT`) already exist in `analysis/track_b/out/per_object_metrics_all.csv`.

- [ ] **Step 1: Write the eval command templates**

Write `$QA/runs/eval/COMMANDS.md`:
```text
## FLUX+TRELLIS (P0): generate FLUX images for the 232 test prompts, run zero-shot TRELLIS, then eval
# 1. FLUX inference with the CarCaption LoRA over data/meshfleet/meshfleet_test.csv refined_3d_prompt (232 objs)
#    -> produce data/ablation/gen/flux_trellis_cc1620/<sha>/000.png..011.png
# 2. Evaluate:
cd /home/damian/Projects/quality_assessment_library
python run_meshfleet_eval.py \
  --config meshfleet_benchmark/benchmark_config.yaml \
  --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
  --gen-folder data/ablation/gen/flux_trellis_cc1620 \
  --metadata-file data/meshfleet/meshfleet_test.csv \
  --output-folder runs/flux_trellis_cc1620/metrics \
  --per-object --model-name FLUX_TRELLIS_CC1620

## TRELLIS-Txt (P1) and TRELLIS-image (P2): generate with BOTH finetuned stages (ss+slat), then eval
python run_meshfleet_eval.py --config meshfleet_benchmark/benchmark_config.yaml \
  --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
  --gen-folder data/ablation/gen/trellis_txt_cc1620 \
  --metadata-file data/meshfleet/meshfleet_test.csv \
  --output-folder runs/trellis_txt_cc1620/metrics --per-object --model-name TRELLIS_TXT_CC1620
python run_meshfleet_eval.py --config meshfleet_benchmark/benchmark_config.yaml \
  --gt-folder data/meshfleet/benchmark_data/meshfleet_eval_images \
  --gen-folder data/ablation/gen/trellis_img_cc1620 \
  --metadata-file data/meshfleet/meshfleet_test.csv \
  --output-folder runs/trellis_img_cc1620/metrics --per-object --model-name TRELLIS_IMG_CC1620
```
Add a note: match the MeshFleet generation settings (preprocessing flags) used to produce the existing `TRELLIS`/`TRELLIS_FT` rows — check `analysis/track_b/out/RUN_LOG.md` for the exact flags and replicate them.

- [ ] **Step 2: Verify the GT/gen contract on one object**

Run:
```bash
ls data/meshfleet/benchmark_data/meshfleet_eval_images | head -1 | xargs -I{} ls data/meshfleet/benchmark_data/meshfleet_eval_images/{}
```
Expected: `000.png … 011.png` (12 files). Confirms the per-object gen-folder layout the generation step must match.

- [ ] **Step 3: Commit**

```bash
git add runs/eval/COMMANDS.md
git commit -m "docs(ablation): per-family generate->eval command templates"
```

### Task 5.2: Results table assembler (TDD)

**Files:**
- Create: `scripts/assemble_ablation_table.py`
- Test: `tests/ablations/test_assemble_table.py`

- [ ] **Step 1: Write failing test**

```python
# tests/ablations/test_assemble_table.py
import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "asm", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "assemble_ablation_table.py"))
asm = importlib.util.module_from_spec(spec); spec.loader.exec_module(asm)

def _per_object(tmp_path, model, mse, clip):
    p = tmp_path / f"{model}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["object_id","model","metric","value","n_views","family"])
        for i in range(3):
            w.writerow([f"{i:064x}", model, "MSE", mse, 12.0, "appearance"])
            w.writerow([f"{i:064x}", model, "CLIP-S", clip, 12.0, "appearance"])
    return str(p)

def test_means_and_rows(tmp_path):
    f1 = _per_object(tmp_path, "TRELLIS", 0.03, 0.90)
    f2 = _per_object(tmp_path, "TRELLIS_IMG_CC1620", 0.02, 0.93)
    rows = asm.summarize([f1, f2], metrics=["MSE","CLIP-S"])
    by = {r["model"]: r for r in rows}
    assert abs(by["TRELLIS"]["MSE"] - 0.03) < 1e-9
    assert abs(by["TRELLIS_IMG_CC1620"]["CLIP-S"] - 0.93) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/ablations/test_assemble_table.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# scripts/assemble_ablation_table.py
"""Aggregate per-object metric CSVs into the controlled-ablation summary table."""
import argparse, csv, glob, os, statistics

def summarize(per_object_csvs, metrics):
    """Return one row per model with the mean of each requested metric across objects."""
    acc = {}  # model -> metric -> list[value]
    for path in per_object_csvs:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["metric"] in metrics:
                    acc.setdefault(row["model"], {}).setdefault(row["metric"], []).append(float(row["value"]))
    rows = []
    for model, md in sorted(acc.items()):
        r = {"model": model}
        for m in metrics:
            vals = md.get(m, [])
            r[m] = statistics.mean(vals) if vals else None
        rows.append(r)
    return rows

def to_markdown(rows, metrics):
    head = "| Model | " + " | ".join(metrics) + " |\n"
    head += "| --- | " + " | ".join("---:" for _ in metrics) + " |\n"
    body = ""
    for r in rows:
        body += "| " + r["model"] + " | " + " | ".join(
            (f"{r[m]:.4f}" if r[m] is not None else "n/a") for m in metrics) + " |\n"
    return head + body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="per_object_metrics_*.csv files (incl. analysis/track_b/out/per_object_metrics_all.csv)")
    ap.add_argument("--metrics", default="MSE,CLIP-S,clip_score,image_reward")
    ap.add_argument("--out-md", default="tables/carcaption3k_1620_controlled_ablation_summary.md")
    ap.add_argument("--out-csv", default="tables/carcaption3k_1620_controlled_ablation_summary.csv")
    a = ap.parse_args()
    metrics = a.metrics.split(",")
    files = []
    for pat in a.inputs:
        files.extend(glob.glob(pat))
    rows = summarize(files, metrics)
    os.makedirs(os.path.dirname(a.out_md), exist_ok=True)
    with open(a.out_md, "w") as f:
        f.write(to_markdown(rows, metrics))
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model"] + metrics); w.writeheader(); w.writerows(rows)
    print(f"OK wrote {a.out_md} and {a.out_csv} ({len(rows)} models)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/ablations/test_assemble_table.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-run against existing reference data**

Run:
```bash
python scripts/assemble_ablation_table.py --inputs "analysis/track_b/out/per_object_metrics_all.csv"
cat tables/carcaption3k_1620_controlled_ablation_summary.md
```
Expected: a markdown table including the existing `TRELLIS` and `TRELLIS_FT` reference rows (the CC1620 rows fill in once real eval CSVs exist).

- [ ] **Step 6: Commit**

```bash
git add scripts/assemble_ablation_table.py tests/ablations/test_assemble_table.py tables/carcaption3k_1620_controlled_ablation_summary.md tables/carcaption3k_1620_controlled_ablation_summary.csv
git commit -m "feat(ablation): results table assembler"
```

---

## Phase 6 — Sanity gate, run logging, deliverables

### Task 6.1: Pre-launch validator (TDD)

**Files:**
- Create: `scripts/validate_ablation_ready.py`
- Test: `tests/ablations/test_validate.py`

Encodes handoff §7.

- [ ] **Step 1: Write failing test**

```python
# tests/ablations/test_validate.py
import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "val", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "validate_ablation_ready.py"))
val = importlib.util.module_from_spec(spec); spec.loader.exec_module(val)

def _manifest(tmp_path, n, dup=False):
    p = tmp_path / "m.csv"; assets = tmp_path / "a"; assets.mkdir()
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256","asset_path"]); w.writeheader()
        for i in range(n):
            s = f"{(0 if dup else i):064x}"
            (assets / f"{s}.glb").write_text("m")
            w.writerow({"sha256": s, "asset_path": str(assets / f"{s}.glb")})
    return str(p)

def test_passes_clean_manifest(tmp_path):
    m = _manifest(tmp_path, 5)
    ok, problems = val.check_manifest(m, expected_n=5)
    assert ok and not problems

def test_flags_count_and_dupes(tmp_path):
    m = _manifest(tmp_path, 5, dup=True)
    ok, problems = val.check_manifest(m, expected_n=1620)
    assert not ok
    assert any("count" in p for p in problems) and any("unique" in p for p in problems)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/ablations/test_validate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# scripts/validate_ablation_ready.py
"""Pre-launch sanity gate for the CarCaption3K-1620 ablation (handoff §7)."""
import argparse, csv, hashlib, json, os, sys

def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def check_manifest(manifest_csv, expected_n):
    problems = []
    if not os.path.exists(manifest_csv):
        return False, [f"manifest missing: {manifest_csv}"]
    rows = _rows(manifest_csv)
    if len(rows) != expected_n:
        problems.append(f"row count {len(rows)} != expected {expected_n}")
    shas = [r["sha256"] for r in rows]
    if len(set(shas)) != len(shas):
        problems.append("sha256 not unique")
    for r in rows:
        if not os.path.exists(r["asset_path"]):
            problems.append(f"missing asset: {r['asset_path']}"); break
    return (not problems), problems

def check_configs_reference_manifest(config_globs, manifest_hash_file):
    problems = []
    if not os.path.exists(manifest_hash_file):
        problems.append(f"manifest hash file missing: {manifest_hash_file}")
    return (not problems), problems

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/carcaption3k_1620_locked.csv")
    ap.add_argument("--expected-n", type=int, default=1620)
    ap.add_argument("--gt-folder", default="data/meshfleet/benchmark_data/meshfleet_eval_images")
    a = ap.parse_args()
    ok, problems = check_manifest(a.manifest, a.expected_n)
    # held-out eval set unchanged?
    if not os.path.isdir(a.gt_folder):
        ok = False; problems.append(f"held-out GT folder missing: {a.gt_folder}")
    else:
        n_obj = len([d for d in os.listdir(a.gt_folder) if os.path.isdir(os.path.join(a.gt_folder, d))])
        if n_obj != 232:
            problems.append(f"held-out GT object count {n_obj} != 232 (must not change)")
            ok = False
    print(json.dumps({"ready": ok, "problems": problems}, indent=2))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/ablations/test_validate.py -v`
Expected: PASS.

- [ ] **Step 5: Run the real gate**

Run: `python scripts/validate_ablation_ready.py`
Expected: JSON `"ready": true, "problems": []`. If false, fix the reported problem before emitting "ready to launch".

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_ablation_ready.py tests/ablations/test_validate.py
git commit -m "feat(ablation): pre-launch sanity gate"
```

### Task 6.2: Run-log scaffolding + final deliverables index

**Files:**
- Create: `runs/README.md`

- [ ] **Step 1: Scaffold per-run log template**

Write `runs/README.md` documenting that each launched run must produce `runs/<run_id>/{RUN_LOG.md, config_resolved.{yaml,json}, manifest_sha256.txt, metrics.json, metrics.csv, COMMANDS.md}`, and listing the deliverables (handoff §8): the manifest + summary, `configs/ablations/*` (here the per-repo config clones), `runs/*/RUN_LOG.md`, `runs/*/metrics.*`, and `tables/carcaption3k_1620_controlled_ablation_summary.{csv,md}`. Include a `RUN_LOG.md` template with: GPU type/count, wall-clock, code commit hashes (QA + TRELLIS + ai-toolkit), exact command, failures/restarts, and the `manifest_sha256` used.

- [ ] **Step 2: Copy the manifest hash into each emitted run dir**

Run:
```bash
for d in runs/flux_carcaption3k_1620 runs/trellis_txt_cc1620 runs/trellis_img_cc1620; do
  mkdir -p "$d"; cp manifests/carcaption3k_1620_locked.sha256 "$d/manifest_sha256.txt"
done
```
Expected: each run dir has `manifest_sha256.txt` (proves every family uses the same locked manifest — handoff §6).

- [ ] **Step 3: Final readiness report + commit**

Run `python scripts/validate_ablation_ready.py` once more; paste its output into `runs/RECON.md` under a "Ready to launch" heading. Commit:
```bash
git add runs/README.md runs/*/manifest_sha256.txt runs/RECON.md
git commit -m "docs(ablation): run-log scaffolding + readiness report"
```

---

## Self-review checklist (run before declaring done)

- [ ] Manifest has exactly `N` rows, unique sha, all assets exist (`validate_ablation_ready.py` green).
- [ ] The same `manifest_sha256.txt` is present in every run dir.
- [ ] Held-out GT folder still has 232 objects (validator checks this).
- [ ] FLUX config diff vs MeshFleet config is only name/repo/dataset path; `steps`/`lr`/`lora` unchanged.
- [ ] Four TRELLIS configs have matched `max_steps` (100k/100k/100k/30k) and `min_aesthetic_score` that passes all selected objects (tryrun shows non-zero instances).
- [ ] Smoke pipeline produced slat+ss latents and cond renders; mesh-format coverage recorded.
- [ ] Every emitted launch command is in a `runs/*/COMMANDS.md`; the agent launched no full GPU job.
- [ ] GATE A (object count) and GATE B (aesthetic scale) decisions are recorded in `runs/RECON.md`.

## Deliverables (handoff §8)

```
manifests/carcaption3k_1620_locked.csv
manifests/carcaption3k_1620_locked_summary.json
manifests/carcaption3k_1620_locked.sha256
ai-toolkit/config/flux_trellis_carcaption3k_1620.yaml
TRELLIS/dataset_toolkits/datasets/CarCaption3K1620.py
TRELLIS/configs/generation/*_carcaption3k_1620.json   (4 files)
runs/reference_budgets.json
runs/RECON.md
runs/*/COMMANDS.md         (flux, trellis_prep, trellis_train, eval)
runs/*/manifest_sha256.txt
tables/carcaption3k_1620_controlled_ablation_summary.{csv,md}
scripts/{build_carcaption3k_1620_subset,build_flux_filtered_dataset,build_trellis_metadata,validate_ablation_ready,assemble_ablation_table}.py
tests/ablations/*.py
```
