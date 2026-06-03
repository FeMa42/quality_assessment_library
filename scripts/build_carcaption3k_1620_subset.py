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
    ap.add_argument("--timestamp", default="", help="ISO timestamp passed in")
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
