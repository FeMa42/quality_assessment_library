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
