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
