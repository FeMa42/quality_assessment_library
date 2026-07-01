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
        if src is None:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/carcaption3k_1620_locked.csv")
    ap.add_argument("--split", required=True, choices=["train","val"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aesthetic-score", type=float, default=0.0)
    a = ap.parse_args()
    df = build_metadata_df(a.manifest, a.split, a.aesthetic_score)
    os.makedirs(a.out_dir, exist_ok=True)
    df.to_csv(os.path.join(a.out_dir, "metadata.csv"), index=False)
    stage_meshes(df, a.out_dir, a.manifest)
    print(f"OK wrote {len(df)} rows to {a.out_dir}/metadata.csv and staged meshes")

if __name__ == "__main__":
    main()
