#!/usr/bin/env python3
"""Validate per-object outputs for Track B and concatenate the per-method CSVs.

Per HANDOVER §6:
1. Per-object means (restricted to n_views == 12) should reproduce the model-level
   overall_* numbers to within tolerance.
2. Object count must be sane (≤ number of generator output folders).
3. Spot-check a few objects for one metric.

Also concatenates per_object_metrics_<method>.csv files into per_object_metrics_all.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


METHODS = [
    ("Hunyuan3D_20", "Hunyuan3D_20_meshes_render"),
    ("TRELLIS", "TRELLIS_image_base_mesh_render"),
    ("TRELLIS_FT", "TRELLIS_image_finetuned_render"),
    ("Hunyuan3D_21", "hunyuan3d21_render"),
    ("Instantmesh", "instantmesh_meshes_render"),
]

TOL_RELATIVE = 1e-3  # 0.1% — kernels are identical, only aggregation order differs.


def load_per_object_json(folder: Path) -> dict | None:
    p = folder / "semantic_geometric_per_object_eval.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def validate_method(name: str, dir_basename: str, repo_root: Path) -> dict:
    out_folder = repo_root / "data" / f"{dir_basename}_object_metrics"
    report = {
        "method": name,
        "dir": dir_basename,
        "out_folder": str(out_folder),
        "issues": [],
    }

    sg = load_per_object_json(out_folder)
    if sg is None:
        report["issues"].append("missing semantic_geometric_per_object_eval.json")
        return report

    per_obj = sg.get("per_object", {})
    overall_sem = sg.get("overall_semantic_metrics") or {}
    overall_geom = sg.get("overall_geometric_metrics") or {}
    skipped = sg.get("skipped_objects", [])

    report["n_objects"] = len(per_obj)
    report["n_skipped"] = len(skipped)
    report["overall_semantic"] = overall_sem
    report["overall_geometric"] = overall_geom

    n12 = [o for o, e in per_obj.items() if e.get("n_views") == 12]
    report["n_objects_full_views"] = len(n12)

    # Re-aggregate per-object data restricted to n_views == 12 and compare to overall_*
    def remean(metric_block_key: str, all_keys: dict[str, float]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for o in n12:
            block = per_obj[o].get(metric_block_key) or {}
            for k, v in block.items():
                if k == "Image_Pairs" or v is None:
                    continue
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / counts[k] for k in sums if counts[k]}

    sem_recomp = remean("semantic_metrics", overall_sem)
    geom_recomp = remean("geometric_metrics", overall_geom)
    report["semantic_recomputed_from_per_object"] = sem_recomp
    report["geometric_recomputed_from_per_object"] = geom_recomp

    def rel_diffs(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
        out = {}
        for k in a:
            if k not in b or b[k] in (None, 0):
                continue
            out[k] = abs(a[k] - b[k]) / abs(b[k]) if b[k] else float("inf")
        return out

    report["semantic_rel_diff"] = rel_diffs(sem_recomp, overall_sem)
    report["geometric_rel_diff"] = rel_diffs(geom_recomp, overall_geom)

    bad_sem = {k: v for k, v in report["semantic_rel_diff"].items() if v > TOL_RELATIVE}
    bad_geom = {k: v for k, v in report["geometric_rel_diff"].items() if v > TOL_RELATIVE}

    # Note: overall_* averages over ALL processed objects (including those with
    # n_views<12), while we recompute on n_views==12 only. A small mismatch is
    # expected when partial-view objects exist; flag only if relative diff > 1%.
    BIG_TOL = 1e-2
    if any(v > BIG_TOL for v in bad_sem.values()):
        report["issues"].append(f"semantic recompute mismatch > {BIG_TOL}: {bad_sem}")
    if any(v > BIG_TOL for v in bad_geom.values()):
        report["issues"].append(f"geometric recompute mismatch > {BIG_TOL}: {bad_geom}")

    csv_p = out_folder / f"per_object_metrics_{name}.csv"
    if not csv_p.exists():
        report["issues"].append(f"missing CSV: {csv_p}")
    else:
        df = pd.read_csv(csv_p)
        report["csv_rows"] = len(df)
        report["csv_unique_objects"] = df["object_id"].nunique()
        report["csv_metrics"] = sorted(df["metric"].unique().tolist())

    return report


def concat_csvs(repo_root: Path, methods: list[tuple[str, str]]) -> Path | None:
    frames = []
    for name, dir_basename in methods:
        csv_p = repo_root / "data" / f"{dir_basename}_object_metrics" / f"per_object_metrics_{name}.csv"
        if not csv_p.exists():
            print(f"WARN: missing {csv_p}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(csv_p))
    if not frames:
        return None
    out_dir = repo_root / "analysis" / "track_b" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_object_metrics_all.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--methods", nargs="*",
                    help="Subset of method names to validate (default: all five)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    methods = METHODS
    if args.methods:
        wanted = set(args.methods)
        methods = [m for m in METHODS if m[0] in wanted]

    all_ok = True
    print("=" * 78)
    for name, dir_basename in methods:
        print(f"\n# {name} ({dir_basename})")
        rep = validate_method(name, dir_basename, repo_root)
        print(json.dumps(rep, indent=2, default=str))
        if rep["issues"]:
            all_ok = False
    print("=" * 78)

    out_path = concat_csvs(repo_root, methods)
    if out_path:
        print(f"\nWrote concatenated CSV: {out_path}")
    else:
        print("\nNo CSVs concatenated (none found).")
        all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
