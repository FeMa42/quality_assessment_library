#!/usr/bin/env python3
"""Backfill the vehicle_dimensions stage for a method whose initial run failed it.

The first Hunyuan3D_20 sweep failed the vehicle_dimensions stage because timm.layers
was not installed. This script runs ONLY the vehicle_dim stage and merges its outputs
into the existing data/<dir>_object_metrics/ folder, then re-emits the per-object CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-folder", required=True,
                    help="Pre-scaled generated folder, e.g. data/meshfleet/benchmark_data/scaled/Hunyuan3D_20_meshes_render")
    ap.add_argument("--metadata-file", default="data/meshfleet/meshfleet_test.csv")
    ap.add_argument("--output-folder", required=True,
                    help="Existing per-object output folder, e.g. data/Hunyuan3D_20_meshes_render_object_metrics")
    ap.add_argument("--model-name", required=True,
                    help="Logical model name (must match the existing per_object_metrics_<model>.csv)")
    args = ap.parse_args()

    # Local imports — heavy modules
    from metrics.helpers import evaluate_vehicle_dimensions
    from metrics.viewpoint_florence import FlorenceWheelbaseOD

    out_dir = Path(args.output_folder)
    if not out_dir.is_dir():
        print(f"Output folder does not exist: {out_dir}", file=sys.stderr)
        return 1

    print(f"[backfill] Running vehicle_dimensions on {args.gen_folder}")
    florence = FlorenceWheelbaseOD()
    average_diff, std_diff, per_object = evaluate_vehicle_dimensions(
        args.gen_folder, args.metadata_file, florence
    )
    print(f"[backfill] Done. {len(per_object)} objects with dims; avg = {average_diff}")

    # Write the standalone JSON files
    veh_path = out_dir / "vehicle_dimensions_eval.json"
    with veh_path.open("w") as f:
        json.dump({"average_diff": average_diff, "std_diff": std_diff}, f, indent=4)
    veh_per_obj_path = out_dir / "vehicle_dimensions_per_object_eval.json"
    with veh_per_obj_path.open("w") as f:
        json.dump(per_object, f, indent=4)
    print(f"[backfill] wrote {veh_path}")
    print(f"[backfill] wrote {veh_per_obj_path}")

    # Merge into combined metrics JSON if it exists
    combined_path = out_dir / "meshfleet_combined_metrics.json"
    if combined_path.exists():
        with combined_path.open() as f:
            combined = json.load(f)
        combined["vehicle_dimensions"] = {
            "average_diff": average_diff,
            "std_diff": std_diff,
        }
        combined["vehicle_dimensions_per_object"] = per_object
        with combined_path.open("w") as f:
            json.dump(combined, f, indent=4)
        print(f"[backfill] merged into {combined_path}")

    # Append vehicle_dim rows to the per-object CSV
    csv_path = out_dir / f"per_object_metrics_{args.model_name}.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        # Drop any prior vehicle_dim rows in case this is being re-run
        existing = existing[existing["family"] != "vehicle_dim"]
        new_rows = []
        for obj, scores in per_object.items():
            for metric, value in scores.items():
                new_rows.append({
                    "object_id": obj,
                    "model": args.model_name,
                    "metric": metric,
                    "value": value,
                    "n_views": None,
                    "family": "vehicle_dim",
                })
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_csv(csv_path, index=False)
        print(f"[backfill] appended {len(new_rows)} vehicle_dim rows -> {csv_path} (total {len(merged)})")
    else:
        print(f"[backfill] CSV not found, skipping CSV merge: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
