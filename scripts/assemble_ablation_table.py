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
