"""Build the held-out prompt CSV (sha256, refined_3d_prompt) for exactly the benchmark GT objects."""
import argparse, csv, os

def heldout_shas(gt_dir):
    return sorted(d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d)))

def load_prompts(prompt_sources):
    """Map sha256 -> refined_3d_prompt from sources in priority order (first non-empty wins)."""
    prompts = {}
    for src in prompt_sources:
        if not src or not os.path.exists(src):
            continue
        with open(src, newline="") as f:
            for row in csv.DictReader(f):
                sha = (row.get("sha256") or "").strip()
                p = (row.get("refined_3d_prompt") or "").strip()
                if sha and p and sha not in prompts:
                    prompts[sha] = p
    return prompts

def build(gt_dir, prompt_sources, out_csv):
    shas = heldout_shas(gt_dir)
    prompts = load_prompts(prompt_sources)
    rows, missing = [], []
    for sha in shas:
        if sha in prompts:
            rows.append({"sha256": sha, "refined_3d_prompt": prompts[sha]})
        else:
            missing.append(sha)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256", "refined_3d_prompt"]); w.writeheader(); w.writerows(rows)
    return rows, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/meshfleet/benchmark_data/meshfleet_eval_images")
    ap.add_argument("--prompt-sources", nargs="+",
                    default=["/home/damian/Projects/TRELLIS/datasets/meshfleet_benchmark/meshfleet_test.csv",
                             "data/meshfleet/meshfleet_test.csv"])
    ap.add_argument("--out-csv", default="manifests/meshfleet_heldout_232_prompts.csv")
    a = ap.parse_args()
    rows, missing = build(a.gt_dir, a.prompt_sources, a.out_csv)
    print(f"OK wrote {len(rows)} prompts to {a.out_csv}")
    print("All held-out objects have a prompt." if not missing
          else f"WARNING: {len(missing)} held-out shas had NO prompt: {missing[:5]}")

if __name__ == "__main__":
    main()
