"""Stage generation `sample_<sha>.glb` files into <staging>/<sha>/<sha>.glb for render_for_quality_assessment.py."""
import argparse, glob, os

def stage(gen_dir, staging_dir):
    n = 0
    for path in sorted(glob.glob(os.path.join(gen_dir, "sample_*.glb"))):
        sha = os.path.basename(path)[len("sample_"):-len(".glb")]
        obj_dir = os.path.join(staging_dir, sha); os.makedirs(obj_dir, exist_ok=True)
        dst = os.path.join(obj_dir, f"{sha}.glb")
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.abspath(path), dst)
        n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True, help="generation output_dir containing sample_<sha>.glb")
    ap.add_argument("--staging-dir", required=True, help="output: <staging>/<sha>/<sha>.glb")
    a = ap.parse_args()
    n = stage(a.gen_dir, a.staging_dir)
    print(f"OK staged {n} GLBs from {a.gen_dir} into {a.staging_dir}/<sha>/<sha>.glb")

if __name__ == "__main__":
    main()
