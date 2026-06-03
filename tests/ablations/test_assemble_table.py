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
