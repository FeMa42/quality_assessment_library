# tests/ablations/test_flux_adapter.py
import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "flux_adapter",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_flux_filtered_dataset.py"))
fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)

def _manifest(tmp_path, shas, splits):
    src = tmp_path / "aitk"; src.mkdir()
    for s in shas:
        (src / f"{s}.png").write_bytes(b"\x89PNG"); (src / f"{s}.txt").write_text("cap "+s)
    p = tmp_path / "m.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256","split"]); w.writeheader()
        for s, sp in zip(shas, splits): w.writerow({"sha256": s, "split": sp})
    return str(p), str(src)

def test_builds_symlinks_for_selected_split(tmp_path):
    shas=[f"{i:064x}" for i in range(4)]
    man, src = _manifest(tmp_path, shas, ["train","train","val","train"])
    out = tmp_path / "flux_ds"
    n = fa.build_symlink_dataset(man, src, str(out), splits=("train",))
    assert n == 3
    assert os.path.exists(out / f"{shas[0]}.png") and os.path.exists(out / f"{shas[0]}.txt")
    assert not os.path.exists(out / f"{shas[2]}.png")   # val excluded
