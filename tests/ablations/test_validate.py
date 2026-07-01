import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "val", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "validate_ablation_ready.py"))
val = importlib.util.module_from_spec(spec); spec.loader.exec_module(val)

def _manifest(tmp_path, n, dup=False):
    p = tmp_path / "m.csv"; assets = tmp_path / "a"; assets.mkdir()
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256","asset_path"]); w.writeheader()
        for i in range(n):
            s = f"{(0 if dup else i):064x}"
            (assets / f"{s}.glb").write_text("m")
            w.writerow({"sha256": s, "asset_path": str(assets / f"{s}.glb")})
    return str(p)

def test_passes_clean_manifest(tmp_path):
    m = _manifest(tmp_path, 5)
    ok, problems = val.check_manifest(m, expected_n=5)
    assert ok and not problems

def test_flags_count_and_dupes(tmp_path):
    m = _manifest(tmp_path, 5, dup=True)
    ok, problems = val.check_manifest(m, expected_n=1620)
    assert not ok
    assert any("count" in p for p in problems) and any("unique" in p for p in problems)
