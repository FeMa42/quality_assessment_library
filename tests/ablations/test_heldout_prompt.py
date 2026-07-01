import os, csv, importlib.util
spec = importlib.util.spec_from_file_location(
    "hp", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_heldout_prompt_file.py"))
hp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hp)

def _gt(tmp_path, shas):
    gt = tmp_path / "gt"; gt.mkdir()
    for s in shas: (gt / s).mkdir()
    (gt / "not_a_dir.txt").write_text("x")
    return str(gt)

def _src(tmp_path, name, mapping):
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sha256","refined_3d_prompt"]); w.writeheader()
        for s, pr in mapping.items(): w.writerow({"sha256": s, "refined_3d_prompt": pr})
    return str(p)

def test_heldout_shas_sorted_dirs_only(tmp_path):
    shas = [f"{i:064x}" for i in range(3)]
    gt = _gt(tmp_path, shas)
    assert hp.heldout_shas(gt) == sorted(shas)

def test_build_intersects_and_reports_missing(tmp_path):
    shas = [f"{i:064x}" for i in range(3)]
    gt = _gt(tmp_path, shas)
    src1 = _src(tmp_path, "s1.csv", {shas[0]: "prompt A", shas[1]: "prompt B"})  # missing shas[2]
    src2 = _src(tmp_path, "s2.csv", {shas[2]: "prompt C", shas[0]: "OTHER"})     # has shas[2]; shas[0] should NOT override src1
    out = str(tmp_path / "out.csv")
    rows, missing = hp.build(gt, [src1, src2], out)
    by = {r["sha256"]: r["refined_3d_prompt"] for r in rows}
    assert len(rows) == 3 and not missing
    assert by[shas[0]] == "prompt A"   # first source wins
    assert by[shas[2]] == "prompt C"   # filled from second source
    # output is sorted by sha and has exactly the two columns
    got = list(csv.DictReader(open(out)))
    assert [r["sha256"] for r in got] == sorted(shas)
    assert list(got[0].keys()) == ["sha256","refined_3d_prompt"]
