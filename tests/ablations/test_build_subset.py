# tests/ablations/test_build_subset.py
import os, json, csv, importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "build_subset",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_carcaption3k_1620_subset.py"))
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)

def _make_pool(tmp_path, shas, with_caption=True, with_preview=True, ext=".glb"):
    mesh = tmp_path / "meshes"; aitk = tmp_path / "aitk"; ren = tmp_path / "ren"
    for d in (mesh, aitk, ren): d.mkdir(exist_ok=True)
    for s in shas:
        (mesh / f"{s}{ext}").write_text("m")
        if with_caption: (aitk / f"{s}.txt").write_text(f"a car {s}")
        if with_preview: (aitk / f"{s}.png").write_bytes(b"\x89PNG")
        (ren / f"{s}.zip").write_bytes(b"PK")
    return str(mesh), str(aitk), str(ren)

def test_discover_finds_complete_objects(tmp_path):
    shas = [f"{i:064x}" for i in range(5)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    assert {c["sha256"] for c in cands} == set(shas)
    assert all(c["caption"].startswith("a car") for c in cands)

def test_filter_excludes_missing_caption_and_overlap(tmp_path):
    shas = [f"{i:064x}" for i in range(5)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    os.remove(os.path.join(aitk, f"{shas[0]}.txt"))
    cands = bs.discover_candidates(mesh, aitk, ren)
    included, excluded = bs.filter_candidates(cands, meshfleet_test_shas={shas[1]})
    inc = {c["sha256"] for c in included}
    assert shas[0] not in inc
    assert shas[1] not in inc
    assert len(inc) == 3

def test_sample_is_deterministic_and_sized(tmp_path):
    shas = [f"{i:064x}" for i in range(100)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    inc, _ = bs.filter_candidates(cands, meshfleet_test_shas=set())
    a = [c["sha256"] for c in bs.sample_locked(inc, n=10, seed=20260603)]
    b = [c["sha256"] for c in bs.sample_locked(inc, n=10, seed=20260603)]
    assert a == b and len(a) == 10

def test_split_counts(tmp_path):
    shas = [f"{i:064x}" for i in range(20)]
    mesh, aitk, ren = _make_pool(tmp_path, shas)
    cands = bs.discover_candidates(mesh, aitk, ren)
    inc, _ = bs.filter_candidates(cands, meshfleet_test_shas=set())
    sel = bs.sample_locked(inc, n=20, seed=20260603)
    rows = bs.assign_splits(sel, n_val=5, seed=20260603)
    splits = [r["split"] for r in rows]
    assert splits.count("val") == 5 and splits.count("train") == 15
