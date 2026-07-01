import os, importlib.util
spec = importlib.util.spec_from_file_location(
    "sg", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "stage_generated_glbs.py"))
sg = importlib.util.module_from_spec(spec); spec.loader.exec_module(sg)

def test_stage_creates_per_sha_symlinks(tmp_path):
    gen = tmp_path / "gen"; gen.mkdir()
    shas = [f"{i:064x}" for i in range(2)]
    for s in shas: (gen / f"sample_{s}.glb").write_text("glb")
    (gen / "ignore.txt").write_text("x"); (gen / "options.json").write_text("{}")
    staging = tmp_path / "stage"
    n = sg.stage(str(gen), str(staging))
    assert n == 2
    for s in shas:
        link = staging / s / f"{s}.glb"
        assert os.path.islink(link) and os.path.exists(link)
