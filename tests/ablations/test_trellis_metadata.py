import os, csv, json, importlib.util
spec = importlib.util.spec_from_file_location(
    "trellis_meta",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "build_trellis_metadata.py"))
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)

def _manifest(tmp_path):
    shas=[f"{i:064x}" for i in range(3)]
    meshes=tmp_path/"meshes"; meshes.mkdir()
    rows=[]
    for i,s in enumerate(shas):
        ext=".glb"
        (meshes/f"{s}{ext}").write_text("m")
        rows.append({"sha256":s,"asset_path":str(meshes/f"{s}{ext}"),
                     "caption":f'A car number {i}',"split":"train" if i<2 else "val"})
    p=tmp_path/"m.csv"
    with open(p,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sha256","asset_path","caption","split"]); w.writeheader(); w.writerows(rows)
    return str(p), shas

def test_metadata_schema_and_caption_json(tmp_path):
    man, shas = _manifest(tmp_path)
    df = tm.build_metadata_df(man, split="train")
    cols=list(df.columns)
    for c in ["sha256","file_identifier","aesthetic_score","captions","local_path",
              "rendered","voxelized","num_voxels","cond_rendered",
              "feature_dinov2_vitl14_reg","ss_latent_ss_enc_conv3d_16l8_fp16",
              "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"]:
        assert c in cols, c
    assert len(df)==2   # only train rows
    cap=json.loads(df.iloc[0]["captions"]); assert isinstance(cap,list) and cap[0].startswith("A car")
    assert df.iloc[0]["local_path"].startswith("./raw/") and df.iloc[0]["local_path"].endswith(".glb")
    assert bool(df.iloc[0]["rendered"]) is False

def test_stage_meshes_symlinks(tmp_path):
    man, shas = _manifest(tmp_path)
    out=tmp_path/"trellis_train"
    df = tm.build_metadata_df(man, split="train")
    tm.stage_meshes(df, str(out), man)
    assert os.path.islink(out/"raw"/f"{shas[0]}.glb")
