"""Local test of t8b_overlay_pack.py + t8b_data_dir.py on a tiny synthetic pdb_train / fix tree / T8 pack (built with
the real scratchpad/build_pack.py). Run: python -m pytest scratchpad/test_t8b_overlay_data_dir.py -q"""
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t8b_data_dir as tdd  # noqa: E402
from proteinfoundation.datasets.pdb_data import _processed_path_sharded  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
S = REPO / "scratchpad"
MAN = {"depth_table": {"ab": 2, "cd": 3}, "hash_subbucket_for": {"cde": 1}}
# stem: (status, L_old, L_new); 9xyz_A has no bucket in MAN (flat path)
STEMS = {"1ab0_A": ("unchanged", 20, 20), "1ab1_B": ("changed", 22, 22), "2ab2_A": ("changed", 18, 21),
         "3cde_A": ("unchanged", 25, 25), "4cde_B": ("changed", 30, 29), "9xyz_A": ("unchanged", 16, 16)}
EXTRA_CSV = "7ab7_A"  # a csv row outside the table/pack (like pdb_train chains outside T8)
DS_YAML = """datamodule:
  data_dir: {root}/pdb_train_t8b/
  pack_path: unused
  format: "cif"
  overwrite: False
  regenerate_missing: {regen}
  sampling_mode: "cluster-random"
  dataselector:
    _target_: "proteinfoundation.datasets.pdb_data.PDBDataSelector"
    data_dir: {root}/pdb_train_t8b/
    fraction: 1
    molecule_type: "protein"
    experiment_types: ["diffraction", "EM"]
    min_length: 50
    max_length: 512
    oligomeric_min: null
    oligomeric_max: null
    best_resolution: 0.0
    worst_resolution: 5.0
    has_ligands: []
    remove_ligands: []
    remove_non_standard_residues: True
    remove_pdb_unavailable: True
    max_deposition_date: "2019-08-28"
    exclude_ids: []
  datasplitter:
    _target_: "proteinfoundation.datasets.pdb_data.PDBDataSplitter"
    data_dir: {root}/pdb_train_t8b/
    train_val_test: [0.98, 0.019, 0.001]
    split_type: "sequence_similarity"
    split_sequence_similarity: 0.25
    overwrite_sequence_clusters: False
    exclude_ids: []
    exclude_ids_from_file: {root}/pdb_train/rosetta_decoys.txt
    drop_ids_from_file: {root}/pdb_train/unusable_chains.txt
    split_dir: {root}/split
"""


def graph(stem, L, seed, with_map=True):
    g = torch.Generator().manual_seed(seed)
    d = Data(id=stem, coords=torch.randn(L, 37, 3, generator=g), coord_mask=torch.rand(L, 37, generator=g) > 0.3,
             residue_type=torch.randint(0, 20, (L,), generator=g), residue_id=[f"A:{i}:" for i in range(L)],
             bfactor=torch.full((L,), float("nan")), ext_lig=torch.zeros(L, dtype=torch.int8))
    d.coord_mask[:, 1] = True  # CA present -> processed_length == L
    if with_map:
        m = torch.rand(L, L, generator=g)
        d.contact_map_confind = ((m + m.T) / 2).to(torch.float16)
    return d


def save(path, g):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g, path)


def run(*args, ok=True):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert (r.returncode == 0) == ok, f"rc {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout + r.stderr


@pytest.fixture()
def world(tmp_path):
    root = tmp_path
    orig, fix = root / "pdb_train", root / "fix" / "processed"
    (orig / "processed").mkdir(parents=True)
    json.dump(MAN, open(orig / "shard_manifest.json", "w"))
    for i, (s, (st, L0, L1)) in enumerate(STEMS.items()):
        save(_processed_path_sharded(orig / "processed", s, MAN), graph(s, L0, i))
        if st == "changed":
            save(_processed_path_sharded(fix, s, MAN), graph(s, L1, 100 + i))
    save(_processed_path_sharded(orig / "processed", EXTRA_CSV, MAN), graph(EXTRA_CSV, 19, 50))
    with open(root / "status.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["stem", "category", "status", "L_old", "L_new", "fields"])
        for s, (st, L0, L1) in STEMS.items():
            w.writerow([s, "ok" if st == "unchanged" else "merged", st, L0, L1, "" if st == "unchanged" else "coords"])
    names = list(STEMS) + [EXTRA_CSV]
    lens = [STEMS[s][1] if s in STEMS else 19 for s in names]
    pd.DataFrame({"id": names, "pdb": [s[:4] for s in names], "chain": [s[5:] for s in names], "length": lens,
                  "name": ['PROTEIN, "X" KINASE'] * len(names), "sequence": ["M" * L for L in lens],
                  "resolution": [2.07, 1.5, 3.3333333333333335, 0.1, 2.0, 1.0, 2.5],
                  "ligands": ["['HEM', 'SO4']"] * len(names), "processed_length": lens}).to_csv(
        orig / "df_pdb_f1_minl50_maxl512_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv",
        index=False)
    tsv = orig / "cluster_seqid_0.25_df_pdb_f1_minl50_maxl512_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.tsv"
    tsv.write_text("1ab0_A\t1ab0_A\n1ab0_A\t1ab1_B\n2ab2_A\t2ab2_A\n3cde_A\t3cde_A\n3cde_A\t4cde_B\n9xyz_A\t9xyz_A\n7ab7_A\t7ab7_A\n")
    (orig / "rosetta_decoys.txt").write_text("9xyz_A\n")
    (orig / "unusable_chains.txt").write_text("7ab7_A\n")
    (root / "split").mkdir()
    for sp, ids in (("train", ["1ab0_A", "1ab1_B", "2ab2_A"]), ("val", ["3cde_A", "4cde_B"]), ("test", ["9xyz_A"])):
        (root / "split" / f"{sp}_chain_ids.txt").write_text("\n".join(ids) + "\n")
    for regen in ("False", "True"):
        (root / f"ds_regen{regen}.yaml").write_text(DS_YAML.format(root=root, regen=regen))
    (root / "stems.txt").write_text("\n".join(STEMS) + "\n")
    run(S / "build_pack.py", root / "stems.txt", orig, root / "t8.pack")
    return root


def pack_args(root, out="t8b.pack"):
    return ["--old-pack", root / "t8.pack", "--status-tsv", root / "status.tsv", "--fix-processed", root / "fix" / "processed",
            "--orig-data-dir", root / "pdb_train", "--out", root / out, "--workers", 2]


def dd_args(root, regen="False"):
    return ["--config", root / f"ds_regen{regen}.yaml", "--orig-data-dir", root / "pdb_train", "--fix-processed",
            root / "fix" / "processed", "--status-tsv", root / "status.tsv", "--out", root / "pdb_train_t8b"]


def md5(b):
    return hashlib.md5(b).hexdigest()


def test_end_to_end(world):
    root = world
    log = run(S / "t8b_overlay_pack.py", "build", *pack_args(root))
    print(log.strip().splitlines()[-1])
    old, new = np.load(root / "t8.pack.idx.npz", allow_pickle=True), np.load(root / "t8b.pack.idx.npz", allow_pickle=True)
    assert list(old["stems"]) == list(new["stems"]) == sorted(STEMS)
    blob = (root / "t8b.pack").read_bytes()
    for s, o, n, m, om in zip(new["stems"], new["offsets"], new["lengths"], new["md5"], old["md5"]):
        src = _processed_path_sharded((root / "fix" / "processed") if STEMS[s][0] == "changed" else (root / "pdb_train" / "processed"), s, MAN)
        assert blob[o:o + n] == src.read_bytes() and md5(blob[o:o + n]) == m, s
        assert (m == om) == (STEMS[s][0] == "unchanged"), s
    print(run(S / "t8b_overlay_pack.py", "verify", *pack_args(root)).strip().splitlines()[-1])
    # rerun is deterministic and a stale .partial is overwritten
    keep = blob, (root / "t8b.pack.idx.npz").read_bytes()
    for f in ("t8b.pack", "t8b.pack.idx.npz", "t8b.pack.provenance.json"):
        os.remove(root / f)
    (root / "t8b.pack.partial").write_bytes(b"junk")
    run(S / "t8b_overlay_pack.py", "build", *pack_args(root))
    assert ((root / "t8b.pack").read_bytes(), (root / "t8b.pack.idx.npz").read_bytes()) == keep
    assert "exists" in run(S / "t8b_overlay_pack.py", "build", *pack_args(root), ok=False)

    print(run(S / "t8b_data_dir.py", "build", *dd_args(root)).strip().splitlines()[-1])
    out = root / "pdb_train_t8b"
    for s, (st, _, _) in STEMS.items():
        link = _processed_path_sharded(out / "processed", s, MAN)
        want = _processed_path_sharded((root / "fix" / "processed") if st == "changed" else (root / "pdb_train" / "processed"), s, MAN)
        assert link.is_symlink() and os.readlink(link) == str(want), s
    assert not _processed_path_sharded(out / "processed", EXTRA_CSV, MAN).exists()
    df = pd.read_csv(out / tdd.names(root / "ds_regenFalse.yaml")[0]).set_index("id")
    assert df.loc["2ab2_A", "processed_length"] == 21 and df.loc["4cde_B", "processed_length"] == 29
    assert df.loc["1ab0_A", "processed_length"] == 20 and df.loc[EXTRA_CSV, "processed_length"] == 19
    log = run(S / "t8b_data_dir.py", "verify", *dd_args(root), "--pack", root / "t8b.pack", "--n-sample", 2, "--seed", 0)
    print("\n".join(l for l in log.splitlines() if l.startswith("[")))
    assert "[verify] PASS" in log
    assert "exists" in run(S / "t8b_data_dir.py", "build", *dd_args(root), ok=False)
    # the T8 setting regenerate_missing True is refused by verify
    assert "both must be False" in run(S / "t8b_data_dir.py", "verify", *dd_args(root, "True"), "--pack", root / "t8b.pack",
                                       "--n-sample", 2, "--seed", 0, ok=False)
    # a link to the wrong tree is caught
    link = _processed_path_sharded(out / "processed", "1ab1_B", MAN)
    os.remove(link)
    os.symlink(_processed_path_sharded(root / "pdb_train" / "processed", "1ab1_B", MAN), link)
    assert "do not match" in run(S / "t8b_data_dir.py", "verify", *dd_args(root), "--pack", root / "t8b.pack",
                                 "--n-sample", 2, "--seed", 0, ok=False)


@pytest.mark.parametrize("damage, msg", [
    ("drop_map", "no contact_map_confind"), ("fp32_map", "contact_map_confind dtype torch.float32"),
    ("bad_shape", "contact_map_confind shape"), ("extra_key", "keys +['extra']"), ("coord_dtype", "coords dtype")])
def test_guard(world, damage, msg):
    root = world
    p = _processed_path_sharded(root / "fix" / "processed", "4cde_B", MAN)
    g = torch.load(p, weights_only=False)
    if damage == "drop_map":
        del g.contact_map_confind
    elif damage == "fp32_map":
        g.contact_map_confind = g.contact_map_confind.float()
    elif damage == "bad_shape":
        g.contact_map_confind = g.contact_map_confind[:-1, :-1]
    elif damage == "extra_key":
        g.extra = torch.zeros(1)
    else:
        g.coords = g.coords.double()
    torch.save(g, p)
    log = run(S / "t8b_overlay_pack.py", "build", *pack_args(root), ok=False)
    assert msg in log and "guard failed: nothing written" in log, log[-2000:]
    assert not (root / "t8b.pack").exists() and not (root / "t8b.pack.partial").exists()


def test_set_mismatch_and_tamper(world):
    root = world
    lines = (root / "status.tsv").read_text().splitlines()
    (root / "status.tsv").write_text("\n".join(lines[:-1]) + "\n")  # drop 9xyz_A from the table
    assert "in pack not table 1" in run(S / "t8b_overlay_pack.py", "build", *pack_args(root), ok=False)
    (root / "status.tsv").write_text("\n".join(lines) + "\n")
    run(S / "t8b_overlay_pack.py", "build", *pack_args(root))
    z = np.load(root / "t8b.pack.idx.npz", allow_pickle=True)
    i = list(z["stems"]).index("3cde_A")
    with open(root / "t8b.pack", "r+b") as fh:  # flip one byte of an unchanged record
        fh.seek(int(z["offsets"][i]) + 100)
        b = fh.read(1)
        fh.seek(int(z["offsets"][i]) + 100)
        fh.write(bytes([b[0] ^ 1]))
    assert "verify FAILED" in run(S / "t8b_overlay_pack.py", "verify", *pack_args(root), ok=False)


def test_snapshot_sees_writes(tmp_path):
    (tmp_path / "a").write_text("x")
    before = tdd.snapshot(str(tmp_path))
    (tmp_path / "a").write_text("y")
    assert before != tdd.snapshot(str(tmp_path))
    before = tdd.snapshot(str(tmp_path))
    (tmp_path / "b").mkdir()
    assert before != tdd.snapshot(str(tmp_path))


def test_reader_detects_drift(world):
    """Control for the reader gate: a fixed .pt rewritten after the pack build -> pack and data_dir disagree -> FAIL."""
    root = world
    run(S / "t8b_overlay_pack.py", "build", *pack_args(root))
    run(S / "t8b_data_dir.py", "build", *dd_args(root))
    for s in ("1ab1_B", "2ab2_A", "4cde_B"):
        p = _processed_path_sharded(root / "fix" / "processed", s, MAN)
        g = torch.load(p, weights_only=False)
        g.coords[0, 0, 0] += 1.0
        torch.save(g, p)
    log = run(S / "t8b_data_dir.py", "verify", *dd_args(root), "--pack", root / "t8b.pack", "--n-sample", 2, "--seed", 0,
              ok=False)
    assert "reader equality FAILED" in log and "reader diff ['coords']" in log, log[-2000:]
    assert "source .pt != index {'changed': 3}" in run(S / "t8b_overlay_pack.py", "verify", *pack_args(root), ok=False)


def test_same_graph():
    a = Data(x=torch.tensor([1.0, float("nan")]), r=["a"], n=np.array([1.0, np.nan]))
    assert tdd.same_graph(a, a.clone()) == []
    b = a.clone()
    b.x = torch.tensor([2.0, float("nan")])
    b.r = ["b"]
    b.z = torch.zeros(1)
    assert tdd.same_graph(a, b) == ["z", "r", "x"]
    c = a.clone()
    c.x = c.x.double()
    assert tdd.same_graph(a, c) == ["x"]
