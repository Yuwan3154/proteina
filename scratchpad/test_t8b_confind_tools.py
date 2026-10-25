"""Local unit test for t8b_confind_cache.py / t8b_confind_compare.py on a fake bucketed tree (SimpleNamespace graphs)."""
import argparse
import csv
import json
import os
import sys
import tempfile
import zlib
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t8b_confind_cache as tc
import t8b_confind_compare as cmp


def sym_map(L, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(L, L, generator=g) * 0.02
    return ((m + m.T) / 2).to(torch.float16)


def save_graph(path, L, cm=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = SimpleNamespace(coords=torch.zeros(L, 37, 3))
    if cm is not None:
        g.contact_map_confind = cm
    torch.save(g, path)


def write_tsv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "category", "status", "L_old", "L_new"], delimiter="\t")
        w.writeheader()
        for s, st, L in rows:
            w.writerow({"stem": s, "category": "ok", "status": st, "L_old": L, "L_new": L})


def write_status_csv(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "length", "status", "infer_sec", "save_sec", "error"])
        w.writeheader()
        for n in names:
            w.writerow({"file": n, "length": 0, "status": "processed", "infer_sec": 0, "save_sec": 0, "error": ""})


def expect_fail(fn, what):
    with pytest.raises(AssertionError) as e:
        fn()
    print(f"  raises as expected [{what}]: {str(e.value)[:80]}")


def test_metrics():
    ref = torch.zeros(4, 4)
    ref[0, 1] = ref[1, 0] = ref[2, 3] = ref[3, 2] = 0.5
    new = ref.clone()
    new[3, 2] = 0.005  # drop one contact
    new[0, 3] = 0.02  # add one
    j, agree, dp = cmp.map_metrics(new, ref)
    assert abs(j - 3 / 5) < 1e-6 and abs(agree - 14 / 16) < 1e-6 and abs(dp - 0.495) < 1e-6, (j, agree, dp)
    j0, _, dp0 = cmp.map_metrics(ref, ref)
    assert j0 == 1.0 and dp0 == 0.0
    je, _, _ = cmp.map_metrics(torch.zeros(3, 3), torch.zeros(3, 3))
    assert je == 0.0  # S1 convention: clamp_min(1) on the union
    st = cmp.tensor_stats(sym_map(5, 0))
    assert st["asym"] == 0.0 and st["dtype"] == "torch.float16" and st["n_nonfinite"] == 0
    print("test_metrics OK: J=0.6 / agree=0.875 / max|dp|=0.495 on the hand case; identical -> 1, 0")


def test_full_and_gate(tmp):
    fix, orig = os.path.join(tmp, "fix"), os.path.join(tmp, "orig")
    rows = [("1abc_A", "changed", 100), ("2xyz_B", "changed", 512), ("3203_C", "changed", 37),
            ("4u4u_A", "unchanged", 400), ("5v5v_A", "unchanged", 450), ("6w6w_A", "unchanged", 512),
            ("7q7q_A", "unchanged", 384), ("8r8r_A", "unchanged", 513)]
    tsv = os.path.join(tmp, "status.tsv")
    write_tsv(tsv, rows)
    bucket = {"1abc_A": "ab", "2xyz_B": "01", "3203_C": "3203", "4u4u_A": "u4", "5v5v_A": "v5", "6w6w_A": "w6",
              "7q7q_A": "q7", "8r8r_A": "r8"}
    for s, st, L in rows:
        if st == "changed":
            save_graph(os.path.join(fix, bucket[s], f"{s}.pt"), L)
        save_graph(os.path.join(orig, bucket[s], f"{s}.pt"), L, sym_map(L, zlib.crc32(s.encode()) % 1000))
    os.makedirs(os.path.join(fix, "ab"), exist_ok=True)
    open(os.path.join(fix, "ab", "junk.pt.tmp"), "w").close()

    out = os.path.join(tmp, "out")
    a = argparse.Namespace(status_tsv=tsv, tree=fix, workers=2, max_len=512,
                           out_cache=os.path.join(out, "cache.json"), out_stems=os.path.join(out, "stems.txt"))
    tc.build_full(a)
    cache = json.load(open(a.out_cache))
    assert [os.path.basename(p) for p, _, _ in cache] == ["2xyz_B.pt", "3203_C.pt", "1abc_A.pt"]  # path-sorted
    assert [r[0] for r in cache] == sorted(r[0] for r in cache)
    assert {os.path.basename(p): (L, cm) for p, L, cm in cache} == \
        {"1abc_A.pt": (100, False), "2xyz_B.pt": (512, False), "3203_C.pt": (37, False)}
    assert all(p.startswith(fix) for p, _, _ in cache)
    assert open(a.out_stems).read().split() == ["1abc_A", "2xyz_B", "3203_C"]
    print(f"test_full OK: {len(cache)} rows, real buckets resolved (3203_C -> {os.path.dirname(cache[1][0])[-4:]})")

    save_graph(os.path.join(fix, "zz", "1abc_A.pt"), 100)
    expect_fail(lambda: tc.build_full(a), "stem in two buckets")
    os.remove(os.path.join(fix, "zz", "1abc_A.pt"))
    save_graph(os.path.join(fix, "ab", "1abc_A.pt"), 100, sym_map(100, 1))
    expect_fail(lambda: tc.build_full(a), "changed file already has a map")
    save_graph(os.path.join(fix, "ab", "1abc_A.pt"), 99)
    expect_fail(lambda: tc.build_full(a), "coords L != L_new")
    save_graph(os.path.join(fix, "ab", "1abc_A.pt"), 100)
    os.remove(os.path.join(fix, "01", "2xyz_B.pt"))
    expect_fail(lambda: tc.build_full(a), "missing stem")
    save_graph(os.path.join(fix, "01", "2xyz_B.pt"), 512)

    gdir = os.path.join(tmp, "gate")
    g = argparse.Namespace(status_tsv=tsv, tree=orig, gate_dir=gdir, n=3, seed=0, lmin=385, lmax=512)
    tc.build_gate(g)
    man = list(csv.DictReader(open(os.path.join(gdir, "manifest.tsv")), delimiter="\t"))
    assert sorted(r["stem"] for r in man) == ["4u4u_A", "5v5v_A", "6w6w_A"]  # 384 and 513 excluded, changed excluded
    for r in man:
        assert r["gate_path"] == os.path.join(gdir, "processed", bucket[r["stem"]], f"{r['stem']}.pt")
    gc = json.load(open(os.path.join(gdir, "cache_gate.json")))
    assert all(cm for _, _, cm in gc) and len(gc) == 3
    ref = torch.load(os.path.join(gdir, "ref_maps.pt"), weights_only=False)
    assert all(torch.equal(ref[s], sym_map(L, zlib.crc32(s.encode()) % 1000)) for s, _, L in rows if s in ref)
    expect_fail(lambda: tc.build_gate(g), "gate dir exists")
    print("test_gate OK: 3 of 3 eligible unchanged stems copied into their original bucket names")
    return tsv, out, gdir, man, fix


def test_compare_and_check(tsv, out, gdir, man, fix):
    # simulate the GPU gate run: perturb two maps slightly, write a status CSV
    for i, r in enumerate(man):
        g = torch.load(r["gate_path"], weights_only=False)
        if i < 2:
            g.contact_map_confind = (g.contact_map_confind.float() + 0.003 * (i + 1)).clamp(0, 1).to(torch.float16)
        torch.save(g, r["gate_path"])
    gstat = os.path.join(out, "status_gate")
    write_status_csv(os.path.join(gstat, "band1_1_r0.csv"), [f"{r['stem']}.pt" for r in man[:2]])
    ga = argparse.Namespace(gate_dir=gdir, status_dir=gstat)
    expect_fail(lambda: cmp.run_gate(ga), "gate file without processed row")
    write_status_csv(os.path.join(gstat, "band1_1_r1.csv"), [f"{man[2]['stem']}.pt"])
    cmp.run_gate(ga)

    cache = json.load(open(os.path.join(out, "cache.json")))
    fstat = os.path.join(out, "status_full")
    ca = argparse.Namespace(cache=os.path.join(out, "cache.json"), status_tsv=tsv, status_dir=fstat,
                            ref_maps=os.path.join(gdir, "ref_maps.pt"), workers=2)
    for p, L, _ in cache:
        save_graph(p, L, sym_map(L, L))
    write_status_csv(os.path.join(fstat, "band1_9_r0.csv"), [os.path.basename(p) for p, _, _ in cache])
    cmp.run_check(ca)
    p0, L0, _ = cache[0]
    bad = sym_map(L0, 3).float()
    bad[0, 1] += 0.001
    save_graph(p0, L0, bad.to(torch.float16))
    expect_fail(lambda: cmp.run_check(ca), "asymmetric map")
    save_graph(p0, L0, sym_map(L0, 3).float())
    expect_fail(lambda: cmp.run_check(ca), "float32 map")
    save_graph(p0, L0)
    expect_fail(lambda: cmp.run_check(ca), "missing map")
    print("test_compare_and_check OK")


if __name__ == "__main__":
    test_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        test_compare_and_check(*test_full_and_gate(tmp))
    print("ALL TESTS PASSED")
