"""Local test of the June pad-fix chain: t8b_padfix_list.py (batch rule, list, table), t8b_confind_cache.py --padfix,
t8b_confind_compare.py --padfix-check, then the UNCHANGED overlay-pack / data_dir builders on the new table (world of
test_t8b_overlay_data_dir.py: bucketed, hash-sub-bucketed and flat stems). Run: python -m pytest scratchpad/test_t8b_padfix.py -q"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t8b_confind_cache as tc  # noqa: E402
import t8b_confind_compare as cmp  # noqa: E402
import t8b_index_diff as tid  # noqa: E402
import t8b_padfix_list as pl  # noqa: E402
from test_t8b_overlay_data_dir import MAN, STEMS, S, dd_args, pack_args, run, world  # noqa: E402,F401
from proteinfoundation.datasets.pdb_data import _processed_path_sharded  # noqa: E402

HEAD = ["file", "length", "status", "infer_sec", "save_sec", "error"]
# bs 3: batch 0 = a b c; x = load failure (written while batch 1 is built); batch 1 = d e f (e: save failure, keeps the
# batch's infer_sec); final partial batch = g h
JUNE = [("a.pt", 10, "processed", 0.1, ""), ("b.pt", 10, "processed", 0.1, ""), ("c.pt", 9, "processed", 0.1, ""),
        ("x.pt", 9, "failed", 0.0, "no_coords"), ("d.pt", 9, "processed", 0.2, ""), ("e.pt", 8, "failed", 0.2, "OSError"),
        ("f.pt", 9, "processed", 0.2, ""), ("g.pt", 5, "processed", 0.3, ""), ("h.pt", 4, "processed", 0.3, "")]


def write_june(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        for f, L, st, inf, err in rows:
            w.writerow([f, L, st, inf, 0.0, err])
    return str(path)


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_june_batches(tmp_path):
    b, lf, sf = pl.june_batches(write_june(tmp_path / "j.csv", JUNE), 3)
    assert [[r["file"] for r in x] for x in b] == [["a.pt", "b.pt", "c.pt"], ["d.pt", "e.pt", "f.pt"], ["g.pt", "h.pt"]]
    assert [r["file"] for r in lf] == ["x.pt"] and [r["file"] for r in sf] == ["e.pt"]
    bad = {"misaligned (a row lost)": JUNE[1:],
           "adjacent batches share infer_sec": [r if r[3] != 0.2 else (*r[:3], 0.1, r[4]) for r in JUNE],
           "later batch has a longer chain": [r if r[0] != "f.pt" else ("f.pt", 11, "processed", 0.2, "") for r in JUNE]}
    for what, rows in bad.items():
        with pytest.raises(AssertionError):
            pl.june_batches(write_june(tmp_path / "bad.csv", rows), 3)
        print(f"  raises as expected [{what}]")


def list_world(tmp_path, table_rows, june_rows=JUNE, cache_rel=None, man=None):
    """cache_rel: [(path relative to processed/, L)] in cache order; default = one flat copy per CSV row, in CSV order."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    june = write_june(tmp_path / "j.csv", june_rows)
    tsv = write_tsv(tmp_path / "s.tsv", ["stem", "category", "status", "L_old", "L_new", "fields"], table_rows)
    (tmp_path / "tmpl.txt").write_text("q\nc\nd\n")
    (tmp_path / "elig.txt").write_text("c\n")
    (tmp_path / "val.txt").write_text("h\n")
    data = tmp_path / "pdb_train"
    data.mkdir(exist_ok=True)
    json.dump(man or {}, open(data / "shard_manifest.json", "w"))
    rel = cache_rel if cache_rel is not None else [(f, L) for f, L, *_ in june_rows]
    # spelled as the June cache spells it (<data>/processed/...), with a redundant ./ to exercise the normalisation
    json.dump([[f"{data}/processed/./{r}", L, False] for r, L in rel], open(tmp_path / "cache.json", "w"))
    return argparse.Namespace(june_csv=june, length_cache=str(tmp_path / "cache.json"), data_dir=str(data), max_len=512,
                              bs=3, status_tsv=tsv, template_list=str(tmp_path / "tmpl.txt"),
                              eligible=str(tmp_path / "elig.txt"), val_fixed=str(tmp_path / "val.txt"), n_parts=2,
                              out_tsv=str(tmp_path / "out" / "padfix_list.tsv"))


TABLE = [("a", "ok", "unchanged", 10, 10, ""), ("b", "merged", "changed", 10, 11, "coords"),
         ("c", "ok", "unchanged", 9, 9, ""), ("d", "ok", "unchanged", 9, 9, ""), ("f", "ok", "unchanged", 9, 9, ""),
         ("g", "ok", "unchanged", 5, 5, ""), ("h", "ok", "unchanged", 4, 4, "")]


# pdb_train held some stems twice (flat processed/<stem>.pt + the bucket copy T8b reads) -> two CSV rows per basename.
# The CSV alone cannot tell which copy sat in which batch; the cache order does (stable sort, equal L keep cache order).
DUP_MAN = {"depth_table": {k: 2 for k in ("aa", "bb", "cc", "dd", "gg", "hh")}}
JUNE_DUP = [("1aaa_A.pt", 10, "processed", 0.1, ""), ("1bbb_A.pt", 10, "processed", 0.1, ""),
            ("1ccc_A.pt", 9, "processed", 0.1, ""),
            ("1ccc_A.pt", 9, "processed", 0.2, ""), ("1ddd_A.pt", 9, "processed", 0.2, ""), ("1ddd_A.pt", 9, "processed", 0.2, ""),
            ("1ggg_A.pt", 5, "processed", 0.3, ""), ("1hhh_A.pt", 4, "processed", 0.3, "")]
DUP_TABLE = [(s, "ok", "unchanged", L, L, "") for s, L in
             (("1aaa_A", 10), ("1bbb_A", 10), ("1ccc_A", 9), ("1ddd_A", 9), ("1ggg_A", 5), ("1hhh_A", 4))]


def dup_cache(c_read_first):
    """Cache order: the 1ccc_A copy listed first among the L=9 rows lands in batch 0 (padded, max L 10)."""
    c = [("cc/1ccc_A.pt", 9), ("1ccc_A.pt", 9)]
    return ([("aa/1aaa_A.pt", 10), ("bb/1bbb_A.pt", 10)] + (c if c_read_first else c[::-1])
            + [("1ddd_A.pt", 9), ("dd/1ddd_A.pt", 9), ("gg/1ggg_A.pt", 5), ("hh/1hhh_A.pt", 4)])


def test_list_duplicate_copies(tmp_path):
    got = {}
    for read_first in (False, True):
        a = list_world(tmp_path / f"read_first_{read_first}", DUP_TABLE, JUNE_DUP, dup_cache(read_first), DUP_MAN)
        pl.cmd_list(a)
        with open(a.out_tsv, newline="") as fh:
            got[read_first] = {r["stem"]: r for r in csv.DictReader(fh, delimiter="\t")}
    # flat copy padded, read (bucket) copy unpadded -> NOT listed; read copy padded -> listed
    assert sorted(got[False]) == ["1hhh_A"], sorted(got[False])
    assert sorted(got[True]) == ["1ccc_A", "1hhh_A"], sorted(got[True])
    c = got[True]["1ccc_A"]
    assert (c["batch"], c["batch_max_L"], c["pad"], c["n_june_rows"]) == ("0", "10", "1", "2"), c
    assert got[True]["1hhh_A"]["n_june_rows"] == "1"
    print("  duplicate copies: listed only when the copy T8b reads (bucket) was padded")
    bad = {"cache misses a path (replay != CSV batches)": (dup_cache(True)[:-1], "replay does not reproduce"),
           # replay still matches (same basenames), but the copy T8b reads (dd/) is not a cache path
           "read copy absent from the cache": ([(("zz/1ddd_A.pt", L) if r == "dd/1ddd_A.pt" else (r, L))
                                                for r, L in dup_cache(True)], "is in no June batch"),
           "cache path twice after normalisation": (dup_cache(True) + [("aa//1aaa_A.pt", 10)], "not unique")}
    for what, (rel, msg) in bad.items():
        a = list_world(tmp_path / what.replace(" ", "_").replace("/", "_"), DUP_TABLE, JUNE_DUP, rel, DUP_MAN)
        with pytest.raises(AssertionError, match=msg):
            pl.cmd_list(a)
        print(f"  raises as expected [{what}]")


def test_list_and_table(tmp_path):
    a = list_world(tmp_path, TABLE)
    pl.cmd_list(a)
    rows = list(csv.DictReader(open(a.out_tsv), delimiter="\t"))
    assert [(r["stem"], r["batch"], r["batch_max_L"], r["pad"], r["in_template_list"], r["index_part"], r["eligible"],
             r["val_fixed"]) for r in rows] == [("c", "0", "10", "1", "1", "1", "1", "0"),
                                                 ("h", "2", "5", "1", "0", "", "0", "1")]
    with pytest.raises(AssertionError):
        pl.cmd_list(a)  # refuses an existing list
    for what, tbl in {"unchanged stem absent from June": TABLE + [("zz", "ok", "unchanged", 3, 3, "")],
                      "unchanged stem with a June save failure": TABLE + [("e", "ok", "unchanged", 8, 8, "")],
                      "June L != L_old": [r if r[0] != "d" else ("d", "ok", "unchanged", 8, 8, "") for r in TABLE]}.items():
        b = list_world(tmp_path / what.replace(" ", "_"), tbl)
        with pytest.raises(AssertionError):
            pl.cmd_list(b)
        print(f"  raises as expected [{what}]")

    chk = write_tsv(tmp_path / "chk.tsv", ["stem"], [["c"]])
    drop = write_tsv(tmp_path / "drop.tsv", ["stem", "reason"], [["h", "map_identical_to_stored"]])
    t = argparse.Namespace(status_tsv=a.status_tsv, padfix_list=a.out_tsv, check_tsv=chk, dropped_tsv=drop,
                           out=str(tmp_path / "s2.tsv"))
    old = open(a.status_tsv).read()
    pl.cmd_table(t)
    assert open(a.status_tsv).read() == old
    new = {r["stem"]: r for r in csv.DictReader(open(t.out), delimiter="\t")}
    for r in csv.DictReader(open(a.status_tsv), delimiter="\t"):  # only the checked stem moves; dropped h stays
        want = dict(r, status="changed", category="confind_pad", fields="contact_map_confind") if r["stem"] == "c" else r
        assert new[r["stem"]] == want, r["stem"]
    with pytest.raises(AssertionError):
        pl.cmd_table(t)  # refuses an existing output
    nodrop = write_tsv(tmp_path / "drop0.tsv", ["stem", "reason"], [])
    for what, kw in {"check + dropped miss a listed stem": {"dropped_tsv": nodrop},
                     "a stem both checked and dropped": {"dropped_tsv": write_tsv(tmp_path / "drop2.tsv", ["stem", "reason"],
                                                                                  [["h", "x"], ["c", "x"]])}}.items():
        t2 = argparse.Namespace(**{**vars(t), **kw, "out": str(tmp_path / "s3.tsv")})
        with pytest.raises(AssertionError):
            pl.cmd_table(t2)
        assert not os.path.exists(t2.out)
        print(f"  table raises as expected [{what}]")


def fake_gpu(paths, bump):
    """Stand-in for the bs-1 recompute: change row/col L-1 by bump (plus a tiny change elsewhere), stay symmetric."""
    for p in paths:
        g = torch.load(p, weights_only=False)
        m = g.contact_map_confind.float()
        L = m.shape[0]
        m[L - 1, :] = (m[L - 1, :] + bump).clamp(0, 1)
        m[:, L - 1] = m[L - 1, :]
        m[0, 1] = m[1, 0] = (m[0, 1] + 0.001).clamp(0, 1)
        g.contact_map_confind = m.to(torch.float16)
        torch.save(g, p)


def test_padfix_end_to_end(world):
    root = world
    orig, fix = str(root / "pdb_train" / "processed"), str(root / "fix" / "processed")
    pad = ["3cde_A", "9xyz_A"]  # hash-sub-bucketed, flat; 1ab0_A stays unchanged so verify samples both classes
    lst = write_tsv(root / "padfix_list.tsv", ["stem"], [[s] for s in pad])
    out = root / "padfix"
    ca = argparse.Namespace(padfix_list=lst, status_tsv=str(root / "status.tsv"), tree=orig, fix_tree=fix, workers=2,
                            out_cache=str(out / "cache.json"), out_stems=str(out / "stems.txt"))
    tc.build_padfix(ca)
    cache = json.load(open(ca.out_cache))
    want = {s: str(_processed_path_sharded(root / "fix" / "processed", s, MAN)) for s in pad}
    assert sorted((p, L, cm) for p, L, cm in cache) == sorted((want[s], STEMS[s][2], True) for s in pad)
    for s in pad:  # the copy lands exactly where the overlay / data_dir builders look (manifest path)
        assert open(want[s], "rb").read() == open(_processed_path_sharded(root / "pdb_train" / "processed", s, MAN), "rb").read()
    tc.build_padfix(ca)  # rerun over identical copies: no-op
    bad = argparse.Namespace(**{**vars(ca), "padfix_list": write_tsv(root / "bad_list.tsv", ["stem"], [["1ab1_B"]])})
    with pytest.raises(AssertionError):
        tc.build_padfix(bad)  # a changed stem is never a pad-fix target
    with pytest.raises(AssertionError):
        tc.build_padfix(argparse.Namespace(**{**vars(ca), "fix_tree": orig + "/ab"}))  # nested trees

    fake_gpu([p for p, _, _ in cache], 0.3)
    with pytest.raises(AssertionError):
        tc.build_padfix(ca)  # after the recompute a rerun of the copy step is refused (copies != originals)
    sdir = out / "status"
    sdir.mkdir()
    write_june(sdir / "band1_1_r0.csv", [(f"{s}.pt", 0, "processed", 0, "") for s in pad[:-1]])
    # 9xyz_A: the recompute reproduces the stored map exactly -> dropped from the fix, stays unchanged
    p9 = want["9xyz_A"]
    g = torch.load(p9, weights_only=False)
    g.contact_map_confind = torch.load(_processed_path_sharded(root / "pdb_train" / "processed", "9xyz_A", MAN),
                                       weights_only=False).contact_map_confind
    torch.save(g, p9)
    ka = argparse.Namespace(cache=ca.out_cache, tree=orig, fix_tree=fix, status_dir=str(sdir),
                            out_tsv=str(out / "check.tsv"), dropped_tsv=str(out / "dropped.tsv"), workers=2)
    with pytest.raises(AssertionError):
        cmp.run_padfix_check(ka)  # pad[-1] has no processed status row
    assert not os.path.exists(ka.out_tsv) and not os.path.exists(ka.dropped_tsv)
    with open(sdir / "band1_1_r0.csv", "a") as fh:
        fh.write(f"{pad[-1]}.pt,0,processed,0,0,\n")
    cmp.run_padfix_check(ka)
    chk = list(csv.DictReader(open(ka.out_tsv), delimiter="\t"))
    assert [r["stem"] for r in chk] == ["3cde_A"] and chk[0]["max_in_last"] == "True"
    assert list(csv.DictReader(open(ka.dropped_tsv), delimiter="\t")) == [{"stem": "9xyz_A", "reason": "map_identical_to_stored"}]

    # damage control on the kept copy: another key changed -> fails, nothing written
    p0 = want["3cde_A"]
    keep = open(p0, "rb").read()
    g = torch.load(p0, weights_only=False)
    g.coords[0, 0, 0] += 1.0
    torch.save(g, p0)
    bad = argparse.Namespace(**{**vars(ka), "out_tsv": str(out / "x.tsv"), "dropped_tsv": str(out / "xd.tsv")})
    with pytest.raises(AssertionError):
        cmp.run_padfix_check(bad)
    assert not os.path.exists(bad.out_tsv) and not os.path.exists(bad.dropped_tsv)
    open(p0, "wb").write(keep)

    ta = argparse.Namespace(status_tsv=str(root / "status.tsv"), padfix_list=lst, check_tsv=ka.out_tsv,
                            dropped_tsv=ka.dropped_tsv, out=str(root / "status_padfix.tsv"))
    pl.cmd_table(ta)
    st2 = {r["stem"]: r["status"] for r in csv.DictReader(open(ta.out), delimiter="\t")}
    assert st2["3cde_A"] == "changed" and st2["9xyz_A"] == "unchanged" and st2["1ab0_A"] == "unchanged"
    # the unchanged builders on the new table: overlay pack (T8 pack as OLD) + verify, data_dir build + verify
    pa = [str(x) for x in pack_args(root, "t8b_v2.pack")]
    pa[pa.index("--status-tsv") + 1] = ta.out
    log = run(S / "t8b_overlay_pack.py", "build", *pa)
    assert "[done]" in log, log[-2000:]
    log = run(S / "t8b_overlay_pack.py", "verify", *pa)
    print(log.strip().splitlines()[-1])
    z = np.load(root / "t8b_v2.pack.idx.npz", allow_pickle=True)
    blob = (root / "t8b_v2.pack").read_bytes()
    for s, o, n in zip(z["stems"], z["offsets"], z["lengths"]):
        if s in pad:
            src = want[s] if s == "3cde_A" else _processed_path_sharded(root / "pdb_train" / "processed", s, MAN)
            assert blob[o:o + n] == open(src, "rb").read(), s
    da = [str(x) for x in dd_args(root)]
    da[da.index("--status-tsv") + 1] = ta.out
    da[da.index("--out") + 1] = str(root / "pdb_train_t8b")  # the world yaml's data_dir
    print(run(S / "t8b_data_dir.py", "build", *da).strip().splitlines()[-1])
    assert os.readlink(_processed_path_sharded(root / "pdb_train_t8b" / "processed", "3cde_A", MAN)) == want["3cde_A"]
    assert os.readlink(_processed_path_sharded(root / "pdb_train_t8b" / "processed", "9xyz_A", MAN)) == \
        str(_processed_path_sharded(root / "pdb_train" / "processed", "9xyz_A", MAN))
    log = run(S / "t8b_data_dir.py", "verify", *da, "--pack", root / "t8b_v2.pack", "--n-sample", 3, "--seed", 0)
    assert "[verify] PASS" in log, log[-3000:]
    print("end-to-end OK: copies -> check -> new table -> overlay v2 + data_dir on the unchanged builders")


def fake_index(native_he):
    """Merge-format index: chains A (2 templates) and B (1 template); native_he = {stem: he bytes of its native row}."""
    rows = [("A", True), ("A@rw10#0", False), ("A@rw20#1", False), ("B", True), ("B@rw10#0", False)]
    he = [native_he.get(r, [1, 0, 1]) if nat else [k, k + 1] for k, (r, nat) in enumerate(rows)]
    feat = [[0.5 * k, float("nan")] * len(h) for k, h in enumerate(he)]
    runs = [[[0, 3], [5, 7]] for _ in rows]
    align = [[] if nat else [k, -1, k] for k, (_, nat) in enumerate(rows)]
    off = lambda xs: torch.tensor([0] + list(np.cumsum([len(x) for x in xs])), dtype=torch.int64)
    return {"ids": [r for r, _ in rows], "cluster_of": torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
            "members_flat": torch.tensor([1, 2, 4], dtype=torch.int32), "members_offset": torch.tensor([0, 2, 3]),
            "seq_hash": torch.tensor([7, 7, 7, 9, 9]), "row_tm": torch.tensor([1, .6, .7, 1, .8], dtype=torch.float16),
            "row_is_native": torch.tensor([n for _, n in rows]), "row_rewind": torch.tensor([0, 10, 20, 0, 10], dtype=torch.int16),
            "row_usalign_tm": torch.tensor([1, .5, .6, 1, .7], dtype=torch.float16),
            "he_size": torch.tensor([len(h) for h in he], dtype=torch.int16),
            "he_flat": torch.tensor([x for h in he for x in h], dtype=torch.uint8), "he_offset": off(he),
            "feat_flat": torch.tensor([x for f in feat for x in f], dtype=torch.float16), "feat_offset": off(feat),
            "runs_flat": torch.tensor([x for r in runs for x in r], dtype=torch.int16), "runs_offset": off(runs),
            "align_flat": torch.tensor([x for a in align for x in a], dtype=torch.int16), "align_offset": off(align),
            "pair_feature_mean": torch.ones(2), "contact_def": "x"}


def test_index_diff():
    old = fake_index({})
    assert tid.diff(old, fake_index({"B": [1, 1, 1, 0]}), {"B", "zz"}) == (1, 1)  # pad-fix native changed (other length)
    bad = {"non-listed native changed": (fake_index({"A": [0, 0, 1]}), {"B"}),
           "no checked native changed": (fake_index({}), {"B"}),
           "no checked stem has a native row": (fake_index({}), {"zz"})}
    t = fake_index({"B": [1, 1, 1, 0]})
    t["he_flat"][3] ^= 1  # inside template row A@rw10#0
    bad["template row he changed"] = (t, {"B"})
    t = fake_index({"B": [1, 1, 1, 0]})
    t["align_flat"][0] += 1
    bad["template align changed"] = (t, {"A", "B"})
    t = fake_index({"B": [1, 1, 1, 0]})
    t["feat_flat"][-1] = 0.0  # NaN -> 0 in template row B@rw10#0
    bad["template feat NaN payload changed"] = (t, {"B"})
    t = fake_index({})
    t["ids"] = t["ids"][::-1]
    bad["row order"] = (t, {"B"})
    t = fake_index({"B": [1, 1, 1, 0]})
    t["pair_feature_mean"] = torch.zeros(2)
    bad["feature stats"] = (t, {"B"})
    t = fake_index({"B": [1, 0, 1]})
    t["runs_flat"][6] = torch.tensor([9, 9], dtype=torch.int16)  # first run of row 3 = native B
    bad["pad-fix native runs changed"] = (t, {"B"})
    for what, (new, allowed) in bad.items():
        with pytest.raises(AssertionError):
            tid.diff(old, new, allowed)
        print(f"  index diff raises as expected [{what}]")



def test_index_stamp(tmp_path):
    pack = str(tmp_path / "v2.pack")
    open(pack + ".idx.npz", "wb").write(b"idx-v2")
    new = fake_index({})
    with pytest.raises(AssertionError):
        tid.check_stamp(new, pack)  # no stamp
    new["native_source"] = json.dumps({"native_pack": os.path.abspath(pack),
                                       "native_idx_md5": __import__("hashlib").md5(b"idx-v2").hexdigest()})
    tid.check_stamp(new, pack)
    old = fake_index({})
    old["native_source"] = json.dumps({"native_pack": "v1"})
    new2 = fake_index({"B": [0, 0, 0]})
    new2["native_source"] = new["native_source"]
    assert tid.diff(old, new2, {"B"}) == (1, 1)  # the stamp itself is not a row/meta difference
    open(pack + ".idx.npz", "wb").write(b"idx-other")
    with pytest.raises(AssertionError):
        tid.check_stamp(new, pack)  # pack index changed since the build
    print("  stamp: missing / md5 mismatch raise; match passes")


def test_padfix_dir_limit(world):
    """A copy into <mid3>/<hash>/ whose <mid3> parent already holds 1025 entries is refused (root stays exempt)."""
    root = world
    orig, fix = str(root / "pdb_train" / "processed"), str(root / "fix" / "processed")
    dst = _processed_path_sharded(root / "fix" / "processed", "3cde_A", MAN)
    assert dst.parent.parent != root / "fix" / "processed"  # 3cde_A sits in a hash sub-bucket
    dst.parent.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1025):
        (dst.parent.parent / f"d{i}").mkdir()
    lst = write_tsv(root / "l.tsv", ["stem"], [["3cde_A"]])
    ca = argparse.Namespace(padfix_list=lst, status_tsv=str(root / "status.tsv"), tree=orig, fix_tree=fix, workers=1,
                            out_cache=str(root / "c.json"), out_stems=str(root / "s.txt"))
    with pytest.raises(AssertionError, match="1024"):
        tc.build_padfix(ca)
    assert not os.path.exists(ca.out_cache)
