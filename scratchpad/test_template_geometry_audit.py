"""Gate for template_geometry_audit.py on synthetic data: refs -> audit -> summary, run as subprocesses.

Chains (natives L=40, helix CA spacing 3.8 A):
  6onw_A  template with 2 stretched steps + residue 31 collapsed 0.6 A onto 30 (the collapsed step is not a break by the
          > 4.0 rule, but moving residue 31 stretches 31->32) -> 3 excess breaks; also the TEST ref (known-bad control)
  good_A  clean template; a 2nd row outside the TM range is excluded
  gap_A   native with a real 8 A gap that the template copies -> 0 excess, 1 native break; native residue 5 has no CA
  remap_A T2 entry whose file is named by its alias seq_partner (remap_B) -> resolved by the seq_partner rule
  bad_A   file with a DIFFERENT sequence -> guard_fail; an empty-runs row is dropped by refs
  good_A and gap_A are LIVE pool links into a regen tree (readlink); the others are T2 entries with no link (deleted)
  val_A   eligible but in the val split -> excluded; other_A not eligible -> excluded
"""

import csv
import io
import os
import subprocess
import sys
import tempfile
import zlib
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
L = 40


def helix_ca(n, gap_at=None):
    t = np.arange(n)
    ca = np.stack([2.3 * np.cos(t * 1.745), 2.3 * np.sin(t * 1.745), 1.5 * t], -1).astype(np.float32)  # 3.8 A steps
    if gap_at is not None:
        ca[gap_at + 1:] += np.array([0, 0, 8.0], np.float32)
    return ca


def atom37(ca):
    xyz = np.zeros((len(ca), 37, 3), np.float32)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = ca - [1.2, 0, 0], ca, ca + [1.2, 0, 0]
    m = np.zeros((len(ca), 37), bool)
    m[:, [0, 1, 2]] = True
    return xyz, m


with tempfile.TemporaryDirectory() as td:
    names = ["6onw_A", "good_A", "gap_A", "remap_A", "bad_A", "val_A", "other_A"]
    natives = {s: helix_ca(L, gap_at=19 if s == "gap_A" else None) for s in names}
    seq = {s: np.arange(L, dtype=np.int16) % 20 for s in names}
    tpl = {k: v.copy() for k, v in natives.items()}
    tpl["6onw_A"][10:] += [0, 0, 3.0]
    tpl["6onw_A"][25:] += [0, 0, 5.0]
    tpl["6onw_A"][31] = tpl["6onw_A"][30] + [0.6, 0, 0]
    blobs, stems, offs, lens, pos = [], [], [], [], 0
    for s in names:
        xyz, m = atom37(natives[s])
        if s == "gap_A":
            m[5] = False  # native residue 5 unresolved -> steps 4->5 and 5->6 unscored
        b = io.BytesIO()
        torch.save(SimpleNamespace(coords=torch.tensor(xyz), coord_mask=torch.tensor(m), residue_type=torch.tensor(seq[s].astype(np.int64))), b)
        blobs.append(b.getvalue()); stems.append(s); offs.append(pos); lens.append(len(blobs[-1])); pos += lens[-1]
    pack = f"{td}/t.pack"
    open(pack, "wb").write(b"".join(blobs))
    np.savez(pack + ".idx.npz", stems=np.array(stems), offsets=np.array(offs), lengths=np.array(lens))
    t2_orig, t2_new, regen, pool = f"{td}/gone/templates_band", f"{td}/copy/templates_band", f"{td}/regen", f"{td}/pool"
    live = {"good_A", "gap_A"}
    ctrl = []
    for s in names:
        key = "AUTHremap_B" if s == "remap_A" else "AUTH" + s  # file name = seq_partner for T2 entries
        xyz, m = atom37(tpl[s])
        aa = seq[s].copy()
        if s == "bad_A":
            aa[0] = 19 - aa[0]  # a different polymer
        base = regen if s in live else t2_new
        rel = f"shard{zlib.crc32(key.encode()) % 1000:04d}/{key}.npz"
        os.makedirs(os.path.dirname(f"{base}/{rel}"), exist_ok=True)
        np.savez(f"{base}/{rel}", atom_mask=m.astype(np.float32), coords=np.stack([xyz[m], xyz[m]]), rewind_steps=np.array([100, 200]), aatype=aa)
        prel = f"shard{zlib.crc32(('AUTH' + s).encode()) % 1000:04d}/AUTH{s}.npz"
        tgt = f"{regen}/{rel}" if s in live else f"{t2_orig}/{rel}"
        if s in live:
            os.makedirs(os.path.dirname(f"{pool}/{prel}"), exist_ok=True)
            os.symlink(tgt, f"{pool}/{prel}")
        if s in ("6onw_A", "good_A", "remap_A"):
            ctrl.append(f"{prel}\t{tgt}\n")
    open(f"{td}/ctrl.tsv", "w").writelines(ctrl)
    # band: every chain has rungs (slot 0, rw 100, tm 0.7) and (slot 1, rw 200, tm 0.8); good_A rung 1 tm 0.95
    btm = np.full((len(names), 64), 0.2, np.float32); bsl = np.full((len(names), 64), -1, np.int16); brw = np.zeros((len(names), 64), np.int16)
    for i, s in enumerate(names):
        bsl[i, 3], brw[i, 3], btm[i, 3] = 0, 100, 0.7
        bsl[i, 9], brw[i, 9], btm[i, 9] = 1, 200, 0.95 if s == "good_A" else 0.8
        btm[i, 50] = 0.1 + 0.01 * i  # out-of-band rung TMs differ per chain, as independent diffusion draws do
    np.savez(f"{td}/band.npz", chains=np.array(["AUTH" + s for s in names]), slot=bsl, rewind=brw, tm=btm)
    # source-tree band indexes, keyed by each tree's own FILE names (T2: the seq_partner name), rows copied into the pool band
    fname = lambda s: "AUTHremap_B" if s == "remap_A" else "AUTH" + s
    for path, members in ((f"{td}/regen_band.npz", [s for s in names if s in live]), (f"{td}/t2_band.npz", [s for s in names if s not in live])):
        ix = [names.index(s) for s in members]
        np.savez(path, chains=np.array([fname(s) for s in members]), slot=bsl[ix], rewind=brw[ix], tm=btm[ix])
    ids, row_tm, runs = [], [], []
    for s in names:
        ids += [s, f"{s}@rw100#0", f"{s}@rw200#1"]
        row_tm += [1.0, 0.7, 0.95 if s == "good_A" else 0.8]
        runs += [2, 2, 0 if s == "bad_A" else 2]  # bad_A's #1 row has empty runs
    roff = np.concatenate([[0], np.cumsum(runs)])
    torch.save({"ids": ids, "row_tm": torch.tensor(row_tm, dtype=torch.float16), "runs_offset": torch.tensor(roff)}, f"{td}/index.pt")
    open(f"{td}/elig.txt", "w").write("".join(s + "\n" for s in names if s != "other_A"))
    open(f"{td}/train.txt", "w").write("".join(s + "\n" for s in names if s != "val_A"))
    open(f"{td}/alias.tsv", "w").write("label_id\tauth_id\tseq_status\tseq_partner\tsource\taction\n" + "".join(
        f"{s}\tAUTH{s}\t{'remap' if s == 'remap_A' else 'self'}\t{'AUTHremap_B' if s == 'remap_A' else 'AUTH' + s}\tx\tuse_pool\n" for s in names))
    open(f"{td}/test.tsv", "w").write("stem\tref_id\ttm_template\n6onw_A\t6onw_A@rw200#1\t0.799\n")
    run = lambda *a: subprocess.run([sys.executable, f"{HERE}/template_geometry_audit.py", *a], capture_output=True, text=True, check=True).stdout
    fails = lambda *a: subprocess.run([sys.executable, f"{HERE}/template_geometry_audit.py", *a], capture_output=True, text=True).returncode != 0
    REFS = ["refs", "--index", f"{td}/index.pt", "--eligible", f"{td}/elig.txt", "--train_ids", f"{td}/train.txt", "--alias", f"{td}/alias.tsv",
            "--band", f"{td}/band.npz", "--pool", pool, "--t2_orig", t2_orig, "--band_sources", f"{regen}={td}/regen_band.npz,{t2_orig}={td}/t2_band.npz", "--control_targets", f"{td}/ctrl.tsv", "--test_tsv", f"{td}/test.tsv", "--pack", pack, "--out_refs", f"{td}/refs.tsv",
            "--out_native", f"{td}/nat.npz", "--tm_lo", "0.5", "--tm_hi", "0.9"]
    print(run(*REFS))
    refs = list(csv.DictReader(open(f"{td}/refs.tsv"), delimiter="\t"))
    got = sorted((r["set"], r["ref_id"]) for r in refs)
    exp = sorted([("test", "6onw_A@rw200#1")] + [("train", f"{s}@rw100#0") for s in ("6onw_A", "good_A", "gap_A", "remap_A", "bad_A")]
                 + [("train", f"{s}@rw200#1") for s in ("6onw_A", "gap_A", "remap_A")])
    assert got == exp, (got, exp)
    # a ref whose band rung disagrees (rewind) must stop refs
    b = dict(np.load(f"{td}/band.npz")); b["rewind"] = b["rewind"].copy(); b["rewind"][0, 9] = 999
    np.savez(f"{td}/band_bad.npz", **b)
    before = open(f"{td}/refs.tsv").read()
    assert fails(*[x if x != f"{td}/band.npz" else f"{td}/band_bad.npz" for x in REFS]), "refs accepted a band/ref rewind disagreement"
    assert open(f"{td}/refs.tsv").read() == before, "a failed refs run overwrote REFS.tsv"
    # a recorded target the rule does not reproduce must stop refs
    open(f"{td}/ctrl_bad.tsv", "w").write(open(f"{td}/ctrl.tsv").read().replace("AUTHremap_B.npz", "AUTHremap_A.npz"))
    assert fails(*[x if x != f"{td}/ctrl.tsv" else f"{td}/ctrl_bad.tsv" for x in REFS]), "refs accepted a control target mismatch"
    # the T2 band row under the partner name holding ANOTHER chain's fingerprint (a swapped homomer) must stop refs
    z = dict(np.load(f"{td}/t2_band.npz")); z["tm"] = z["tm"].copy()
    j = list(z["chains"]).index("AUTHremap_B"); z["tm"][j, 50] += 0.5
    np.savez(f"{td}/t2_band_bad.npz", **z)
    assert fails(*[x.replace(f"{td}/t2_band.npz", f"{td}/t2_band_bad.npz") for x in REFS]), "refs accepted a band fingerprint mismatch"
    A = ["audit", "--refs", f"{td}/refs.tsv", "--native", f"{td}/nat.npz", "--map", f"{t2_orig}={t2_new}"]
    print(run(*A, "--out", f"{td}/a.tsv"))
    res = {(r["set"], r["ref_id"]): r for r in csv.DictReader(open(f"{td}/a.tsv"), delimiter="\t")}
    b6 = res[("train", "6onw_A@rw200#1")]
    assert b6["status"] == "ok" and b6["n_excess_break"] == "3" and b6["n_native_break"] == "0" and abs(float(b6["min_ca"]) - 0.6) < 1e-3, b6
    g = res[("train", "gap_A@rw100#0")]
    assert g["n_excess_break"] == "0" and g["n_native_break"] == "1" and g["n_scored"] == str(L - 1 - 1 - 2), g
    assert res[("train", "good_A@rw100#0")]["n_excess_break"] == "0"
    assert res[("train", "remap_A@rw100#0")]["status"] == "ok" and res[("train", "remap_A@rw100#0")]["path"].startswith(t2_new) and "AUTHremap_B" in res[("train", "remap_A@rw100#0")]["path"]
    assert res[("train", "good_A@rw100#0")]["path"].startswith(regen)
    assert res[("train", "bad_A@rw100#0")]["status"] == "guard_fail", res[("train", "bad_A@rw100#0")]
    # split run covers every ref exactly once; summary over the parts is complete
    for p in range(3):
        run(*A, "--out", f"{td}/p{p}.tsv", "--part", str(p), "--n_parts", "3")
    parts = [(r["set"], r["ref_id"]) for p in range(3) for r in csv.DictReader(open(f"{td}/p{p}.tsv"), delimiter="\t")]
    assert sorted(parts) == sorted(res), (parts, list(res))
    # absent on host A + ok on host B -> ok wins; a missing part file -> summary fails
    run("audit", "--refs", f"{td}/refs.tsv", "--native", f"{td}/nat.npz", "--out", f"{td}/abs.tsv")  # no map: T2 paths absent here
    st = {r["ref_id"]: r["status"] for r in csv.DictReader(open(f"{td}/abs.tsv"), delimiter="\t")}
    assert st["remap_A@rw100#0"] == "absent" and st["good_A@rw100#0"] == "ok", st
    lines = open(f"{td}/abs.tsv").read().splitlines(True)
    open(f"{td}/abs_only.tsv", "w").writelines([lines[0]] + [l for l in lines[1:] if "\tabsent\t" in l])  # host A's leftovers
    out = run("summary", "--refs", f"{td}/refs.tsv", f"{td}/abs_only.tsv", f"{td}/p0.tsv", f"{td}/p1.tsv", f"{td}/p2.tsv")
    print(out)
    assert "guard_fail 1" in out and "control: 6onw_A flagged with 3" in out
    assert fails("summary", "--refs", f"{td}/refs.tsv", f"{td}/p0.tsv", f"{td}/p1.tsv"), "summary accepted a missing part"
    assert fails("summary", "--refs", f"{td}/refs.tsv", f"{td}/a.tsv", f"{td}/p0.tsv", f"{td}/p1.tsv", f"{td}/p2.tsv"), "summary accepted a ref audited ok twice"
print("ALL PASS")
