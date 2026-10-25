"""T8b overlay pack (REPAIR PLAN S7): the T8 pack with every status==changed stem's record replaced by its re-processed
.pt; every other record byte-identical to the T8 pack. Record = the raw .pt file bytes (scratchpad/build_pack.py), index
OUT.idx.npz = stems / offsets / lengths / md5, sorted by stem (= the T8 index order). The status table is the merged
reproc_status_final.tsv (plan 1057 MERGE RULE: subset rows override; status changed|unchanged only).

build:  guard pass over the changed stems (parallel; nothing written if any fails): the fixed .pt has contact_map_confind
        float16 of shape (L_new, L_new), coords.shape[0] == L_new, and the same keys / per-key tensor dtype as its T8 record;
        then a stream pass in index order (unchanged: T8 slice, md5 == T8 md5; changed: fixed file, md5 == guard md5 and
        != T8 md5) into OUT.partial, a full re-read md5 check, index + provenance, rename. Pack and table stem sets must be
        equal (counted, never reconciled). Safe to rerun: OUT.partial is rewritten from scratch; an existing OUT is refused.
  python t8b_overlay_pack.py build --old-pack T8.pack --status-tsv S.tsv --fix-processed FIX/processed \
      --orig-data-dir pdb_train --out T8B.pack --workers N
verify: index stems == T8 stems == table stems; unchanged md5 == T8 md5, changed != T8 md5; md5(every source .pt: fixed
        tree for changed, original tree for unchanged) == new md5; md5(every pack slice) == new md5; contiguous blob.
  python t8b_overlay_pack.py verify --old-pack T8.pack --status-tsv S.tsv --fix-processed FIX/processed \
      --orig-data-dir pdb_train --out T8B.pack --workers N
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

STATUSES = ("changed", "unchanged")
_FD = None  # per-worker read-only fd of the T8 pack


def load_status(path):
    """stem -> (status, category, L_new) from the merged status table; statuses outside STATUSES are fatal."""
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            assert r["stem"] not in rows, f"duplicate stem {r['stem']} in {path}"
            rows[r["stem"]] = (r["status"], r["category"], int(r["L_new"]))
    c = collections.Counter(v[0] for v in rows.values())
    assert set(c) <= set(STATUSES), f"{path}: statuses other than {STATUSES}: {dict(c)}"
    return rows


def load_index(pack):
    z = np.load(pack + ".idx.npz", allow_pickle=True)
    stems = [str(s) for s in z["stems"]]
    assert len(set(stems)) == len(stems), f"{pack}: duplicate stems in the index"
    return stems, z["offsets"].astype(np.int64), z["lengths"].astype(np.int64), [str(m) for m in z["md5"]]


def source_path(stem, status, fix_processed, orig_processed, man):
    return _processed_path_sharded(fix_processed if status == "changed" else orig_processed, stem, man)


def compare_sets(pack_stems, table):
    only_pack, only_table = sorted(set(pack_stems) - set(table)), sorted(set(table) - set(pack_stems))
    print(f"[sets] pack {len(pack_stems)} stems, table {len(table)} rows: in pack not table {len(only_pack)} "
          f"{only_pack[:5]}, in table not pack {len(only_table)} {only_table[:5]}", flush=True)
    assert not only_pack and not only_table, "pack and status table stem sets differ (no rule for them: ask the user)"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _open_pack(pack):
    global _FD
    _FD = os.open(pack, os.O_RDONLY)


def _read_slice(off, n):
    b = os.pread(_FD, n, off)
    assert len(b) == n, f"short read at {off}: {len(b)} of {n}"
    return b


def key_dtypes(g):
    return {k: (str(v.dtype) if torch.is_tensor(v) else type(v).__name__) for k, v in g.items()}


def guard_one(args):
    """Problems of one changed stem (empty = pass) + md5 / size of the fixed file bytes."""
    stem, path, L_new, off, n = args
    b = pathlib.Path(path).read_bytes()
    g = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    old = torch.load(io.BytesIO(_read_slice(off, n)), map_location="cpu", weights_only=False)
    probs = []
    m = getattr(g, "contact_map_confind", None)
    if m is None:
        probs.append("no contact_map_confind")
    else:
        if m.dtype != torch.float16:
            probs.append(f"contact_map_confind dtype {m.dtype}")
        if tuple(m.shape) != (L_new, L_new):
            probs.append(f"contact_map_confind shape {tuple(m.shape)} != ({L_new}, {L_new})")
    if int(g.coords.shape[0]) != L_new:
        probs.append(f"coords L {int(g.coords.shape[0])} != L_new {L_new}")
    kn, ko = key_dtypes(g), key_dtypes(old)
    if set(kn) != set(ko):
        probs.append(f"keys +{sorted(set(kn) - set(ko))} -{sorted(set(ko) - set(kn))}")
    probs += [f"{k} dtype {ko[k]} -> {kn[k]}" for k in sorted(set(kn) & set(ko)) if kn[k] != ko[k]]
    return stem, hashlib.md5(b).hexdigest(), len(b), probs


def md5_slice(args):
    off, n = args
    return hashlib.md5(_read_slice(off, n)).hexdigest()


def md5_path(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def setup(a):
    stems, offs, lens, md5s = load_index(a.old_pack)
    assert os.path.getsize(a.old_pack) == int((offs + lens).max()), f"{a.old_pack}: size != index end"
    table = load_status(a.status_tsv)
    compare_sets(stems, table)
    man = json.load(open(pathlib.Path(a.orig_data_dir) / "shard_manifest.json"))
    orig_processed = pathlib.Path(a.orig_data_dir) / "processed"
    paths = [source_path(s, table[s][0], pathlib.Path(a.fix_processed), orig_processed, man) for s in stems]
    c = collections.Counter(table[s][0] for s in stems)
    print(f"[table] {dict(c)}", flush=True)
    return stems, offs, lens, md5s, table, paths


def build(a):
    assert not os.path.exists(a.out), f"{a.out} exists -- one fresh overlay per build"
    t0 = time.time()
    stems, offs, lens, md5s, table, paths = setup(a)
    chg = [i for i, s in enumerate(stems) if table[s][0] == "changed"]
    jobs = [(stems[i], str(paths[i]), table[stems[i]][2], int(offs[i]), int(lens[i])) for i in chg]
    probs, new_md5 = collections.Counter(), {}
    examples = collections.defaultdict(list)
    with ProcessPoolExecutor(a.workers, initializer=_open_pack, initargs=(a.old_pack,)) as ex:
        for k, (stem, m, n, p) in enumerate(ex.map(guard_one, jobs, chunksize=64)):
            new_md5[stem] = (m, n)
            for x in p:
                key = re.sub(r"\d+", "N", x)
                probs[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append(f"{stem}: {x}")
            if (k + 1) % 10000 == 0:
                print(f"[guard] {k + 1}/{len(jobs)}", flush=True)
    print(f"[guard] {len(jobs)} changed stems checked, {sum(probs.values())} problems {dict(probs)}", flush=True)
    for key, ex_ in examples.items():
        print(f"  {key}: {ex_}")
    assert not probs, "guard failed: nothing written"
    _open_pack(a.old_pack)
    out_offs, out_lens, out_md5, off = [], [], [], 0
    n_t8_bad, n_same = 0, 0
    with open(a.out + ".partial", "wb") as fo:
        for i, s in enumerate(stems):
            if table[s][0] == "changed":
                b = paths[i].read_bytes()
                m = hashlib.md5(b).hexdigest()
                assert (m, len(b)) == new_md5[s], f"{s}: {paths[i]} changed between the guard and the write"
                n_same += m == md5s[i]
            else:
                b = _read_slice(int(offs[i]), int(lens[i]))
                m = hashlib.md5(b).hexdigest()
                n_t8_bad += m != md5s[i]
            fo.write(b)
            out_offs.append(off); out_lens.append(len(b)); out_md5.append(m)
            off += len(b)
            if (i + 1) % 20000 == 0:
                print(f"[write] {i + 1}/{len(stems)} {off / 1e9:.2f} GB", flush=True)
        fo.flush()
        os.fsync(fo.fileno())
    assert n_t8_bad == 0, f"{n_t8_bad} unchanged T8 records differ from the T8 index md5"
    assert n_same == 0, f"{n_same} changed stems have a fixed .pt byte-identical to the T8 record"
    with ProcessPoolExecutor(a.workers, initializer=_open_pack, initargs=(a.out + ".partial",)) as ex:
        bad = sum(m != r for m, r in zip(ex.map(md5_slice, zip(out_offs, out_lens), chunksize=256), out_md5))
    assert bad == 0, f"{bad} entries differ from their source after the write"
    np.savez(a.out + ".idx.partial.npz", stems=np.array(stems, dtype=object), offsets=np.array(out_offs, np.int64),
             lengths=np.array(out_lens, np.int64), md5=np.array(out_md5, dtype=object))
    c = collections.Counter(table[s][0] for s in stems)
    prov = {"old_pack": os.path.abspath(a.old_pack), "old_pack_bytes": os.path.getsize(a.old_pack),
            "old_idx_sha256": sha256_file(a.old_pack + ".idx.npz"),
            "status_tsv": os.path.abspath(a.status_tsv), "status_tsv_sha256": sha256_file(a.status_tsv),
            "fix_processed": os.path.abspath(a.fix_processed), "orig_data_dir": os.path.abspath(a.orig_data_dir),
            "entries": len(stems), "changed_replaced": c["changed"], "unchanged_copied": c["unchanged"],
            "bytes": off, "rule": "status==changed -> fixed .pt bytes; status==unchanged -> T8 record bytes",
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(prov, open(a.out + ".provenance.json", "w"), indent=1)
    os.replace(a.out + ".idx.partial.npz", a.out + ".idx.npz")
    os.replace(a.out + ".partial", a.out)
    print(f"[done] {len(stems)} entries ({c['changed']} replaced, {c['unchanged']} copied), {off / 1e9:.2f} GB, "
          f"all re-verified by md5, {time.time() - t0:.0f} s -> {a.out}", flush=True)


def verify(a):
    t0 = time.time()
    stems, offs, lens, md5s, table, paths = setup(a)
    n_stems, n_offs, n_lens, n_md5 = load_index(a.out)
    assert n_stems == stems, "overlay index stems != T8 index stems (same order expected)"
    order = np.argsort(n_offs)
    assert n_offs[order[0]] == 0 and np.array_equal(n_offs[order][1:], (n_offs + n_lens)[order][:-1]), "blob not contiguous"
    assert os.path.getsize(a.out) == int(n_lens.sum()), f"{a.out}: size != sum of lengths"
    chg = np.array([table[s][0] == "changed" for s in stems])
    same = np.array([x == y for x, y in zip(n_md5, md5s)])
    print(f"[md5 vs T8] unchanged equal {int((same & ~chg).sum())}/{int((~chg).sum())}, "
          f"changed differ {int((~same & chg).sum())}/{int(chg.sum())}", flush=True)
    assert (same == ~chg).all(), "an unchanged record differs from T8 or a changed record equals it"
    with ProcessPoolExecutor(a.workers, initializer=_open_pack, initargs=(a.out,)) as ex:
        bad_slice = [s for s, m, r in zip(stems, ex.map(md5_slice, zip(n_offs, n_lens), chunksize=256), n_md5) if m != r]
    with ProcessPoolExecutor(a.workers) as ex:
        src = list(ex.map(md5_path, [str(p) for p in paths], chunksize=256))
    bad_src = collections.Counter(table[s][0] for s, m, r in zip(stems, src, n_md5) if m != r)
    print(f"[md5] pack slices != index {len(bad_slice)} {bad_slice[:5]}; source .pt != index {dict(bad_src)} "
          f"(of {len(stems)}), {time.time() - t0:.0f} s", flush=True)
    assert not bad_slice and not bad_src, "verify FAILED"
    print(f"[verify] PASS {len(stems)} entries = table rows; {int(chg.sum())} replaced, {int((~chg).sum())} identical to T8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--old-pack", required=True)
    ap.add_argument("--status-tsv", required=True)
    ap.add_argument("--fix-processed", required=True)
    ap.add_argument("--orig-data-dir", required=True, help="pdb_train: shard_manifest.json + processed/ (unchanged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, required=True)
    a = ap.parse_args()
    build(a) if a.mode == "build" else verify(a)
