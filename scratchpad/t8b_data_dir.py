"""T8b data_dir (REPAIR PLAN S8): ${DATA_PATH}/pdb_train_t8b = per status-table stem a symlink
processed/<bucket>/<stem>.pt -> the fixed tree (status changed) or the original pdb_train tree (unchanged), bucket from
pdb_train's shard_manifest.json (copied), the dataset CSV copied with processed_length = L_new for changed stems (L_new =
coord_mask.any(-1).sum(), the processed_length definition), the cluster tsv the frozen split reads (copied), and an empty
raw/ (PDBLightningDataModule.__init__ mkdirs raw/ and processed/). CSV / tsv names come from the config's dataselector /
datasplitter via the datamodule's own _get_file_identifier / setup_clustering_file_paths.

build:  fails if OUT exists; builds OUT.partial (an old OUT.partial is removed) then renames.
  python t8b_data_dir.py build --config DATASET.yaml --orig-data-dir pdb_train --fix-processed FIX/processed \
      --status-tsv S.tsv --out pdb_train_t8b
verify: links / targets / CSV / copies; for a seeded sample of changed and unchanged stems the graph read through
        PDBDataset(pack_path=PACK) equals PDBDataset on this dir (disk, all keys) and the raw pack record equals torch.load
        of the source .pt; PDBLightningDataModule(config dataselector/datasplitter, overwrite / regenerate_missing from the
        config, both must be False).prepare_data() + setup('fit') leave every path under OUT unchanged (lstat + sha256).
  python t8b_data_dir.py verify --config T8B.yaml ... --out pdb_train_t8b --pack T8B.pack --n-sample N --seed S
"""
import argparse
import collections
import csv
import hashlib
import inspect
import io
import json
import os
import pathlib
import random
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from proteinfoundation.datasets.pdb_data import (PDBDataSelector, PDBDataset, PDBDataSplitter, PDBLightningDataModule,
                                                 _processed_path_sharded)
from proteinfoundation.utils.cluster_utils import setup_clustering_file_paths

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t8b_overlay_pack import load_index, load_status, sha256_file  # noqa: E402

PROV = "T8B_DATA_DIR_PROVENANCE.json"


def names(config):
    """(csv name, cluster tsv name) exactly as the datamodule derives them from the config."""
    cfg = OmegaConf.load(config).datamodule
    sel = OmegaConf.to_container(cfg.dataselector, resolve=False)
    defaults = {k: p.default for k, p in inspect.signature(PDBDataSelector.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}
    ds = SimpleNamespace(**{**defaults, **sel})
    fid = PDBLightningDataModule._get_file_identifier(None, ds)
    _, _, tsv = setup_clustering_file_paths("", fid, cfg.datasplitter.split_sequence_similarity)
    return f"{fid}.csv", tsv.name


def edit_csv(src, dst, table):
    """Copy src -> dst with processed_length = L_new for changed stems; returns (#rows, #values that changed)."""
    text = open(src, newline="").read()
    rows = list(csv.reader(io.StringIO(text)))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    assert buf.getvalue() == text, f"{src}: csv round-trip is not byte-identical -- refusing to rewrite it"
    head = rows[0]
    i_id, i_pl = head.index("id"), head.index("processed_length")
    seen = collections.Counter(r[i_id] for r in rows[1:])
    miss = [s for s in table if seen[s] != 1]
    assert not miss, f"{len(miss)} table stems are not exactly once in {src}, e.g. {miss[:5]}"
    n_new = 0
    for r in rows[1:]:
        t = table.get(r[i_id])
        if t is not None and t[0] == "changed":
            n_new += r[i_pl] == "" or int(float(r[i_pl])) != t[2]
            r[i_pl] = str(t[2])
    with open(dst, "w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return len(rows) - 1, n_new


def check_csv(src, dst, table):
    """pandas view (what setup reads): every column equal except processed_length of changed stems; processed_length ==
    L_new for every table stem."""
    a, b = pd.read_csv(src), pd.read_csv(dst)
    assert list(a.columns) == list(b.columns) and len(a) == len(b), "csv shape changed"
    rest = [c for c in a.columns if c != "processed_length"]
    assert a[rest].equals(b[rest]), "a column other than processed_length changed"
    chg = b["id"].map(lambda s: s in table and table[s][0] == "changed")
    pa, pb = a["processed_length"], b["processed_length"]
    assert ((pa == pb) | (pa.isna() & pb.isna()))[~chg].all(), "processed_length changed for a non-changed row"
    want = b["id"].map(lambda s: table[s][2] if s in table else np.nan)
    in_t = b["id"].isin(table.keys())
    bad = b.loc[in_t & (pb != want), "id"].tolist()
    print(f"[csv] {len(b)} rows ({int(in_t.sum())} table stems, {len(b) - int(in_t.sum())} not in the table); "
          f"processed_length != L_new for {len(bad)} table stems {bad[:5]}", flush=True)
    assert not bad, "processed_length disagrees with L_new"


def targets(table, orig_data_dir, fix_processed, man):
    orig = pathlib.Path(orig_data_dir) / "processed"
    return {s: _processed_path_sharded(pathlib.Path(fix_processed) if v[0] == "changed" else orig, s, man)
            for s, v in table.items()}


def build(a):
    out = pathlib.Path(a.out)
    assert not out.exists(), f"{out} exists -- one fresh data_dir per build"
    t0 = time.time()
    part = pathlib.Path(str(out) + ".partial")
    if part.exists():
        print(f"[build] removing a previous {part}", flush=True)
        shutil.rmtree(part)
    orig = pathlib.Path(a.orig_data_dir)
    table = load_status(a.status_tsv)
    csv_name, tsv_name = names(a.config)
    man = json.load(open(orig / "shard_manifest.json"))
    tg = targets(table, orig, a.fix_processed, man)
    missing = [str(p) for p in tg.values() if not p.exists()]
    assert not missing, f"{len(missing)} link targets do not exist, e.g. {missing[:5]}"
    (part / "raw").mkdir(parents=True)
    for f in ("shard_manifest.json", tsv_name):
        shutil.copyfile(orig / f, part / f)
    n_rows, n_new = edit_csv(orig / csv_name, part / csv_name, table)
    check_csv(orig / csv_name, part / csv_name, table)
    per_dir = collections.Counter()
    for s, t in tg.items():
        link = _processed_path_sharded(part / "processed", s, man)
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(t, link)
        per_dir[link.parent] += 1
    assert max(per_dir.values()) <= 1024, f"a bucket holds {max(per_dir.values())} links (> 1024)"
    c = collections.Counter(v[0] for v in table.values())
    prov = {"status_tsv": os.path.abspath(a.status_tsv), "status_tsv_sha256": sha256_file(a.status_tsv),
            "orig_data_dir": str(orig.resolve()), "fix_processed": os.path.abspath(a.fix_processed),
            "config": os.path.abspath(a.config), "csv": csv_name, "csv_src_sha256": sha256_file(orig / csv_name),
            "cluster_tsv": tsv_name, "links": len(tg), "links_changed_to_fix": c["changed"],
            "links_unchanged_to_orig": c["unchanged"], "csv_rows": n_rows, "csv_processed_length_values_changed": n_new,
            "rule": "status==changed -> fix tree; status==unchanged -> pdb_train; processed_length(changed) = L_new",
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(prov, open(part / PROV, "w"), indent=1)
    part.rename(out)
    print(f"[done] {len(tg)} links ({c['changed']} -> fix, {c['unchanged']} -> pdb_train) in {len(per_dir)} buckets "
          f"(max {max(per_dir.values())}); csv {n_rows} rows, processed_length changed for {n_new}; {time.time() - t0:.0f} s"
          f" -> {out}", flush=True)


def same_graph(a, b):
    """Keys whose values differ (tensors: dtype, shape, values with NaN == NaN)."""
    ka, kb = set(a.keys()), set(b.keys())
    diff = sorted(ka ^ kb)
    for k in sorted(ka & kb):
        x, y = a[k], b[k]
        if torch.is_tensor(x) or torch.is_tensor(y):
            ok = (torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape
                  and bool(((x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)).all()))
        elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            ok = isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
        else:
            ok = x == y
        if not ok:
            diff.append(k)
    return diff


def snapshot(root):
    """Every path under root: lstat (mode, size, mtime_ns, inode) + link target; sha256 of regular files."""
    snap = {}
    for d, dirs, files in os.walk(root):
        for n in dirs + files:
            p = os.path.join(d, n)
            st = os.lstat(p)
            snap[p] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ino,
                       os.readlink(p) if os.path.islink(p) else None,
                       sha256_file(p) if os.path.isfile(p) and not os.path.islink(p) else None)
    st = os.lstat(root)
    snap[root] = (st.st_mode, st.st_mtime_ns, st.st_ino)
    return snap


def verify(a):
    t0 = time.time()
    out, orig = pathlib.Path(a.out), pathlib.Path(a.orig_data_dir)
    table = load_status(a.status_tsv)
    csv_name, tsv_name = names(a.config)
    man = json.load(open(orig / "shard_manifest.json"))
    tg = targets(table, orig, a.fix_processed, man)
    for f in ("shard_manifest.json", tsv_name):
        assert sha256_file(out / f) == sha256_file(orig / f), f"{f} differs from pdb_train"
    check_csv(orig / csv_name, out / csv_name, table)
    links = [os.path.join(d, f) for d, _, fs in os.walk(out / "processed") for f in fs]
    want = {str(_processed_path_sharded(out / "processed", s, man)): t for s, t in tg.items()}
    bad = [l for l in links if l not in want or not os.path.islink(l) or os.readlink(l) != str(want[l])]
    print(f"[links] {len(links)} under processed/ for {len(want)} table stems; wrong / extra {len(bad)} {bad[:5]}",
          flush=True)
    assert len(links) == len(want) and not bad, "processed/ links do not match the status table"
    dangling = [l for l in links if not os.path.exists(l)]
    print(f"[links] dangling {len(dangling)} of {len(links)} {dangling[:5]}", flush=True)
    assert not dangling, "links whose target no longer exists"
    fixed_all = [p for p in pathlib.Path(a.fix_processed).rglob("*.pt")]
    stale = collections.Counter(table[p.stem][0] if p.stem in table else "not_in_table" for p in fixed_all)
    print(f"[fix tree] {len(fixed_all)} .pt files by table status {dict(stale)} (only 'changed' ones are linked)", flush=True)

    rng = random.Random(a.seed)
    sample, n_by = [], {}
    for st in ("changed", "unchanged"):
        pool = sorted(s for s, v in table.items() if v[0] == st)
        n_by[st] = min(a.n_sample, len(pool))
        sample += rng.sample(pool, n_by[st])
    codes, chains = [s.split("_", 1)[0] for s in sample], [s.split("_", 1)[1] for s in sample]
    packed = PDBDataset(pdb_codes=codes, chains=chains, data_dir=str(out), file_names=sample, num_workers=0,
                        pack_path=a.pack)
    disk = PDBDataset(pdb_codes=codes, chains=chains, data_dir=str(out), file_names=sample, num_workers=0)
    stems, offs, lens, _ = load_index(a.pack)
    where = {s: (int(o), int(n)) for s, o, n in zip(stems, offs, lens)}
    fails = collections.Counter()
    with open(a.pack, "rb") as fh:
        for i, s in enumerate(sample):
            g_pack, g_disk = packed[i], disk[i]
            assert g_pack.id == g_disk.id, s
            d1 = same_graph(g_pack, g_disk)
            fh.seek(where[s][0])
            raw = torch.load(io.BytesIO(fh.read(where[s][1])), map_location="cpu", weights_only=False)
            d2 = same_graph(raw, torch.load(tg[s], map_location="cpu", weights_only=False))
            real = os.path.realpath(_processed_path_sharded(out / "processed", s, man)) == os.path.realpath(tg[s])
            if d1 or d2 or not real:
                fails[table[s][0]] += 1
                print(f"  FAIL {s} ({table[s][0]}): reader diff {d1}, raw diff {d2}, link resolves right {real}")
    n_keys = len(list(packed[0].keys()))
    print(f"[reader] {len(sample)} stems (seed {a.seed}; {n_by}), {n_keys} reader keys on the first; "
          f"failures {dict(fails)}", flush=True)
    assert len(sample) == sum(n_by.values()) > 0 and not fails, "reader equality FAILED"

    cfg = OmegaConf.load(a.config).datamodule
    assert cfg.regenerate_missing is False and cfg.overwrite is False, \
        f"{a.config}: regenerate_missing={cfg.regenerate_missing} overwrite={cfg.overwrite} (both must be False)"
    for r in (cfg.data_dir, cfg.dataselector.data_dir, cfg.datasplitter.data_dir):
        assert os.path.realpath(r) == os.path.realpath(out), f"{a.config}: data_dir {r} is not {out}"
    sel = OmegaConf.to_container(cfg.dataselector, resolve=True)
    spl = OmegaConf.to_container(cfg.datasplitter, resolve=True)
    for d in (sel, spl):
        d.pop("_target_")
    before = snapshot(str(out))
    dm = PDBLightningDataModule(data_dir=cfg.data_dir, dataselector=PDBDataSelector(**sel),
                                datasplitter=PDBDataSplitter(**spl), pack_path=a.pack, overwrite=cfg.overwrite,
                                regenerate_missing=cfg.regenerate_missing, format=cfg.format,
                                sampling_mode=cfg.sampling_mode, batch_size=1, num_workers=0)
    dm.prepare_data()
    dm.setup("fit")  # = training; a second setup() call trips the pre-existing `if not self.df_data` (DataFrame truth)
    after = snapshot(str(out))
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    print(f"[no-write] prepare_data + setup(fit): train {len(dm.train_ds)} / val {len(dm.val_ds)} chains; {len(before)} paths snapshotted, {len(changed)} changed {changed[:5]}", flush=True)
    assert len(before) > len(want) and not changed, "the datamodule wrote under the T8b data_dir"
    print(f"[verify] PASS {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--config", required=True, help="dataset yaml (verify: the T8b one, data_dir must resolve to --out)")
    ap.add_argument("--orig-data-dir", required=True)
    ap.add_argument("--fix-processed", required=True)
    ap.add_argument("--status-tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", help="verify: the T8b overlay pack")
    ap.add_argument("--n-sample", type=int, help="verify: stems per status class")
    ap.add_argument("--seed", type=int, help="verify: random.Random seed of the sample")
    a = ap.parse_args()
    if a.mode == "verify":
        assert a.pack and a.n_sample and a.seed is not None, "verify needs --pack, --n-sample, --seed"
    build(a) if a.mode == "build" else verify(a)
