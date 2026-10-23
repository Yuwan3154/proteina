#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Per-MEMBER topology index of the CAT-pair regime (reads/writes a datasets.cat_unit_store layout).

For every ENTRY: DSSP of the WHOLE chain (pydssp, proline donor mask: the T8 recipe) and the whole-chain Frame2ConFind
map (>= 0.01, bit-packed into the store). For every MEMBER (entry, CAT, domain): the domain-restricted element view,
i.e. the whole-chain DSSP and contact map cut to the domain residues, then the T8 element rows (runs, element contact,
structural features) -- the same `topology_row` the T8 index uses, so assemble_reference serves both. Per member the
domain's CA coordinates and residue->element map are stored so the pair-specific USalign alignment target can be made
online.

Definitions that must equal the T8 index (the model was trained on them): element types, loop axis, min_len, DSSP
definition, contact definition, and the standardisation constants (pair-feature mean/std, element-length mean/std).
They are copied from a metadata json extracted from the T8 index (`extract-meta`), never re-estimated, and the build
refuses a mismatching element-type / loop-axis setting.

  extract-meta --index topology_index_confind_synth_t8.pt --out t8_meta.json
  build  --store S --part i --n-parts n --out-dir O --f2c-checkpoint CKPT --meta t8_meta.json
  merge  --store S --parts-dir O --n-parts n --out cat_pair_index.pt --meta t8_meta.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proteinfoundation.datasets.cat_unit_store import UnitStore, seg_positions  # noqa: E402
from proteinfoundation.datasets.sse_topology import (  # noqa: E402
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    STRUCTURAL_PAIR_FEATURES,
)
from proteinfoundation.utils.dssp_utils import compute_dssp_target  # noqa: E402
from proteinfoundation.utils.precompute_synthetic_topology_index import (  # noqa: E402
    CA_IDX,
    CONFIND_THRESHOLD,
    PRO_IDX,
    _f2c_inputs,
    topology_row,
)

META_KEYS = ("pair_feature_mean", "pair_feature_std", "elem_feature_mean", "elem_feature_std", "element_types",
             "loop_axis", "min_len", "dssp_def", "contact_def", "pair_feature_names", "structural_feature_names")


def cmd_extract_meta(args):
    idx = torch.load(args.index, map_location="cpu", weights_only=False, mmap=True)
    meta = {}
    for k in META_KEYS:
        v = idx[k]
        meta[k] = v.tolist() if isinstance(v, torch.Tensor) else v
    assert list(meta["pair_feature_names"]) == list(PAIR_FEATURE_NAMES), "T8 index pair features differ from this code"
    assert list(meta["structural_feature_names"]) == list(STRUCTURAL_PAIR_FEATURES)
    json.dump(meta, open(args.out, "w"), indent=1)
    print("meta ->", args.out, {k: meta[k] for k in ("element_types", "loop_axis", "min_len", "dssp_def", "contact_def", "elem_feature_mean", "elem_feature_std")})


def load_entry_tensors(store, entry_id):
    e = store.entry(entry_id)
    mask = torch.from_numpy(np.array(e["mask"]))
    full = torch.from_numpy(np.array(e["coords"]))
    return e, full, mask


def entry_rows(store, eid, entry_members, cm, element_types, loop_axis, min_len):
    """Member rows of one entry from its whole-chain contact map `cm` ([L, L] float 0/1): whole-chain DSSP, then each
    member's domain view -> topology_row. Returns (rows, skip lines)."""
    e, f, m = load_entry_tensors(store, eid)
    aatype = torch.from_numpy(np.array(e["aatype"])).long()
    dssp = compute_dssp_target(f[None], m[:, CA_IDX][None], coord_mask=m[None], coord_layout="atom37",
                               donor_mask=(aatype != PRO_IDX)[None])
    if dssp is None:
        return [], [f"{eid}\tentry\tdssp_none"]
    dssp = dssp[0]
    rows, skips = [], []
    for mem in entry_members.itertuples():
        pos = torch.from_numpy(seg_positions(mem.seg)).long()
        got = topology_row(cm[pos][:, pos], f[pos], m[pos], dssp[pos], min_len, element_types=element_types, loop_axis=loop_axis)
        if got is None:
            skips.append(f"{mem.member_id}\tmember\tdssp_all_ignore")
            continue
        assert bool(m[pos, CA_IDX].all()), f"{mem.member_id}: a domain residue without CA (the store keeps only N/CA/C residues)"
        rows.append(dict(member_id=mem.member_id, row=got[:5], elem=got[5].astype(np.int16), ca=f[pos, CA_IDX].numpy().astype(np.float32)))
    return rows, skips


def cmd_build(args):
    meta = json.load(open(args.meta))
    element_types = tuple(int(t) for t in meta["element_types"])
    loop_axis, min_len = meta["loop_axis"], int(meta["min_len"])
    store = UnitStore(args.store)
    entries = store.entries.iloc[args.part::args.n_parts]
    sys.path.insert(0, os.path.expanduser(args.f2c_parent))
    from Frame2ConFind.inference.api import Frame2ConFindPredictor
    predictor = Frame2ConFindPredictor(checkpoint=os.path.expanduser(args.f2c_checkpoint), amp_dtype=args.amp_dtype, compile_model=False)
    print(f"f2c {args.f2c_checkpoint} on {predictor.device}; part {args.part}: {len(entries)} entries", flush=True)
    by_entry = {e: g for e, g in store.members.groupby("entry_id")}
    order = entries.sort_values("n_res", ascending=False)
    ids = order.entry_id.tolist()
    rows, cbits, skips = [], {}, []
    t0 = time.perf_counter()
    for i in range(0, len(ids), args.batch_size):
        chunk = ids[i:i + args.batch_size]
        tensors = [load_entry_tensors(store, c) for c in chunk]
        Lmax = max(int(t[2].shape[0]) for t in tensors)
        full = torch.zeros(len(chunk), Lmax, 37, 3)
        amask = torch.zeros(len(chunk), Lmax, 37, dtype=torch.bool)
        for j, (_, f, m) in enumerate(tensors):
            L = int(m.shape[0])
            full[j, :L], amask[j, :L] = f, m
        xs, ms = zip(*(_f2c_inputs(full[j:j + 1], amask[j]) for j in range(len(chunk))))
        probs = predictor.predict_batch(torch.cat(xs), torch.stack(ms)).float().cpu()
        for j, (eid, (e, f, m)) in enumerate(zip(chunk, tensors)):
            L = int(m.shape[0])
            cm = (probs[j, :L, :L] >= CONFIND_THRESHOLD).float()
            cbits[eid] = np.packbits(cm.numpy().astype(bool).reshape(-1))
            got, sk = entry_rows(store, eid, by_entry[eid], cm, element_types, loop_axis, min_len)
            rows += got
            skips += sk
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(ids)} entries, {len(rows)} member rows, {time.perf_counter() - t0:.0f} s", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = os.path.join(args.out_dir, f"part_{args.part:04d}.tmp")
    torch.save(dict(rows=rows, cbits=cbits, skips=skips), tmp)
    os.replace(tmp, os.path.join(args.out_dir, f"part_{args.part:04d}.pt"))
    print(f"part {args.part}: {len(rows)} member rows, {len(cbits)} contact maps, {len(skips)} skips", flush=True)
    print("CATPAIR_PART_DONE", flush=True)


def cmd_merge(args):
    meta = json.load(open(args.meta))
    store = UnitStore(args.store)
    rows, cbits, skips = {}, {}, []
    for part in range(args.n_parts):
        p = os.path.join(args.parts_dir, f"part_{part:04d}.pt")
        assert os.path.exists(p), f"missing {p} -- rebuild that part before merging"
        d = torch.load(p, map_location="cpu", weights_only=False)
        for r in d["rows"]:
            assert r["member_id"] not in rows, f"member {r['member_id']} in two parts"
            rows[r["member_id"]] = r
        cbits.update(d["cbits"])
        skips += d["skips"]
    merge_rows(store, rows, cbits, skips, meta, args.out)


def merge_rows(store, rows, cbits, skips, meta, out_path):
    members = store.members
    missing = members[~members.member_id.isin(rows)]
    n_entry_no_cbits = len(set(store.entries.entry_id) - set(cbits))
    assert n_entry_no_cbits == 0, f"{n_entry_no_cbits} entries have no contact map (a part is missing)"
    print(f"member rows {len(rows)} of {len(members)}; {len(missing)} members without a row (see skip log): {len(skips)} skips", flush=True)
    members = members[members.member_id.isin(rows)].reset_index(drop=True)
    ids = members.member_id.tolist()

    runs_flat, runs_offset = [], [0]
    he_flat, he_offset, he_size, feat_flat, feat_offset = [], [0], [], [], [0]
    ca_flat, elem_flat, ca_offset = [], [], [0]
    for mid in ids:
        r = rows[mid]
        runs, ref_b, t_he, feat_b, _ = r["row"]
        runs_flat.extend(runs)
        runs_offset.append(runs_offset[-1] + len(runs))
        he_size.append(t_he)
        ref = torch.frombuffer(bytearray(ref_b), dtype=torch.uint8) if ref_b else torch.zeros(0, dtype=torch.uint8)
        he_flat.append(ref)
        he_offset.append(he_offset[-1] + ref.numel())
        f = torch.frombuffer(bytearray(feat_b), dtype=torch.float16) if feat_b else torch.zeros(0, dtype=torch.float16)
        feat_flat.append(f)
        feat_offset.append(feat_offset[-1] + f.numel())
        ca_flat.append(torch.from_numpy(r["ca"]))
        elem_flat.append(torch.from_numpy(r["elem"]))
        ca_offset.append(ca_offset[-1] + r["ca"].shape[0])

    entry_codes = {e: i for i, e in enumerate(store.entries.entry_id)}
    src_codes = {u: i for i, u in enumerate(store.entries.src_unit.unique())}
    entry_src = dict(zip(store.entries.entry_id, store.entries.src_unit))
    pool_key = members["cat"] + "|" + members["split"]
    pool_codes, pool_names = {}, []
    pool_of_row = []
    for k in pool_key:
        if k not in pool_codes:
            pool_codes[k] = len(pool_names)
            pool_names.append(k)
        pool_of_row.append(pool_codes[k])
    pool_members = [[] for _ in pool_names]
    for row, p in enumerate(pool_of_row):
        pool_members[p].append(row)
    pool_off = np.concatenate([[0], np.cumsum([len(m) for m in pool_members])]).astype(np.int64)

    out = {
        "ids": ids,
        "runs_flat": torch.tensor(runs_flat, dtype=torch.int16) if runs_flat else torch.zeros((0, 2), dtype=torch.int16),
        "runs_offset": torch.tensor(runs_offset, dtype=torch.int64),
        "he_offset": torch.tensor(he_offset, dtype=torch.int64),
        "he_size": torch.tensor(he_size, dtype=torch.int16),
        "he_flat": torch.cat(he_flat),
        "feat_offset": torch.tensor(feat_offset, dtype=torch.int64),
        "feat_flat": torch.cat(feat_flat),
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "structural_feature_names": list(STRUCTURAL_PAIR_FEATURES),
        "pair_feature_mean": torch.tensor(meta["pair_feature_mean"], dtype=torch.float32),
        "pair_feature_std": torch.tensor(meta["pair_feature_std"], dtype=torch.float32),
        "elem_feature_names": ["length"],
        "elem_feature_mean": torch.tensor(meta["elem_feature_mean"], dtype=torch.float32),
        "elem_feature_std": torch.tensor(meta["elem_feature_std"], dtype=torch.float32),
        "min_len": meta["min_len"],
        "contact_threshold": 0.5,
        "contact_def": meta["contact_def"],
        "dssp_def": meta["dssp_def"],
        "reference_source": "cat_pair",
        "element_types": meta["element_types"],
        "loop_axis": meta["loop_axis"],
        "template_gate": "none",
        "ca_flat": torch.cat(ca_flat),
        "elem_flat": torch.cat(elem_flat),
        "ca_offset": torch.tensor(ca_offset, dtype=torch.int64),
        "row_entry": torch.tensor([entry_codes[e] for e in members.entry_id], dtype=torch.int32),
        "row_src": torch.tensor([src_codes[entry_src[e]] for e in members.entry_id], dtype=torch.int32),
        "row_pool": torch.tensor(pool_of_row, dtype=torch.int32),
        "pool_names": pool_names,
        "pool_members_flat": torch.tensor([r for m in pool_members for r in m], dtype=torch.int32),
        "pool_members_offset": torch.from_numpy(pool_off),
    }
    assert out["pair_feature_mean"].numel() == N_PAIR_FEATURES
    torch.save(out, out_path)
    write_cbits(store, cbits)
    with open(out_path + ".skips.tsv", "w") as fh:
        fh.write("\n".join(skips) + ("\n" if skips else ""))
    print(f"wrote {out_path}: {len(ids)} member rows, {len(pool_names)} (CAT, split) pools; contact maps written into the store", flush=True)
    print("CATPAIR_MERGE_DONE", flush=True)


def write_cbits(store, cbits):
    """Shard-level cbit_off.npy / cbits.npy, entries in each shard's row order."""
    ent = store.entries
    for shard, g in ent.groupby("shard"):
        g = g.sort_values("row")
        assert g.row.tolist() == list(range(len(g))), f"shard {shard}: rows are not 0..n-1"
        chunks = [cbits[e] for e in g.entry_id]
        off = np.concatenate([[0], np.cumsum([c.size for c in chunks])]).astype(np.int64)
        d = os.path.join(store.root, "shards", f"s{int(shard):04d}")
        np.save(os.path.join(d, "cbit_off.npy"), off)
        np.save(os.path.join(d, "cbits.npy"), np.concatenate(chunks))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract-meta")
    e.add_argument("--index", required=True)
    e.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--meta", required=True)
    b.add_argument("--part", type=int, required=True)
    b.add_argument("--n-parts", type=int, required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--f2c-checkpoint", required=True)
    b.add_argument("--f2c-parent", default="~", help="directory containing the Frame2ConFind package")
    b.add_argument("--amp-dtype", default="bf16")
    b.add_argument("--batch-size", type=int, default=8, help="entries per Frame2ConFind forward (T8 winner rerun: 8)")
    m = sub.add_parser("merge")
    m.add_argument("--store", required=True)
    m.add_argument("--meta", required=True)
    m.add_argument("--parts-dir", required=True)
    m.add_argument("--n-parts", type=int, required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    {"extract-meta": cmd_extract_meta, "build": cmd_build, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
