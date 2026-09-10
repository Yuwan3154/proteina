#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Build the SYNTHETIC-reference topology index consumed by TopologyReferenceTransform(
reference_source="synthetic").

Rows are grouped per chain: the chain's own NATIVE row (DSSP runs, CB-8 SSE contacts, structural
pair features) followed by one row per synthetic template (a Protpardelle-1c partial-diffusion
variant of that native, from the T2 band tree: `<root>/shard{crc32(chain)%1000:04d}/<chain>.npz`
with `index_band.npz` supplying each row's template-vs-native TM and rewind). Every template row
also carries the USalign (`-TMscore 0`, sequence-independent) alignment of the native onto it,
expressed as the reference helix/strand ELEMENT each native residue aligns to (-1 = none), which is
the ground truth for the Q x T alignment head.

Contacts are CB-8 Å in ContactEBM's convention (CB where resolved else CA, d <= 8, diagonal 1);
the native reads CB at ATOM_NUMBERING index 4 (disk order), the templates at atom37 index 3.

Restartable: `build` writes one part file for an interleaved slice of the chain list (a SLURM
array), `merge` concatenates the parts into the flat index and writes the list of chains that have
at least one template inside the training TM range (the sampler's eligible set). Every skipped
chain or template row is written to the part's skip log with a reason -- a count is not an audit.

Usage:
  python -m proteinfoundation.utils.precompute_synthetic_topology_index build \
      --base-index $DATA_PATH/pdb_train/topology_index.pt \
      --processed-dir $DATA_PATH/pdb_train/processed --manifest $DATA_PATH/pdb_train/shard_manifest.json \
      --templates /orcd/compute/so3/002/chenxi/of_run/pp1c_work/templates_band:/orcd/compute/so3/002/chenxi/of_run/pp1c_work/index_band.npz \
      --part 0 --n-parts 40 --out-dir /orcd/scratch/orcd/011/chenxiou/synth_index/parts
  python -m proteinfoundation.utils.precompute_synthetic_topology_index merge \
      --parts-dir /orcd/scratch/orcd/011/chenxiou/synth_index/parts --n-parts 40 \
      --out /orcd/scratch/orcd/011/chenxiou/synth_index/topology_index_cb8_synth.pt \
      --eligible-out /orcd/scratch/orcd/011/chenxiou/synth_index/eligible_ids_tm0.5-0.9.txt --tm-range 0.5 0.9
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proteinfoundation.datasets.pdb_data import _processed_path_sharded  # noqa: E402
from proteinfoundation.datasets.sse_topology import (  # noqa: E402
    DSSP_HELIX,
    DSSP_STRAND,
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    STRUCTURAL_PAIR_FEATURES,
    assemble_pair_features,
    dssp_to_runs,
    runs_to_spans,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.utils.dssp_utils import compute_dssp_target  # noqa: E402

CB_CUTOFF = 8.0                      # ContactEBM CB_CONTACT_CUTOFF
CA_IDX = 1                           # CA is index 1 in BOTH atom orders
CB_DISK_IDX, CB_ATOM37_IDX = 4, 3    # ATOM_NUMBERING (on-disk .pt) vs atom37 (templates)
ALIGN_NONE = -1
CONTACT_DEF = "CB<=8.0A (CA where CB unresolved), diag 1; ContactEBM convention"

# Set once per worker from the CLI (a module global survives the fork; argparse objects do not).
_USALIGN = "/home/chenxiou/.local/bin/USalign"


# ─────────────────────────────────── per-structure pieces ────────────────────────────────────
def cb8_contacts(coords: torch.Tensor, atom_mask: torch.Tensor, cb_idx: int) -> torch.Tensor:
    """ContactEBM's CB-8 map: CB where resolved else CA, d <= 8, symmetric, diag 1; residues with no
    CA carry no contacts (the binary-map port of ContactEBM's pair mask)."""
    cb = coords[:, cb_idx, :].clone().float()
    miss = atom_mask[:, cb_idx] < 0.5
    cb[miss] = coords[miss, CA_IDX, :].float()
    d = torch.cdist(cb[None], cb[None])[0]
    cm = d <= CB_CUTOFF
    has_ca = atom_mask[:, CA_IDX] >= 0.5
    return (cm & has_ca[:, None] & has_ca[None, :]).float()


def topology_row(cm, coords, atom_mask, dssp, min_len):
    """(runs, contact bytes, T, structural bytes, stats, elem_of_residue) for one structure, or None
    when the DSSP carries no usable residue."""
    if dssp is None or bool((dssp < 0).all()):
        return None
    runs = dssp_to_runs(dssp, min_len=min_len)
    ref, keep = sse_contact_reference(cm, runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
    structural = sse_structural_pair_features(cm, coords, atom_mask, runs, keep)
    feat = assemble_pair_features(ref, structural, runs, keep)
    flat = feat.reshape(-1, N_PAIR_FEATURES)
    stats = (torch.stack([flat.sum(0), (flat ** 2).sum(0)]).tolist(), int(flat.shape[0]))
    spans = runs_to_spans(runs)
    elem = np.full((int(cm.shape[0]),), ALIGN_NONE, dtype=np.int16)
    for e, ia in enumerate(keep):
        s, t = spans[ia]
        elem[s:t] = e
    return (
        runs,
        ref.to(torch.uint8).flatten().numpy().tobytes(),
        len(keep),
        structural.to(torch.float16).flatten().numpy().tobytes(),
        stats,
        elem,
    )


def write_ca_pdb(ca_xyz: np.ndarray, present: np.ndarray, path: str):
    """CA-only PDB in structure order; returns written_index -> structure_index."""
    idx_map = []
    with open(path, "w") as fh:
        n = 0
        for i in range(ca_xyz.shape[0]):
            if not present[i]:
                continue
            x, y, z = (float(v) for v in ca_xyz[i])
            n += 1
            fh.write(f"ATOM  {n:5d}  CA  GLY A{n:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            idx_map.append(i)
        fh.write("TER\nEND\n")
    return idx_map


def usalign_pairs(pdb_q: str, pdb_r: str):
    """Aligned (query_written, ref_written) pairs and the query-normalised TM, sequence-independent."""
    out = subprocess.run([_USALIGN, pdb_q, pdb_r, "-TMscore", "0"], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"rc={out.returncode}")
    lines = out.stdout.splitlines()
    tm_q = float("nan")
    for l in lines:
        if l.startswith("TM-score=") and "Chain_1" in l:
            tm_q = float(l.split("=")[1].split()[0])
            break
    anchor = next((i for i, l in enumerate(lines) if l.startswith('(":" denotes')), None)
    if anchor is None or anchor + 3 >= len(lines):
        raise RuntimeError("no alignment block")
    seq_q, seq_r = lines[anchor + 1], lines[anchor + 3]
    if len(seq_q) != len(seq_r):
        raise RuntimeError("ragged alignment block")
    pairs, qi, ri = [], 0, 0
    for cq, cr in zip(seq_q, seq_r):
        q_res, r_res = cq != "-", cr != "-"
        if q_res and r_res:
            pairs.append((qi, ri))
        qi += q_res
        ri += r_res
    return pairs, tm_q


# ───────────────────────────────────── the per-chain worker ──────────────────────────────────
def _chain_job(args):
    """One chain -> (native row | None, [template rows], [(what, reason)] skips)."""
    stem, pt_path, npz_path, band_tm, band_rewind, band_slot, min_len, tm_min, tm_max, selftest = args
    skips = []
    if not os.path.exists(pt_path):
        return stem, None, [], [("native", "missing_file")], None
    try:
        g = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:  # recorded per chain and audited afterwards, never silent
        return stem, None, [], [("native", f"load_failed:{type(e).__name__}")], None
    seq = getattr(g, "sequence", None)
    coords, cmask, dssp = getattr(g, "coords", None), getattr(g, "coord_mask", None), getattr(g, "dssp_target", None)
    if coords is None or cmask is None:
        return stem, None, [], [("native", "no_coords")], seq
    if dssp is None:
        return stem, None, [], [("native", "no_dssp_attr")], seq
    cmask = cmask.bool()
    cm_n = cb8_contacts(coords, cmask, CB_DISK_IDX)
    native = topology_row(cm_n, coords, cmask, dssp, min_len)
    if native is None:
        return stem, None, [], [("native", "dssp_all_ignore")], seq
    L_n = int(coords.shape[0])

    rows = []
    if npz_path is None:
        skips.append(("templates", "no_npz"))
        return stem, native, rows, skips, seq
    if band_tm is None:
        skips.append(("templates", "no_band_index_entry"))
        return stem, native, rows, skips, seq
    npz = np.load(npz_path)
    tcoords, amask = npz["coords"], torch.from_numpy(npz["atom_mask"].astype(bool))
    L_t = int(amask.shape[0])
    slot_to_rung = {int(s): r for r, s in enumerate(band_slot.tolist()) if s >= 0}
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "q.pdb")
        qmap = write_ca_pdb(coords[:, CA_IDX].numpy(), cmask[:, CA_IDX].numpy(), pq)
        if not qmap:
            skips.append(("templates", "native_has_no_ca"))
            return stem, native, rows, skips, seq
        if selftest:
            # the probe's gate: a chain aligned to ITSELF must map every element residue onto its own
            # element -- catches an index-map, parse or span off-by-one that a cross-chain number hides
            pairs, _ = usalign_pairs(pq, pq)
            elem_n = native[5]
            self_align = np.full((L_n,), ALIGN_NONE, dtype=np.int16)
            for qw, rw in pairs:
                self_align[qmap[qw]] = elem_n[qmap[rw]]
            mism = int((self_align != elem_n).sum())
            skips.append(("selftest", f"self_alignment_mismatch_residues:{mism}"))
        for k in range(int(tcoords.shape[0])):
            r = slot_to_rung.get(k)
            if r is None:
                skips.append((f"tpl#{k}", "row_not_in_band_index"))
                continue
            tm, rewind = float(band_tm[r]), int(band_rewind[r])
            if not (tm_min <= tm <= tm_max):
                skips.append((f"tpl#{k}", f"tm_out_of_build_range:{tm:.3f}"))
                continue
            full = torch.zeros(L_t, 37, 3)
            full[amask] = torch.from_numpy(tcoords[k]).float()
            dssp_t = compute_dssp_target(full[None], amask[:, CA_IDX][None], coord_mask=amask[None], coord_layout="atom37")
            if dssp_t is None:
                skips.append((f"tpl#{k}", "tpl_dssp_none"))
                continue
            cm_t = cb8_contacts(full, amask, CB_ATOM37_IDX)
            row = topology_row(cm_t, full, amask, dssp_t[0], min_len)
            if row is None:
                skips.append((f"tpl#{k}", "tpl_dssp_all_ignore"))
                continue
            pr = os.path.join(td, f"r{k}.pdb")
            rmap = write_ca_pdb(full[:, CA_IDX].numpy(), amask[:, CA_IDX].numpy(), pr)
            if not rmap:
                skips.append((f"tpl#{k}", "tpl_has_no_ca"))
                continue
            try:
                pairs, tm_us = usalign_pairs(pq, pr)
            except Exception as e:  # recorded; the row is dropped, not silently zeroed
                skips.append((f"tpl#{k}", f"usalign_failed:{e}"))
                continue
            elem_t = row[5]
            align = np.full((L_n,), ALIGN_NONE, dtype=np.int16)
            for qw, rw in pairs:
                align[qmap[qw]] = elem_t[rmap[rw]]
            rows.append((row[:5], tm, rewind, k, align.tobytes(), tm_us))
    return stem, native, rows, skips, seq


def _hash_sequence(seq, stem):
    key = seq if isinstance(seq, str) and seq else stem
    return int(hashlib.blake2b(str(key).encode(), digest_size=8).hexdigest(), 16) % (2 ** 62)


# ──────────────────────────────────────────── build ──────────────────────────────────────────
def load_template_roots(specs):
    """--templates ROOT:INDEX_BAND.npz (repeatable) -> [(root, chains->row, tm, rewind, slot)]."""
    roots = []
    for spec in specs:
        root, idx_path = spec.split(":")
        z = np.load(idx_path, allow_pickle=True)
        chains = {str(c): i for i, c in enumerate(z["chains"])}
        roots.append((root, chains, z["tm"], z["rewind"], z["slot"], float(z["min_tm"]), float(z["max_tm"])))
        print(f"templates: {root} ({len(chains)} chains, band {float(z['min_tm']):.2f}-{float(z['max_tm']):.2f})", flush=True)
    return roots


def find_template(stem, roots):
    for root, chains, tm, rewind, slot, _, _ in roots:
        p = os.path.join(root, f"shard{zlib.crc32(stem.encode()) % 1000:04d}", f"{stem}.npz")
        if os.path.exists(p):
            i = chains.get(stem)
            if i is None:
                return p, None, None, None
            return p, tm[i], rewind[i], slot[i]
    return None, None, None, None


def cmd_build(args):
    global _USALIGN
    _USALIGN = args.usalign
    base = torch.load(args.base_index, map_location="cpu", weights_only=False, mmap=True)
    ids = [str(s) for s in base["ids"]]
    del base
    manifest = json.load(open(args.manifest)) if args.manifest and os.path.exists(args.manifest) else None
    roots = load_template_roots(args.templates)
    mine = ids[args.part::args.n_parts]
    if args.limit > 0:
        mine = mine[: args.limit]
    print(f"part {args.part}/{args.n_parts}: {len(mine)} chains", flush=True)

    jobs = []
    for stem in mine:
        pt = str(_processed_path_sharded(Path(args.processed_dir), stem, manifest))
        npz, tm, rw, sl = find_template(stem, roots)
        jobs.append((stem, pt, npz, tm, rw, sl, args.min_len, args.tm_min, args.tm_max, bool(args.selftest)))

    out_rows = []          # per chain: dict
    skip_lines = []
    n_tpl = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for j, (stem, native, rows, skips, seq) in enumerate(ex.map(_chain_job, jobs, chunksize=4)):
            for what, reason in skips:
                skip_lines.append(f"{stem}\t{what}\t{reason}")
            out_rows.append({"stem": stem, "native": native, "rows": rows, "seq_hash": _hash_sequence(seq, stem)})
            n_tpl += len(rows)
            if (j + 1) % 500 == 0:
                print(f"  {j + 1}/{len(mine)} chains, {n_tpl} template rows, {len(skip_lines)} skips", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    part_path = os.path.join(args.out_dir, f"part_{args.part:04d}.pt")
    torch.save({"chains": out_rows, "n_parts": args.n_parts, "part": args.part, "min_len": args.min_len,
                "tm_build_range": (args.tm_min, args.tm_max), "contact_def": CONTACT_DEF}, part_path)
    with open(os.path.join(args.out_dir, f"skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    by_reason = Counter(l.split("\t")[2].split(":")[0] for l in skip_lines)
    print(f"wrote {part_path}: {len(out_rows)} chains, {n_tpl} template rows; skips by reason: {dict(by_reason)}", flush=True)
    print("SYNTH_INDEX_PART_DONE", flush=True)


# ──────────────────────────────────────────── merge ──────────────────────────────────────────
def cmd_merge(args):
    ids, cluster_of, seq_hash = [], [], []
    members_flat, members_offset = [], [0]
    runs_flat, runs_offset = [], [0]
    he_flat, he_offset, he_size = [], [0], []
    feat_flat, feat_offset = [], [0]
    align_flat, align_offset = [], [0]
    row_tm, row_is_native, row_rewind, row_usalign_tm = [], [], [], []
    feat_sum = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    feat_sumsq = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    feat_count = 0
    eligible = []
    n_chains_with_tpl = 0
    lo, hi = args.tm_range

    def push(runs, ref_bytes, t_he, feat_bytes, stats, is_tpl):
        nonlocal feat_count
        runs_flat.extend(runs)
        runs_offset.append(runs_offset[-1] + len(runs))
        he_size.append(t_he)
        ref = torch.frombuffer(bytearray(ref_bytes), dtype=torch.uint8) if ref_bytes else torch.zeros(0, dtype=torch.uint8)
        he_flat.append(ref)
        he_offset.append(he_offset[-1] + ref.numel())
        f = torch.frombuffer(bytearray(feat_bytes), dtype=torch.float16) if feat_bytes else torch.zeros(0, dtype=torch.float16)
        feat_flat.append(f)
        feat_offset.append(feat_offset[-1] + f.numel())
        if is_tpl:  # the transform standardises REFERENCES, and references are templates
            sums, cnt = stats
            feat_sum.add_(torch.tensor(sums[0], dtype=torch.float64))
            feat_sumsq.add_(torch.tensor(sums[1], dtype=torch.float64))
            feat_count += cnt

    min_len, contact_def = None, None
    for part in range(args.n_parts):
        p = os.path.join(args.parts_dir, f"part_{part:04d}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p} -- rebuild that part before merging")
        d = torch.load(p, map_location="cpu", weights_only=False)
        min_len, contact_def = d["min_len"], d["contact_def"]
        for ch in d["chains"]:
            if ch["native"] is None:
                continue                                # skipped chains are in the skip logs
            g = len(members_offset) - 1
            ids.append(ch["stem"])
            cluster_of.append(g)
            seq_hash.append(ch["seq_hash"])
            runs, ref_b, t_he, feat_b, stats, _elem = ch["native"]
            push(runs, ref_b, t_he, feat_b, stats, is_tpl=False)
            align_offset.append(align_offset[-1])
            row_tm.append(1.0)
            row_is_native.append(True)
            row_rewind.append(0)
            row_usalign_tm.append(1.0)
            any_in_range = False
            for (rrow, tm, rewind, k, align_b, tm_us) in ch["rows"]:
                members_flat.append(len(ids))
                ids.append(f"{ch['stem']}@rw{rewind}#{k}")
                cluster_of.append(g)
                seq_hash.append(ch["seq_hash"])
                push(rrow[0], rrow[1], rrow[2], rrow[3], rrow[4], is_tpl=True)
                a = torch.frombuffer(bytearray(align_b), dtype=torch.int16)
                align_flat.append(a)
                align_offset.append(align_offset[-1] + a.numel())
                row_tm.append(tm)
                row_is_native.append(False)
                row_rewind.append(rewind)
                row_usalign_tm.append(tm_us)
                any_in_range |= lo <= tm <= hi
            members_offset.append(len(members_flat))
            if ch["rows"]:
                n_chains_with_tpl += 1
            if any_in_range:
                eligible.append(ch["stem"])
        print(f"merged part {part}: rows so far {len(ids)}", flush=True)

    mean = feat_sum / max(feat_count, 1)
    std = (feat_sumsq / max(feat_count, 1) - mean ** 2).clamp(min=0.0).sqrt().clamp(min=1e-6)
    for name, m, s in zip(PAIR_FEATURE_NAMES, mean.tolist(), std.tolist()):
        print(f"  {name:<24} mean={m:10.4f} std={s:10.4f}", flush=True)
    out = {
        "ids": ids,
        "cluster_of": torch.tensor(cluster_of, dtype=torch.int32),
        "members_flat": torch.tensor(members_flat, dtype=torch.int32),
        "members_offset": torch.tensor(members_offset, dtype=torch.int64),
        "seq_hash": torch.tensor(seq_hash, dtype=torch.int64),
        "runs_flat": torch.tensor(runs_flat, dtype=torch.int16) if runs_flat else torch.zeros((0, 2), dtype=torch.int16),
        "runs_offset": torch.tensor(runs_offset, dtype=torch.int64),
        "he_offset": torch.tensor(he_offset, dtype=torch.int64),
        "he_size": torch.tensor(he_size, dtype=torch.int16),
        "he_flat": torch.cat(he_flat) if he_flat else torch.zeros(0, dtype=torch.uint8),
        "feat_offset": torch.tensor(feat_offset, dtype=torch.int64),
        "feat_flat": torch.cat(feat_flat) if feat_flat else torch.zeros(0, dtype=torch.float16),
        "align_offset": torch.tensor(align_offset, dtype=torch.int64),
        "align_flat": torch.cat(align_flat) if align_flat else torch.zeros(0, dtype=torch.int16),
        "row_tm": torch.tensor(row_tm, dtype=torch.float16),
        "row_is_native": torch.tensor(row_is_native, dtype=torch.bool),
        "row_rewind": torch.tensor(row_rewind, dtype=torch.int16),
        "row_usalign_tm": torch.tensor(row_usalign_tm, dtype=torch.float16),
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "structural_feature_names": list(STRUCTURAL_PAIR_FEATURES),
        "pair_feature_mean": mean.to(torch.float32),
        "pair_feature_std": std.to(torch.float32),
        "min_len": min_len,
        "contact_threshold": 0.5,       # the map is already binary; kept for consumers that read the key
        "contact_def": contact_def,
        "reference_source": "synthetic",
    }
    torch.save(out, args.out)
    n_native = int(sum(row_is_native))
    print(f"wrote {args.out}: {len(ids)} rows = {n_native} natives + {len(ids) - n_native} template rows; "
          f"{n_chains_with_tpl} chains have templates; {len(eligible)} chains eligible in TM {lo}-{hi}", flush=True)
    if args.eligible_out:
        with open(args.eligible_out, "w") as fh:
            fh.write("\n".join(eligible) + "\n")
        print(f"eligible ids -> {args.eligible_out}", flush=True)
    print("SYNTH_INDEX_MERGE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base-index", required=True, help="existing topology_index.pt; its ids define the chain list")
    b.add_argument("--processed-dir", required=True)
    b.add_argument("--manifest", default="", help="shard_manifest.json of the processed dir")
    b.add_argument("--templates", action="append", required=True, help="ROOT:index_band.npz, repeatable")
    b.add_argument("--part", type=int, required=True)
    b.add_argument("--n-parts", type=int, required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--min-len", type=int, default=1)
    b.add_argument("--tm-min", type=float, default=0.0, help="build-time filter on template rows (default: keep the whole band)")
    b.add_argument("--tm-max", type=float, default=1.0)
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--usalign", default=_USALIGN)
    b.add_argument("--limit", type=int, default=0, help="debug: only the first N chains of the part")
    b.add_argument("--selftest", action="store_true", help="also USalign every native to itself and record mismatches (debug)")
    m = sub.add_parser("merge")
    m.add_argument("--parts-dir", required=True)
    m.add_argument("--n-parts", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--eligible-out", default="")
    m.add_argument("--tm-range", type=float, nargs=2, default=(0.5, 0.9))
    args = ap.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    else:
        cmd_merge(args)


if __name__ == "__main__":
    main()
