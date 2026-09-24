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
# contact definition of the index: "cb8" (default, the v4 build) or "confind" (Frame2ConFind maps >= threshold:
# the native's stored contact_map_confind, the templates' maps from the `f2c` subcommand)
_CONTACT_SOURCE = "cb8"
_MAPS_ROOT = ""
_ROOTS = []
CONFIND_THRESHOLD = 0.01             # the `contact_method: confind` transform's threshold
CONTACT_DEF_CONFIND = "ConFind via Frame2ConFind, prob >= 0.01 (native: stored contact_map_confind)"


def f2c_map_path(maps_root, npz_path, roots):
    """Template npz -> its packed-map npz: <maps_root>/<root index>/<path relative to that root>."""
    for i, r in enumerate(roots):
        if npz_path.startswith(r.rstrip("/") + "/"):
            return os.path.join(maps_root, str(i), os.path.relpath(npz_path, r))
    raise ValueError(f"{npz_path} is under none of the template roots")


def confind_native_contacts(g, L):
    cm = getattr(g, "contact_map_confind", None)
    if cm is None or tuple(cm.shape) != (L, L):
        return None
    return (cm.float() >= CONFIND_THRESHOLD).float()  # same float 0/1 form as cb8_contacts


def load_f2c_maps(path):
    """Packed-map npz -> {rung k: bool [L, L]}."""
    z = np.load(path)
    L = int(z["L"])
    return {int(k): torch.from_numpy(np.unpackbits(z["packed"][i], count=L * L).reshape(L, L).astype(np.float32))
            for i, k in enumerate(z["rungs"])}


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
    # integer residue types, for the identity gate below: `sequence` may be a string or absent, and
    # the npz stores aatype indices, so this is the form the two can actually be compared in
    seq_ref = getattr(g, "residue_type", None)
    coords, cmask, dssp = getattr(g, "coords", None), getattr(g, "coord_mask", None), getattr(g, "dssp_target", None)
    if coords is None or cmask is None:
        return stem, None, [], [("native", "no_coords")], seq
    if dssp is None:
        return stem, None, [], [("native", "no_dssp_attr")], seq
    cmask = cmask.bool()
    if _CONTACT_SOURCE == "confind":
        cm_n = confind_native_contacts(g, int(coords.shape[0]))
        if cm_n is None:
            return stem, None, [], [("native", "no_contact_map_confind")], seq
    else:
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
    # ⛔⛔ IDENTITY GATE. The npz is located by NAME, and a name is not a molecule: T2 keys templates
    # on <pdbid>_<auth_asym_id> while proteina keys on <pdbid>_<label_asym_id>, so a string join can
    # hand back a DIFFERENT POLYMER of the same entry, with a perfectly plausible structure attached.
    # That defect reached 14.4% of our chains and no length check could see it (a mis-joined chain can
    # have the identical residue count). openfold's own consumer never suffered it because its pool
    # compares the npz sequence against the query on every draw -- this is the same guard, applied
    # once at build time. Refuse the template rather than train on another chain's topology.
    t_aatype = npz["aatype"] if "aatype" in npz else None
    if t_aatype is not None and seq_ref is not None:
        t_seq = np.asarray(t_aatype).astype(np.int16)
        q_seq = np.asarray(seq_ref).astype(np.int16)
        if t_seq.shape != q_seq.shape or not bool((t_seq == q_seq).all()):
            skips.append(("templates", f"identity_mismatch:npz_{t_seq.shape[0]}_vs_query_{q_seq.shape[0]}"))
            return stem, native, rows, skips, seq
    tcoords, amask = npz["coords"], torch.from_numpy(npz["atom_mask"].astype(bool))
    L_t = int(amask.shape[0])
    f2c = None
    if _CONTACT_SOURCE == "confind":
        mp = f2c_map_path(_MAPS_ROOT, npz_path, _ROOTS)
        if not os.path.exists(mp):
            skips.append(("templates", "no_f2c_maps"))
            return stem, native, rows, skips, seq
        f2c = load_f2c_maps(mp)
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
            if f2c is not None:
                cm_t = f2c.get(k)
                if cm_t is None:
                    skips.append((f"tpl#{k}", "no_f2c_map_for_rung"))
                    continue
            else:
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


def _enumerate_jobs(args):
    """Chain list of this part -> builder job tuples (the alias join and template lookup live here only)."""
    base = torch.load(args.base_index, map_location="cpu", weights_only=False, mmap=True)
    ids = [str(s) for s in base["ids"]]
    del base
    manifest = json.load(open(args.manifest)) if args.manifest and os.path.exists(args.manifest) else None
    roots = load_template_roots(args.templates)
    mine = ids[args.part::args.n_parts]
    if args.limit > 0:
        mine = mine[: args.limit]
    print(f"part {args.part}/{args.n_parts}: {len(mine)} chains", flush=True)

    # ⛔ The ONE translation layer. The template pool is keyed by auth_asym_id, the dataset by
    # label_asym_id; a string join between them silently selects a DIFFERENT POLYMER (measured: 14.4%
    # of our chains). The alias is applied HERE, once, and FAILS CLOSED: a stem absent from the alias
    # gets NO template and a recorded reason, never a fallback to its own name. That silent
    # passthrough is what produced the defect.
    alias = {}
    if args.chain_alias:
        with open(args.chain_alias) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            il = head.index("label_id")
            ia = head.index("auth_id")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) > max(il, ia) and f[ia]:
                    alias[f[il]] = f[ia]
        print(f"chain alias: {len(alias)} label->auth entries from {args.chain_alias}", flush=True)

    jobs = []
    n_alias_miss = 0
    for stem in mine:
        pt = str(_processed_path_sharded(Path(args.processed_dir), stem, manifest))
        if args.chain_alias:
            key = alias.get(stem)
            if not key:
                n_alias_miss += 1
                jobs.append((stem, pt, None, None, None, None, args.min_len, args.tm_min,
                             args.tm_max, bool(args.selftest)))
                continue
        else:
            key = stem
        npz, tm, rw, sl = find_template(key, roots)
        jobs.append((stem, pt, npz, tm, rw, sl, args.min_len, args.tm_min, args.tm_max, bool(args.selftest)))
    if args.chain_alias:
        print(f"  {n_alias_miss} chains have no alias entry -> no template (recorded, not guessed)",
              flush=True)
    return mine, jobs, roots


def cmd_build(args):
    global _USALIGN, _CONTACT_SOURCE, _MAPS_ROOT, _ROOTS
    _USALIGN = args.usalign
    _CONTACT_SOURCE, _MAPS_ROOT = args.contact_source, args.f2c_maps
    if _CONTACT_SOURCE == "confind" and not _MAPS_ROOT:
        raise SystemExit("--contact-source confind needs --f2c-maps")
    mine, jobs, roots = _enumerate_jobs(args)
    _ROOTS = [r[0] for r in roots]

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
                "tm_build_range": (args.tm_min, args.tm_max),
                "contact_def": CONTACT_DEF_CONFIND if _CONTACT_SOURCE == "confind" else CONTACT_DEF}, part_path)
    with open(os.path.join(args.out_dir, f"skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    by_reason = Counter(l.split("\t")[2].split(":")[0] for l in skip_lines)
    print(f"wrote {part_path}: {len(out_rows)} chains, {n_tpl} template rows; skips by reason: {dict(by_reason)}", flush=True)
    print("SYNTH_INDEX_PART_DONE", flush=True)


# ──────────────────────────────────────────── f2c ────────────────────────────────────────────
F2C_MAX_LEN = 384                    # Frame2ConFind was trained to length 384


def _f2c_inputs(full, amask):
    """atom37 [n, L, 37, 3] + mask [L, 37] -> Frame2ConFind x [n, L, 5, 3] (N, CA, C, CB, O) and residue mask."""
    from proteinfoundation.utils.frame2confind_utils import _place_cb
    x = full[:, :, :5, :].clone()                       # atom37 order N, CA, C, CB, O == the model's order
    mask = amask[:, 0] & amask[:, 1] & amask[:, 2]
    miss = ~amask[:, 3] & mask
    if miss.any():
        for j in range(x.shape[0]):
            x[j, miss, 3] = _place_cb(full[j, miss, 0], full[j, miss, 1], full[j, miss, 2])
    return x, mask


def cmd_f2c(args):
    """Frame2ConFind maps (thresholded, bit-packed) for every template rung the build would keep."""
    sys.path.insert(0, os.path.expanduser(args.f2c_parent))
    from Frame2ConFind.inference.api import Frame2ConFindPredictor
    predictor = Frame2ConFindPredictor(checkpoint=os.path.expanduser(args.checkpoint), amp_dtype=args.amp_dtype,
                                       compile_model=False)
    print(f"f2c model {args.checkpoint} on {predictor.device}", flush=True)
    mine, jobs, roots = _enumerate_jobs(args)
    root_dirs = [r[0] for r in roots]
    skip_lines, n_done, n_maps = [], 0, 0
    import time
    t_infer = 0.0

    if args.native_check > 0:
        from proteinfoundation.utils.precompute_frame2confind_maps import _collate_items, _graph_to_f2s_item
        agree, jac, worst, n_chk = [], [], 0.0, 0
        items = []
        for stem, pt, *_ in jobs:
            if len(items) >= args.native_check:
                break
            if not os.path.exists(pt):
                continue
            g = torch.load(pt, map_location="cpu", weights_only=False)
            ref = getattr(g, "contact_map_confind", None)
            it = _graph_to_f2s_item(g)
            if ref is None or it is None or it["length"] > F2C_MAX_LEN:
                continue
            items.append((it, ref.float()))
        # native_batch > 1 reproduces the backfill's batching: longest first, padded to the batch max
        items.sort(key=lambda x: -x[0]["length"])
        preds = []
        for i in range(0, len(items), args.native_batch):
            chunk = [x[0] for x in items[i:i + args.native_batch]]
            col = _collate_items(chunk)
            pb = predictor.predict_batch(col["x_f2s"], col["mask"]).float().cpu()
            preds += [pb[j, :c["length"], :c["length"]] for j, c in enumerate(chunk)]
        for pr, (_, ref) in zip(preds, items):
            worst = max(worst, (pr - ref).abs().max().item())
            a, b = pr >= CONFIND_THRESHOLD, ref >= CONFIND_THRESHOLD
            agree.append((a == b).float().mean().item())
            # overlap of the CONTACTS themselves; cell agreement is inflated by the ~97% negatives
            jac.append(((a & b).sum() / (a | b).sum().clamp_min(1)).item())
            n_chk += 1
        js = sorted(jac)
        print(f"NATIVE_CHECK n={n_chk} amp={args.amp_dtype} batch={args.native_batch} cell-agreement min={min(agree):.6f} mean={sum(agree)/len(agree):.6f} "
              f"| contact Jaccard min={js[0]:.4f} median={js[len(js)//2]:.4f} mean={sum(js)/len(js):.4f} "
              f"| max|prob diff|={worst:.4g}", flush=True)

    for stem, pt, npz_path, band_tm, _rw, band_slot, _ml, tm_min, tm_max, _st in jobs:
        if npz_path is None or band_tm is None:
            continue
        out = f2c_map_path(args.maps_out, npz_path, root_dirs)
        if os.path.exists(out):
            n_done += 1
            continue
        z = np.load(npz_path)
        tcoords, amask = z["coords"], torch.from_numpy(z["atom_mask"].astype(bool))
        L = int(amask.shape[0])
        if L > F2C_MAX_LEN:
            skip_lines.append(f"{stem}\ttemplates\tlen_gt_{F2C_MAX_LEN}:{L}")
            continue
        slot_to_rung = {int(sl): r for r, sl in enumerate(band_slot.tolist()) if sl >= 0}
        rungs = [k for k in range(int(tcoords.shape[0]))
                 if k in slot_to_rung and tm_min <= float(band_tm[slot_to_rung[k]]) <= tm_max]
        if not rungs:
            skip_lines.append(f"{stem}\ttemplates\tno_rung_in_tm_range")
            continue
        packed = []
        for i in range(0, len(rungs), args.batch_size):
            ks = rungs[i:i + args.batch_size]
            full = torch.zeros(len(ks), L, 37, 3)
            for j, k in enumerate(ks):
                full[j][amask] = torch.from_numpy(tcoords[k]).float()
            x, mask = _f2c_inputs(full, amask)
            t0 = time.perf_counter()
            probs = predictor.predict_batch(x, mask[None].expand(len(ks), -1))[:, :L, :L].float().cpu()
            t_infer += time.perf_counter() - t0
            bits = (probs >= CONFIND_THRESHOLD).numpy().reshape(len(ks), -1)
            packed.append(np.packbits(bits, axis=1))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, rungs=np.asarray(rungs, dtype=np.int16), L=np.int32(L), packed=np.concatenate(packed),
                 threshold=np.float32(CONFIND_THRESHOLD), checkpoint=str(args.checkpoint))
        os.replace(tmp, out)
        n_done += 1
        n_maps += len(rungs)
        if n_done % 200 == 0:
            print(f"  {n_done} chains, {n_maps} maps this run", flush=True)
    os.makedirs(args.maps_out, exist_ok=True)
    with open(os.path.join(args.maps_out, f"f2c_skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    print(f"part {args.part}: {n_done} chains with maps, {n_maps} new maps, {len(skip_lines)} skips; "
          f"inference {t_infer:.1f} s = {t_infer / max(n_maps, 1):.4f} s/map", flush=True)
    print("F2C_PART_DONE", flush=True)


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
    # chain-selection args shared by build and f2c, so both select exactly the same template rungs
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--base-index", required=True, help="existing topology_index.pt; its ids define the chain list")
    sel.add_argument("--processed-dir", required=True)
    sel.add_argument("--manifest", default="", help="shard_manifest.json of the processed dir")
    sel.add_argument("--templates", action="append", required=True, help="ROOT:index_band.npz, repeatable")
    sel.add_argument("--chain-alias", default="",
                     help="TSV with label_id and auth_id columns. The template pool is keyed by "
                          "auth_asym_id and the dataset by label_asym_id; without this they join by "
                          "string and silently select a different polymer. Fails closed: a stem with "
                          "no alias entry gets no template.")
    sel.add_argument("--part", type=int, required=True)
    sel.add_argument("--n-parts", type=int, required=True)
    sel.add_argument("--min-len", type=int, default=1)
    sel.add_argument("--tm-min", type=float, default=0.0, help="build-time filter on template rows (default: keep the whole band)")
    sel.add_argument("--tm-max", type=float, default=1.0)
    sel.add_argument("--limit", type=int, default=0, help="debug: only the first N chains of the part")
    sel.add_argument("--selftest", action="store_true", help="also USalign every native to itself and record mismatches (debug)")
    b = sub.add_parser("build", parents=[sel])
    b.add_argument("--out-dir", required=True)
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--usalign", default=_USALIGN)
    b.add_argument("--contact-source", choices=("cb8", "confind"), default="cb8")
    b.add_argument("--f2c-maps", default="", help="maps root written by `f2c` (required for --contact-source confind)")
    f = sub.add_parser("f2c", parents=[sel])
    f.add_argument("--maps-out", required=True)
    f.add_argument("--checkpoint", default="~/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/best.pt")
    f.add_argument("--f2c-parent", default="~", help="directory containing the Frame2ConFind package")
    f.add_argument("--amp-dtype", default="bf16")
    f.add_argument("--batch-size", type=int, default=4, help="rungs per forward (the backfill used 4)")
    f.add_argument("--native-check", type=int, default=0, help="compare F2C on N natives vs their stored maps first")
    f.add_argument("--native-batch", type=int, default=1, help="natives per forward in the check (backfill used 4)")
    m = sub.add_parser("merge")
    m.add_argument("--parts-dir", required=True)
    m.add_argument("--n-parts", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--eligible-out", default="")
    m.add_argument("--tm-range", type=float, nargs=2, default=(0.5, 0.9))
    args = ap.parse_args()
    {"build": cmd_build, "f2c": cmd_f2c, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
