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

T8b final-selection mode (--final-selection; default off, so the T8 pack-mode build is unchanged):
  (1) templates = the pack's final_selection.npz (t2_select_diverse's 64/chain greedy min-max pick, read via FinalSet)
      instead of the cap-64-evenly-in-TM pick; the TM window (--tm-min/--tm-max) and the CA clash gate still apply, no
      cap -- user 2026-10-20 "Rebuild from final_selection".
  (2) two packs: --template-pack + --final-selection (the original t2_synth_v2 pack) and --override-pack +
      --override-selection (the T8b-only regen pack, user 2026-10-22 "T8b-only pack"). A chain listed in
      --override-chains (the T8b regen list) takes its templates from the override pack ONLY; one with no templates
      there gets NONE, never the original pack's (those were built from corrupted natives) -- from user 2026-10-21
      "Regenerate them" + the 2026-10-24 regen list; the precedence / zero-template rule is the plan's 2026-10-25
      reading of those answers, flagged to the user.
  (3) natives from the T8b overlay training pack (--native-pack tri_t8b_512.pack; same reader as T8).
  (4) template keys carry the pack identity: row id <stem>@rw<rewind>#<pack tag>/<round>/<draw>, and Frame2ConFind
      maps live in <maps root>/<pack tag>/<crc32 shard>/<stem>.npz keyed by (round, draw) with the tag stamped inside.
      `f2c --reuse-maps` copies a T8 map (t8_f2c_maps, keyed by draw only, computed from the pre-append pack) for a
      base-round template whose (shard, off, L, A) in the pre-append tmpl.idx equals the current row; `f2c-plan`
      counts reusable vs to-compute maps without a GPU.
"""

import argparse
import hashlib
import io
import json
import mmap
import os
import subprocess
import sys
import tempfile
import time
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
    DSSP_LOOP,
    DSSP_STRAND,
    LOOP_AXIS_MODES,
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    STRUCTURAL_PAIR_FEATURES,
    assemble_pair_features,
    dssp_to_runs,
    element_lengths,
    runs_to_spans,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.utils.dssp_utils import compute_dssp_target  # noqa: E402
from proteinfoundation.utils.t2_pack_reader import FinalSet, layout  # noqa: E402

CB_CUTOFF = 8.0                      # ContactEBM CB_CONTACT_CUTOFF
CA_IDX = 1                           # CA is index 1 in BOTH atom orders
CB_DISK_IDX, CB_ATOM37_IDX = 4, 3    # ATOM_NUMBERING (on-disk .pt) vs atom37 (templates)
ALIGN_NONE = -1
# T8 (user 2026-10-09): which DSSP types are reference ELEMENTS (default helix + strand = the v4 build) and how a loop
# element's axis is defined when loops are elements (--element-types / --loop-axis; pinned in every part and the index)
_ELEMENT_TYPES = (DSSP_HELIX, DSSP_STRAND)
_LOOP_AXIS = None
CONTACT_DEF = "CB<=8.0A (CA where CB unresolved), diag 1; ContactEBM convention"

# Set once per worker from the CLI (a module global survives the fork; argparse objects do not).
_USALIGN = "/home/chenxiou/.local/bin/USalign"
# contact definition of the index: "cb8" (default, the v4 build) or "confind" (Frame2ConFind maps >= threshold:
# the native's stored contact_map_confind, the templates' maps from the `f2c` subcommand)
_CONTACT_SOURCE = "cb8"
# --proline-donor-mask (user 2026-10-09): recompute every DSSP (native too) with proline unable to donate a backbone H-bond
_PRO_DONOR_MASK = False
PRO_IDX = 14                         # openfold restype order
DSSP_DEF = "native: stored dssp_target; templates: pydssp; no donor mask"
DSSP_DEF_PRO = "native + templates: pydssp with donor_mask = (aatype != PRO); native validity = N/CA/C/O coord_mask > 0.5"
_MAPS_ROOT = ""
_ROOTS = []
# --template-pack (T8, user 2026-10-12/13): templates from the t2 template pack (TemplatePack, keyed by the dataset's own
# label chain id); opened in the parent, memory-mapped lazily per forked worker
_TPACK = None
# --native-pack (A6000 build, user 2026-10-09): natives read from the training pack (the processed .pt bytes) instead of
# --processed-dir; stem -> (offset, length), opened once in the parent so the forked workers share it
_PACK_IDX, _PACK_MM = None, None
# --max-ca-clashes / --ca-clash-a (T8, user 2026-10-13 "gate at 3.4 A" + "<=4"): a template with more than k non-bonded
# (|i - j| >= 2) CA-CA pairs closer than the cutoff is dropped BEFORE the cap; (k, cutoff, cap) or None = no gate
_CA_CLASH_GATE = None
# --final-selection (T8b, see the module docstring): [(pack tag, TemplatePack, pack root)], index 0 = --template-pack, 1 =
# --override-pack; the template-source definition stamped into every part (JSON string; None outside this mode)
_SOURCES = []
_TEMPLATE_SOURCE = None
# f2c --reuse-pack-idx: the pre-append tmpl.idx the T8 maps were computed from (columns, loaded in cmd_f2c / f2c-plan)
_REUSE_IDX = None
# build: the stamps every final-selection maps file must carry (set in cmd_build; dict or None outside confind mode)
_MAP_EXPECT = None
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


def pack_map_path(maps_root, stem):
    """Pack mode: one maps file per chain, sharded by crc32 (<= ~100 files per directory)."""
    return os.path.join(maps_root, f"{zlib.crc32(stem.encode()) % 1000:04d}", f"{stem}.npz")


def load_pack_maps(path):
    """Pack-mode maps npz -> {draw: float 0/1 [L, L]} (templates differ in length: per-template L and byte offsets)."""
    z = np.load(path)
    out = {}
    for i, d in enumerate(z["draws"]):
        L = int(z["L"][i])
        bits = z["packed"][int(z["byte_off"][i]):int(z["byte_off"][i + 1])]
        out[int(d)] = torch.from_numpy(np.unpackbits(bits, count=L * L).reshape(L, L).astype(np.float32))
    return out


def final_map_path(maps_root, tag, stem):
    """Final-selection mode: one maps file per (pack tag, chain); the tag directory keeps the packs apart."""
    return os.path.join(maps_root, tag, f"{zlib.crc32(stem.encode()) % 1000:04d}", f"{stem}.npz")


def check_map_stamps(z, path, tag, expect):
    """A final-selection maps file must match the run: pack tag, batch size, checkpoint path + md5, and no T8 copies
    unless reuse is allowed (expect: batch_size, checkpoint, checkpoint_md5, allow_reuse)."""
    got = {"pack_tag": str(z["pack_tag"]), "batch_size": int(z["batch_size"]), "checkpoint": str(z["checkpoint"]),
           "checkpoint_md5": str(z["checkpoint_md5"])}
    want = {"pack_tag": tag, "batch_size": int(expect["batch_size"]), "checkpoint": str(expect["checkpoint"]),
            "checkpoint_md5": expect["checkpoint_md5"]}
    bad = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
    if not expect["allow_reuse"] and "t8_reuse" in set(map(str, z["origin"])):
        bad["origin"] = ("t8_reuse present", "reuse off")
    if bad:
        raise SystemExit(f"{path}: maps file does not match this run {bad}")


def load_final_maps(path, tag, expect):
    """Final-selection maps npz -> {(round, draw): float 0/1 [L, L]}; refuses a file whose stamps differ (pack tag,
    batch size, checkpoint, reuse)."""
    z = np.load(path)
    check_map_stamps(z, path, tag, expect)
    out = {}
    for i, (rd, d) in enumerate(zip(z["rounds"], z["draws"])):
        L = int(z["L"][i])
        bits = z["packed"][int(z["byte_off"][i]):int(z["byte_off"][i + 1])]
        out[(str(rd), int(d))] = torch.from_numpy(np.unpackbits(bits, count=L * L).reshape(L, L).astype(np.float32))
    return out


def template_key_id(tag, rnd, draw):
    return f"{tag}/{rnd}/{draw}"


def select_pack_rows(tp, stem, tm_min, tm_max, cap, keep=None):
    """User 2026-10-13 "(b)": at most `cap` templates per chain, spread evenly over the TM range -- evenly in TM VALUE:
    for each of `cap` targets linspace(tm_min, tm_max, cap), the nearest still-unused in-range row (ties: lower draw).
    `keep` (row -> bool, optional) filters the in-range rows BEFORE the cap (the clash gate); cap None = no cap.
    Returned in (tm_native, draw) order."""
    if stem not in tp:
        return None
    tm, draw = tp.col["tm_native"], tp.col["draw"]
    rows = [r for r in tp.rows(stem) if tm_min <= float(tm[r]) <= tm_max]
    assert len({int(draw[r]) for r in rows}) == len(rows), f"{stem}: draw ids repeat (rows, maps and ids key on draw)"
    if keep is not None:
        rows = [r for r in rows if keep(r)]
    if cap is not None and len(rows) > cap:
        free, picked = set(rows), []
        for t in np.linspace(tm_min, tm_max, cap):
            r = min(free, key=lambda r: (abs(float(tm[r]) - t), int(draw[r])))
            free.discard(r)
            picked.append(r)
        rows = picked
    rows.sort(key=lambda r: (float(tm[r]), int(draw[r])))
    return rows


# ─────────────────────────────────── per-structure pieces ────────────────────────────────────
def ca_clash_pairs(ca: np.ndarray, present: np.ndarray, cutoff: float) -> int:
    """Non-bonded (residue index |i - j| >= 2) CA-CA pairs closer than `cutoff` (same metric as t8_clash_gate_stats)."""
    rn = np.flatnonzero(present)
    c = ca[rn]
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    i, j = np.triu_indices(len(rn), k=1)
    return int(((rn[j] - rn[i] >= 2) & (d[i, j] < cutoff)).sum())


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


_FROM_GLOBALS = object()


def topology_row(cm, coords, atom_mask, dssp, min_len, element_types=_FROM_GLOBALS, loop_axis=_FROM_GLOBALS):
    """(runs, contact bytes, T, structural bytes, stats, elem_of_residue) for one structure, or None
    when the DSSP carries no usable residue. element_types / loop_axis default to the module globals the
    `build` command sets; the CAT-pair builder passes them explicitly."""
    element_types = _ELEMENT_TYPES if element_types is _FROM_GLOBALS else element_types
    loop_axis = _LOOP_AXIS if loop_axis is _FROM_GLOBALS else loop_axis
    if dssp is None or bool((dssp < 0).all()):
        return None
    runs = dssp_to_runs(dssp, min_len=min_len)
    ref, keep = sse_contact_reference(cm, runs, keep_types=element_types)
    structural = sse_structural_pair_features(cm, coords, atom_mask, runs, keep, loop_axis=loop_axis)
    feat = assemble_pair_features(ref, structural, runs, keep)
    flat = feat.reshape(-1, N_PAIR_FEATURES)
    el = element_lengths(runs, keep)  # stats for the per-element length feature (standardised by the transform)
    stats = (torch.stack([flat.sum(0), (flat ** 2).sum(0)]).tolist(), int(flat.shape[0]),
             [float(el.sum()), float((el ** 2).sum())], int(el.numel()))
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


def usalign_pairs(pdb_q: str, pdb_r: str, n_q: int, n_r: int, binary: str = None):
    """Aligned (query_written, ref_written) pairs and the query-normalised TM, sequence-independent. n_q / n_r = residues
    written to each file: USalign must have read exactly those (it reads ATOM records only), or the indices would shift."""
    out = subprocess.run([binary or _USALIGN, pdb_q, pdb_r, "-TMscore", "0"], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"rc={out.returncode}")
    lines = out.stdout.splitlines()
    tm_q = float("nan")
    for l in lines:
        # US-align 20241108 says "Structure_1", older builds "Chain_1"
        if l.startswith("TM-score=") and ("Structure_1" in l or "Chain_1" in l):
            tm_q = float(l.split("=")[1].split()[0])
            break
    if tm_q != tm_q:
        raise RuntimeError("no query-normalised TM-score line")
    anchor = next((i for i, l in enumerate(lines) if l.startswith('(":" denotes')), None)
    if anchor is None or anchor + 3 >= len(lines):
        raise RuntimeError("no alignment block")
    seq_q, seq_r = lines[anchor + 1], lines[anchor + 3]
    if len(seq_q) != len(seq_r):
        raise RuntimeError("ragged alignment block")
    read_q, read_r = len(seq_q) - seq_q.count("-"), len(seq_r) - seq_r.count("-")
    if (read_q, read_r) != (n_q, n_r):
        raise RuntimeError(f"residues_read_{read_q}_{read_r}_vs_written_{n_q}_{n_r}")
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
    stem, pt_path, npz_path, band_tm, band_rewind, band_slot, min_len, tm_min, tm_max, selftest, pack_rows = args
    skips = []
    if _PACK_IDX is not None:
        ent = _PACK_IDX.get(stem)
        if ent is None:
            return stem, None, [], [("native", "not_in_pack")], None
        src = io.BytesIO(_PACK_MM[ent[0]:ent[0] + ent[1]])
    elif not os.path.exists(pt_path):
        return stem, None, [], [("native", "missing_file")], None
    else:
        src = pt_path
    try:
        g = torch.load(src, map_location="cpu", weights_only=False)
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
    if _PRO_DONOR_MASK:
        if seq_ref is None or len(seq_ref) != int(coords.shape[0]):
            return stem, None, [], [("native", "no_residue_type_for_donor_mask")], seq
        # same atoms and validity as precompute_dssp_targets._compute_dssp (disk order, O = 3), plus the donor mask
        dssp = compute_dssp_target(coords[None].float(), torch.ones(1, int(coords.shape[0]), dtype=torch.bool),
                                   coord_mask=(cmask.float() > 0.5)[None], coord_layout="pdb",
                                   donor_mask=(torch.as_tensor(seq_ref) != PRO_IDX)[None])
        if dssp is None:
            return stem, None, [], [("native", "native_dssp_none")], seq
        dssp = dssp[0]
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
    if _SOURCES:
        return _final_chain_rows(stem, native, coords, cmask, L_n, pack_rows, min_len, skips, seq)
    if _TPACK is not None:
        return _pack_chain_rows(stem, native, coords, cmask, L_n, pack_rows, min_len, skips, seq)
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
            pairs, _ = usalign_pairs(pq, pq, len(qmap), len(qmap))
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
            cm_t = None
            if f2c is not None:
                cm_t = f2c.get(k)
                if cm_t is None:
                    skips.append((f"tpl#{k}", "no_f2c_map_for_rung"))
                    continue
            got, why = _template_row(full, amask, t_aatype, cm_t, pq, qmap, L_n, td, k, min_len)
            if got is None:
                skips.append((f"tpl#{k}", why))
                continue
            rows.append((got[0], tm, rewind, k, got[1], got[2]))
    return stem, native, rows, skips, seq


def _pack_chain_rows(stem, native, coords, cmask, L_n, pack_rows, min_len, skips, seq):
    """Pack mode: the chain's selected template rows (select_pack_rows) -> template rows. No sequence or length gate
    (user 2026-10-13: templates may be redesigned and indels change the length); the pack is keyed by the dataset's
    own label chain id, so there is no auth/label translation to get wrong."""
    rows = []
    if pack_rows is None:
        skips.append(("templates", "not_in_template_pack"))
        return stem, native, rows, skips, seq
    if not pack_rows:
        skips.append(("templates", "no_template_in_tm_range"))
        return stem, native, rows, skips, seq
    if _CA_CLASH_GATE is not None:
        # the parent passed every in-range row uncapped; gate here (decoding is the cost), THEN cap evenly in TM
        max_k, cutoff, cap, tm_min, tm_max = _CA_CLASH_GATE
        n_clash = {}
        for r in pack_rows:
            rec = _TPACK._read(r)
            full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
            full[rec["atom_mask"]] = rec["coords"]
            n_clash[r] = ca_clash_pairs(full[:, CA_IDX], rec["atom_mask"][:, CA_IDX], cutoff)
            if n_clash[r] > max_k:
                skips.append((f"tpl#{rec['draw']}", f"ca_clashes_gt_{max_k}:{n_clash[r]}"))
        pack_rows = select_pack_rows(_TPACK, stem, tm_min, tm_max, cap, keep=lambda r: n_clash[r] <= max_k)
        if not pack_rows:
            skips.append(("templates", "no_template_after_clash_gate"))
            return stem, native, rows, skips, seq
    f2c = None
    if _CONTACT_SOURCE == "confind":
        mp = pack_map_path(_MAPS_ROOT, stem)
        if not os.path.exists(mp):
            skips.append(("templates", "no_f2c_maps"))
            return stem, native, rows, skips, seq
        f2c = load_pack_maps(mp)
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "q.pdb")
        qmap = write_ca_pdb(coords[:, CA_IDX].numpy(), cmask[:, CA_IDX].numpy(), pq)
        if not qmap:
            skips.append(("templates", "native_has_no_ca"))
            return stem, native, rows, skips, seq
        for r in pack_rows:
            rec = _TPACK._read(r)
            k = rec["draw"]
            amask = torch.from_numpy(rec["atom_mask"])
            full = torch.zeros(int(amask.shape[0]), 37, 3)
            full[amask] = torch.from_numpy(rec["coords"]).float()
            cm_t = None
            if f2c is not None:
                cm_t = f2c.get(k)
                if cm_t is None or tuple(cm_t.shape) != (int(amask.shape[0]),) * 2:
                    skips.append((f"tpl#{k}", "no_f2c_map_for_draw"))
                    continue
            got, why = _template_row(full, amask, rec["aatype"], cm_t, pq, qmap, L_n, td, k, min_len)
            if got is None:
                skips.append((f"tpl#{k}", why))
                continue
            rows.append((got[0], rec["tm_native"], rec["rewind"], k, got[1], got[2]))
    return stem, native, rows, skips, seq


def _final_chain_rows(stem, native, coords, cmask, L_n, sel, min_len, skips, seq):
    """Final-selection mode: sel = (source index, in-window rows of that ONE pack | None, note); the parent resolved the
    pack precedence. Clash gate as in pack mode, no cap; the row key carries the pack tag."""
    src, pack_rows, note = sel
    rows = []
    if note:
        skips.append(("templates", note))
    if pack_rows is None:
        return stem, native, rows, skips, seq
    if not pack_rows:
        skips.append(("templates", "no_template_in_tm_range"))
        return stem, native, rows, skips, seq
    tag, tp, _ = _SOURCES[src]
    if _CA_CLASH_GATE is not None:
        max_k, cutoff = _CA_CLASH_GATE[0], _CA_CLASH_GATE[1]
        kept = []
        for r in pack_rows:
            rec = tp._read(r)
            full = np.zeros((rec["atom_mask"].shape[0], 37, 3), dtype=np.float32)
            full[rec["atom_mask"]] = rec["coords"]
            n = ca_clash_pairs(full[:, CA_IDX], rec["atom_mask"][:, CA_IDX], cutoff)
            if n > max_k:
                skips.append((f"tpl#{template_key_id(tag, rec['round'], rec['draw'])}", f"ca_clashes_gt_{max_k}:{n}"))
            else:
                kept.append(r)
        pack_rows = kept
        if not pack_rows:
            skips.append(("templates", "no_template_after_clash_gate"))
            return stem, native, rows, skips, seq
    f2c = None
    if _CONTACT_SOURCE == "confind":
        mp = final_map_path(_MAPS_ROOT, tag, stem)
        if not os.path.exists(mp):
            skips.append(("templates", "no_f2c_maps"))
            return stem, native, rows, skips, seq
        f2c = load_final_maps(mp, tag, _MAP_EXPECT)
    with tempfile.TemporaryDirectory() as td:
        pq = os.path.join(td, "q.pdb")
        qmap = write_ca_pdb(coords[:, CA_IDX].numpy(), cmask[:, CA_IDX].numpy(), pq)
        if not qmap:
            skips.append(("templates", "native_has_no_ca"))
            return stem, native, rows, skips, seq
        for r in pack_rows:
            rec = tp._read(r)
            kid = template_key_id(tag, rec["round"], rec["draw"])
            amask = torch.from_numpy(rec["atom_mask"])
            full = torch.zeros(int(amask.shape[0]), 37, 3)
            full[amask] = torch.from_numpy(rec["coords"]).float()
            cm_t = None
            if f2c is not None:
                cm_t = f2c.get((rec["round"], rec["draw"]))
                if cm_t is None or tuple(cm_t.shape) != (int(amask.shape[0]),) * 2:
                    skips.append((f"tpl#{kid}", "no_f2c_map_for_draw"))
                    continue
            got, why = _template_row(full, amask, rec["aatype"], cm_t, pq, qmap, L_n, td, r, min_len)
            if got is None:
                skips.append((f"tpl#{kid}", why))
                continue
            rows.append((got[0], rec["tm_native"], rec["rewind"], kid, got[1], got[2]))
    return stem, native, rows, skips, seq


def _template_row(full, amask, t_aatype, cm_t, pq, qmap, L_n, td, k, min_len):
    """One template (atom37 coords [L_t, 37, 3], mask [L_t, 37], its own aatype) -> ((topology row[:5], USalign alignment
    bytes, USalign TM), None), or (None, skip reason). cm_t None = CB-8 contacts from the coordinates."""
    donor = None
    if _PRO_DONOR_MASK:
        if t_aatype is None or len(t_aatype) != int(amask.shape[0]):
            return None, "no_aatype_for_donor_mask"
        donor = torch.from_numpy(np.asarray(t_aatype) != PRO_IDX)[None]
    dssp_t = compute_dssp_target(full[None], amask[:, CA_IDX][None], coord_mask=amask[None], coord_layout="atom37",
                                 donor_mask=donor)
    if dssp_t is None:
        return None, "tpl_dssp_none"
    if cm_t is None:
        cm_t = cb8_contacts(full, amask, CB_ATOM37_IDX)
    row = topology_row(cm_t, full, amask, dssp_t[0], min_len)
    if row is None:
        return None, "tpl_dssp_all_ignore"
    pr = os.path.join(td, f"r{k}.pdb")
    rmap = write_ca_pdb(full[:, CA_IDX].numpy(), amask[:, CA_IDX].numpy(), pr)
    if not rmap:
        return None, "tpl_has_no_ca"
    try:
        pairs, tm_us = usalign_pairs(pq, pr, len(qmap), len(rmap))
    except Exception as e:  # recorded; the row is dropped, not silently zeroed
        return None, f"usalign_failed:{e}"
    elem_t = row[5]
    align = np.full((L_n,), ALIGN_NONE, dtype=np.int16)
    for qw, rw in pairs:
        align[qmap[qw]] = elem_t[rmap[rw]]
    return (row[:5], align.tobytes(), tm_us), None


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
    global _TPACK
    if not (args.chain_list or args.base_index):
        raise SystemExit("give --chain-list or --base-index")
    if args.chain_list:
        ids = [l.strip() for l in open(args.chain_list) if l.strip()]
    else:
        base = torch.load(args.base_index, map_location="cpu", weights_only=False, mmap=True)
        ids = [str(s) for s in base["ids"]]
        del base
    manifest = json.load(open(args.manifest)) if args.manifest and os.path.exists(args.manifest) else None
    if bool(args.templates) == bool(args.template_pack):
        raise SystemExit("give exactly one of --templates (npz band tree) or --template-pack")
    roots = load_template_roots(args.templates) if args.templates else []
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

    if args.final_selection:
        return mine, _enumerate_final(args, mine, manifest), roots
    if args.template_pack:
        from proteinfoundation.utils.t2_pack_reader import TemplatePack
        _TPACK = TemplatePack(args.template_pack)
        # load every index column HERE: the lazy np.load zip handle must not be shared by forked workers
        # (a zipfile's seek+read is not safe across processes), and the columns are then shared copy-on-write
        for k in _TPACK.col._z.files:
            _TPACK.col[k]
        if args.tm_min_given is None or args.tm_max_given is None:
            raise SystemExit("pack mode: give --tm-min and --tm-max explicitly (T8: 0.5 0.9, user 2026-10-10)")
        if args.chain_alias:
            raise SystemExit("pack mode: --chain-alias does not apply (the pack is keyed by the dataset's label id)")
        print(f"template pack {args.template_pack}: {len(_TPACK.stems)} chains; cap {args.max_templates} per chain, "
              f"TM {args.tm_min}-{args.tm_max}, spread evenly over TM", flush=True)
        jobs = []
        for stem in mine:
            pt = str(_processed_path_sharded(Path(args.processed_dir), stem, manifest))
            # with the clash gate the cap is applied in the worker, after the gate
            cap = None if getattr(args, "max_ca_clashes", None) is not None else args.max_templates
            sel = select_pack_rows(_TPACK, stem, args.tm_min, args.tm_max, cap)
            jobs.append((stem, pt, None, None, None, None, args.min_len, args.tm_min, args.tm_max, bool(args.selftest), sel))
        return mine, jobs, roots

    jobs = []
    n_alias_miss = 0
    for stem in mine:
        pt = str(_processed_path_sharded(Path(args.processed_dir), stem, manifest))
        if args.chain_alias:
            key = alias.get(stem)
            if not key:
                n_alias_miss += 1
                jobs.append((stem, pt, None, None, None, None, args.min_len, args.tm_min,
                             args.tm_max, bool(args.selftest), None))
                continue
        else:
            key = stem
        npz, tm, rw, sl = find_template(key, roots)
        jobs.append((stem, pt, npz, tm, rw, sl, args.min_len, args.tm_min, args.tm_max, bool(args.selftest), None))
    if args.chain_alias:
        print(f"  {n_alias_miss} chains have no alias entry -> no template (recorded, not guessed)",
              flush=True)
    return mine, jobs, roots


def _md5(path):
    return hashlib.md5(open(os.path.expanduser(path), "rb").read()).hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _final_rows(fs, stem):
    """The chain's final-selection (round, draw) keys -> pack rows; a key absent from the pack fails the build."""
    tp = fs.tp
    by_key = {(tp.round_of(r), int(tp.col["draw"][r])): r for r in tp.rows(stem)}
    if len(by_key) != len(tp.rows(stem)):
        raise SystemExit(f"{stem}: (round, draw) repeats in the pack (maps and ids key on it)")
    keys = fs.keys(stem)
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"{stem}: final_selection keys not in the pack: {missing[:5]}")
    return [by_key[k] for k in keys]


def _enumerate_final(args, mine, manifest):
    """Final-selection mode job tuples; the last element is (source index, in-window rows | None, note)."""
    global _SOURCES, _TEMPLATE_SOURCE
    _SOURCES = []
    if args.tm_min_given is None or args.tm_max_given is None:
        raise SystemExit("final-selection mode: give --tm-min and --tm-max explicitly (T8: 0.5 0.9, user 2026-10-10)")
    if args.chain_alias:
        raise SystemExit("final-selection mode: --chain-alias does not apply (the packs are keyed by the label id)")
    over = (args.override_pack, args.override_selection, args.override_chains)
    if any(over) and not all(over):
        raise SystemExit("give --override-pack, --override-selection and --override-chains together")
    specs = [(args.template_pack_tag, args.template_pack, args.final_selection)]
    if args.override_pack:
        specs.append((args.override_pack_tag, args.override_pack, args.override_selection))
    if len({t for t, _, _ in specs}) != len(specs) or not all(t for t, _, _ in specs):
        raise SystemExit(f"pack tags must be non-empty and distinct: {[t for t, _, _ in specs]}")
    sets = []
    for tag, root, path in specs:
        fs = FinalSet(root, path)
        for k in fs.tp.col._z.files:  # load every column before the fork (see pack mode)
            fs.tp.col[k]
        sets.append(fs)
        _SOURCES.append((tag, fs.tp, root))
        print(f"final selection [{tag}] {path}: {len(fs.stems)} chains, {len(fs.draw)} templates "
              f"(pack {root}: {len(fs.tp.stems)} chains)", flush=True)
    override = set()
    if args.override_pack:
        override = {l.strip() for l in open(args.override_chains) if l.strip()}
        extra = sorted(set(map(str, sets[1].stems)) - override)
        if extra:
            raise SystemExit(f"{len(extra)} override-selection chains are not in --override-chains: {extra[:5]}")
    _TEMPLATE_SOURCE = json.dumps({
        "mode": "final_selection", "tm_window": [args.tm_min, args.tm_max], "cap": None,
        "packs": [{"tag": t, "root": r, "selection": p, "selection_sha256": _sha256(p)} for t, r, p in specs],
        "override_chains": args.override_chains or None,
        "override_chains_sha256": _sha256(args.override_chains) if args.override_chains else None,
        "rule": "override-listed chains take the override pack's final selection only (none there = no templates)",
    }, sort_keys=True)
    jobs, n = [], Counter()
    for stem in mine:
        pt = str(_processed_path_sharded(Path(args.processed_dir), stem, manifest))
        src = 1 if stem in override else 0
        fs, tp = sets[src], sets[src].tp
        if src == 1 and stem in sets[0]:
            n["override_replaces_original"] += 1
        if stem not in fs or not fs.keys(stem):  # a selection entry with count 0 = no templates
            why = "override_chain_without_override_templates" if src == 1 else "not_in_final_selection"
            n[why] += 1
            sel = (src, None, why)
        else:
            rows = _final_rows(fs, stem)
            tm = tp.col["tm_native"]
            keep = sorted((r for r in rows if args.tm_min <= float(tm[r]) <= args.tm_max),
                          key=lambda r: (float(tm[r]), tp.round_of(r), int(tp.col["draw"][r])))
            n_out = len(rows) - len(keep)
            n[f"templates_in_window[{specs[src][0]}]"] += len(keep)
            n[f"templates_out_of_window[{specs[src][0]}]"] += n_out
            sel = (src, keep, f"final_selection_tm_out_of_window:{n_out}" if n_out else "")
        jobs.append((stem, pt, None, None, None, None, args.min_len, args.tm_min, args.tm_max, bool(args.selftest), sel))
    print(f"final-selection jobs: {len(jobs)} chains, {len(override & set(mine))} in the override list; {dict(n)}",
          flush=True)
    return jobs


def _gate_def():
    if _CA_CLASH_GATE is None:
        return "none"
    if _CA_CLASH_GATE[2] is None:
        return (f"drop templates with > {_CA_CLASH_GATE[0]} non-bonded (|i-j|>=2) CA-CA pairs < {_CA_CLASH_GATE[1]} A; "
                f"no cap (templates = the final selection)")
    return (f"drop templates with > {_CA_CLASH_GATE[0]} non-bonded (|i-j|>=2) CA-CA pairs < {_CA_CLASH_GATE[1]} A, "
            f"before the cap of {_CA_CLASH_GATE[2]} evenly in TM")


def cmd_build(args):
    global _USALIGN, _CONTACT_SOURCE, _MAPS_ROOT, _ROOTS, _PRO_DONOR_MASK, _PACK_IDX, _PACK_MM, _ELEMENT_TYPES, _LOOP_AXIS
    global _CA_CLASH_GATE, _MAP_EXPECT
    _USALIGN = args.usalign
    if (args.max_ca_clashes is None) != (args.ca_clash_a is None):
        raise SystemExit("give --max-ca-clashes and --ca-clash-a together (T8: 4 and 3.4, user 2026-10-13)")
    if args.max_ca_clashes is not None:
        if not args.template_pack:
            raise SystemExit("--max-ca-clashes applies to pack mode only")
        _CA_CLASH_GATE = (args.max_ca_clashes, args.ca_clash_a, args.max_templates, args.tm_min, args.tm_max)
    _PRO_DONOR_MASK = args.proline_donor_mask
    _ELEMENT_TYPES = tuple(int(x) for x in args.element_types.split(","))
    _LOOP_AXIS = args.loop_axis
    if (DSSP_LOOP in _ELEMENT_TYPES) != (_LOOP_AXIS is not None):
        raise SystemExit("--loop-axis is required exactly when --element-types includes loops (0)")
    if args.native_pack:
        z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
        _PACK_IDX = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
        fh = open(args.native_pack, "rb")  # kept open for the mmap's lifetime (module global below)
        _PACK_MM = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        print(f"natives from pack {args.native_pack}: {len(_PACK_IDX)} stems", flush=True)
    _CONTACT_SOURCE, _MAPS_ROOT = args.contact_source, args.f2c_maps
    if _CONTACT_SOURCE == "confind" and not _MAPS_ROOT:
        raise SystemExit("--contact-source confind needs --f2c-maps")
    if args.final_selection and _CONTACT_SOURCE == "confind":
        if args.f2c_batch_size is None:
            raise SystemExit("final-selection confind build: give --f2c-batch-size (the f2c run's --batch-size)")
        _MAP_EXPECT = {"batch_size": args.f2c_batch_size, "checkpoint": args.f2c_checkpoint,
                       "checkpoint_md5": _md5(args.f2c_checkpoint), "allow_reuse": args.f2c_allow_reuse}
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
    part = {"chains": out_rows, "n_parts": args.n_parts, "part": args.part, "min_len": args.min_len,
            "tm_build_range": (args.tm_min, args.tm_max),
            "contact_def": CONTACT_DEF_CONFIND if _CONTACT_SOURCE == "confind" else CONTACT_DEF,
            "dssp_def": DSSP_DEF_PRO if _PRO_DONOR_MASK else DSSP_DEF,
            "element_types": list(_ELEMENT_TYPES), "loop_axis": _LOOP_AXIS, "template_gate": _gate_def()}
    if _TEMPLATE_SOURCE is not None:
        part["template_source"] = _TEMPLATE_SOURCE
        if args.native_pack:  # final-selection mode only (T8 parts stay byte-identical): which natives built the rows
            part["native_source"] = json.dumps({"native_pack": os.path.abspath(args.native_pack),
                                                "native_idx_md5": _md5(args.native_pack + ".idx.npz")}, sort_keys=True)
    torch.save(part, part_path)
    with open(os.path.join(args.out_dir, f"skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    by_reason = Counter(l.split("\t")[2].split(":")[0] for l in skip_lines)
    print(f"wrote {part_path}: {len(out_rows)} chains, {n_tpl} template rows; skips by reason: {dict(by_reason)}", flush=True)
    print("SYNTH_INDEX_PART_DONE", flush=True)


# ──────────────────────────────────────────── f2c ────────────────────────────────────────────
# The winner rerun that wrote the stored maps ran every length (bands to 512/768/1024/4096,
# backfill/winner_rerun.sh), so no cap here either: a cap would drop ~10% of template rows that v4 has.
F2C_MAX_LEN = 4096


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
            if ref is None or it is None or it["length"] > args.max_len:
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

    if _SOURCES:
        _load_reuse_idx(args)
        args.checkpoint_md5 = _md5(args.checkpoint)
        n_done, n_maps, t_infer = _final_f2c(args, jobs, predictor, skip_lines)
        jobs = []
    if _TPACK is not None:
        n_done, n_maps, t_infer = _pack_f2c(args, jobs, predictor, skip_lines)
        jobs = []  # pack mode done; the npz loop below is the band-tree path
    for stem, pt, npz_path, band_tm, _rw, band_slot, _ml, tm_min, tm_max, _st, _pr in jobs:
        if npz_path is None or band_tm is None:
            continue
        out = f2c_map_path(args.maps_out, npz_path, root_dirs)
        if os.path.exists(out):
            n_done += 1
            continue
        z = np.load(npz_path)
        tcoords, amask = z["coords"], torch.from_numpy(z["atom_mask"].astype(bool))
        L = int(amask.shape[0])
        if L > args.max_len:
            skip_lines.append(f"{stem}\ttemplates\tlen_gt_{args.max_len}:{L}")
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
    os.makedirs(os.path.join(args.maps_out, "_skips"), exist_ok=True)  # the root holds the 1000 crc shard dirs
    with open(os.path.join(args.maps_out, "_skips", f"f2c_skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    print(f"part {args.part}: {n_done} chains with maps, {n_maps} new maps, {len(skip_lines)} skips; "
          f"inference {t_infer:.1f} s = {t_infer / max(n_maps, 1):.4f} s/map", flush=True)
    print("F2C_PART_DONE", flush=True)


def _pack_f2c(args, jobs, predictor, skip_lines):
    """Pack mode: per chain, Frame2ConFind maps (thresholded, bit-packed) of its selected templates. Templates of one
    chain differ in length (indels): a batch is padded to its longest template, each map is cut to its own L."""
    import time
    n_done = n_maps = 0
    t_infer = 0.0
    for stem, *_rest, sel in jobs:
        if not sel:
            skip_lines.append(f"{stem}\ttemplates\t{'not_in_template_pack' if sel is None else 'no_template_in_tm_range'}")
            continue
        out = pack_map_path(args.maps_out, stem)
        if os.path.exists(out):
            n_done += 1
            continue
        recs = [_TPACK._read(r) for r in sel]
        keep = [rc for rc in recs if int(rc["atom_mask"].shape[0]) <= args.max_len]
        for rc in recs:
            if int(rc["atom_mask"].shape[0]) > args.max_len:
                skip_lines.append(f"{stem}\ttpl#{rc['draw']}\tlen_gt_{args.max_len}:{rc['atom_mask'].shape[0]}")
        if not keep:
            continue
        keep.sort(key=lambda rc: -int(rc["atom_mask"].shape[0]))  # similar lengths share a batch
        draws, Ls, bits = [], [], []
        for i in range(0, len(keep), args.batch_size):
            chunk = keep[i:i + args.batch_size]
            Lmax = max(int(rc["atom_mask"].shape[0]) for rc in chunk)
            full = torch.zeros(len(chunk), Lmax, 37, 3)
            amask = torch.zeros(len(chunk), Lmax, 37, dtype=torch.bool)
            for j, rc in enumerate(chunk):
                am = torch.from_numpy(rc["atom_mask"])
                L = int(am.shape[0])
                amask[j, :L] = am
                full[j, :L][am] = torch.from_numpy(rc["coords"]).float()
            xs, ms = zip(*(_f2c_inputs(full[j:j + 1], amask[j]) for j in range(len(chunk))))
            x, mask = torch.cat(xs), torch.stack(ms)
            t0 = time.perf_counter()
            probs = predictor.predict_batch(x, mask).float().cpu()
            t_infer += time.perf_counter() - t0
            for j, rc in enumerate(chunk):
                L = int(rc["atom_mask"].shape[0])
                draws.append(int(rc["draw"]))
                Ls.append(L)
                bits.append(np.packbits((probs[j, :L, :L] >= CONFIND_THRESHOLD).numpy().reshape(-1)))
        byte_off = np.concatenate([[0], np.cumsum([b.size for b in bits])]).astype(np.int64)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, draws=np.asarray(draws, dtype=np.int32), L=np.asarray(Ls, dtype=np.int32), byte_off=byte_off,
                 packed=np.concatenate(bits), threshold=np.float32(CONFIND_THRESHOLD), checkpoint=str(args.checkpoint))
        os.replace(tmp, out)
        n_done += 1
        n_maps += len(draws)
        if n_done % 200 == 0:
            print(f"  {n_done} chains, {n_maps} maps this run, {t_infer / max(n_maps, 1):.4f} s/map", flush=True)
    return n_done, n_maps, t_infer


def _load_reuse_idx(args):
    """--reuse-maps needs --reuse-pack-idx: the T8 maps key on draw only, so the template they were computed from is
    pinned by its pre-append (shard, off, L, A)."""
    global _REUSE_IDX
    if bool(args.reuse_maps) != bool(args.reuse_pack_idx):
        raise SystemExit("give --reuse-maps and --reuse-pack-idx together")
    if not args.reuse_maps:
        return
    z = np.load(args.reuse_pack_idx, allow_pickle=False)
    _REUSE_IDX = {k: z[k] for k in ("chain_stems", "chain_first", "chain_count", "draw", "shard", "off", "L", "A")}
    _REUSE_IDX["row"] = {str(s): i for i, s in enumerate(_REUSE_IDX["chain_stems"])}
    print(f"reuse: T8 maps {args.reuse_maps}, pre-append index {args.reuse_pack_idx} "
          f"({len(_REUSE_IDX['draw'])} templates)", flush=True)


def _t8_reuse(args, stem, tp, rows, why, dry):
    """{pack row: packed bits (None if dry)} of the rows whose T8 map can be copied; every other row's reason is
    counted in `why`."""
    old = pack_map_path(args.reuse_maps, stem)
    if not os.path.exists(old):
        why["no_t8_maps_file"] += len(rows)
        return {}
    z = np.load(old)
    if str(z["checkpoint"]) != str(args.checkpoint) or np.float32(z["threshold"]) != np.float32(CONFIND_THRESHOLD):
        why["t8_checkpoint_or_threshold_differs"] += len(rows)
        return {}
    pos = {int(d): i for i, d in enumerate(z["draws"])}
    pre = {}
    c = _REUSE_IDX["row"].get(stem)
    if c is not None:
        f = int(_REUSE_IDX["chain_first"][c])
        for q in range(f, f + int(_REUSE_IDX["chain_count"][c])):
            pre[int(_REUSE_IDX["draw"][q])] = tuple(int(_REUSE_IDX[k][q]) for k in ("shard", "off", "L", "A"))
    col, out = tp.col, {}
    for r in rows:
        d = int(col["draw"][r])
        if tp.round_of(r) != "base":
            why["not_base_round"] += 1
        elif d not in pos:
            why["draw_not_in_t8_maps"] += 1
        elif pre.get(d) != tuple(int(col[k][r]) for k in ("shard", "off", "L", "A")):
            why["pre_append_row_differs"] += 1
        elif int(z["L"][pos[d]]) != int(col["L"][r]):
            why["t8_map_length_differs"] += 1
        else:
            i = pos[d]
            out[r] = None if dry else z["packed"][int(z["byte_off"][i]):int(z["byte_off"][i + 1])].copy()
    return out


def _backbone_incomplete(tp, r):
    """True if any residue of template row r lacks N, CA or C in its atom mask (reads the mask bytes only)."""
    c = tp.col
    L, A = int(c["L"][r]), int(c["A"][r])
    o = int(c["off"][r]) + layout(L, A)["atom_mask"]
    m = np.frombuffer(tp._sh.get(int(c["shard"][r]))[o:o + 37 * L], dtype=np.uint8).reshape(L, 37)
    return not bool(m[:, :3].all())


def _final_f2c(args, jobs, predictor, skip_lines):
    """Final-selection mode: per chain, Frame2ConFind maps of its in-window final-selection templates (no clash gate:
    the superset any gate draws from, as T8), one file per (pack tag, chain) keyed by (round, draw). T8 maps are copied
    where _t8_reuse allows (--template-pack only), the rest computed exactly as pack mode does. predictor None = plan
    only: count, write nothing. Returns (chains written, maps computed, inference s) and fills args.plan_counts."""
    counts, why = Counter(), Counter()
    n_done = n_maps = 0
    t_infer = 0.0
    for stem, *_rest, sel in jobs:
        src, rows, note = sel
        if not rows:
            skip_lines.append(f"{stem}\ttemplates\t{note if rows is None else 'no_template_in_tm_range'}")
            continue
        tag, tp, _ = _SOURCES[src]
        col = tp.col
        out = final_map_path(args.maps_out, tag, stem)
        counts[f"{tag}:chains"] += 1
        counts[f"{tag}:templates"] += len(rows)
        if predictor is None:
            counts[f"{tag}:templates_missing_N_CA_C"] += sum(_backbone_incomplete(tp, r) for r in rows)
        if os.path.exists(out):
            if predictor is not None:
                check_map_stamps(np.load(out), out, tag, {"batch_size": args.batch_size, "checkpoint": args.checkpoint,
                                                          "checkpoint_md5": args.checkpoint_md5,
                                                          "allow_reuse": bool(args.reuse_maps)})
            counts[f"{tag}:chains_already_written"] += 1
            n_done += 1
            continue
        reuse = _t8_reuse(args, stem, tp, rows, why, predictor is None) if (src == 0 and args.reuse_maps) else {}
        todo = []
        for r in rows:
            if r in reuse:
                continue
            if int(col["L"][r]) > args.max_len:
                skip_lines.append(f"{stem}\ttpl#{template_key_id(tag, tp.round_of(r), int(col['draw'][r]))}\t"
                                  f"len_gt_{args.max_len}:{int(col['L'][r])}")
                continue
            todo.append(r)
        counts[f"{tag}:maps_reused"] += len(reuse)
        counts[f"{tag}:maps_to_compute"] += len(todo)
        counts[f"{tag}:maps_over_max_len"] += len(rows) - len(reuse) - len(todo)
        if predictor is None or not (reuse or todo):
            continue
        bits = dict(reuse)
        todo.sort(key=lambda r: -int(col["L"][r]))  # similar lengths share a batch
        for i in range(0, len(todo), args.batch_size):
            chunk = [tp._read(r) for r in todo[i:i + args.batch_size]]
            Lmax = max(int(rc["atom_mask"].shape[0]) for rc in chunk)
            full = torch.zeros(len(chunk), Lmax, 37, 3)
            amask = torch.zeros(len(chunk), Lmax, 37, dtype=torch.bool)
            for j, rc in enumerate(chunk):
                am = torch.from_numpy(rc["atom_mask"])
                L = int(am.shape[0])
                amask[j, :L] = am
                full[j, :L][am] = torch.from_numpy(rc["coords"]).float()
            xs, ms = zip(*(_f2c_inputs(full[j:j + 1], amask[j]) for j in range(len(chunk))))
            x, mask = torch.cat(xs), torch.stack(ms)
            t0 = time.perf_counter()
            probs = predictor.predict_batch(x, mask).float().cpu()
            t_infer += time.perf_counter() - t0
            for j, r in enumerate(todo[i:i + args.batch_size]):
                L = int(col["L"][r])
                bits[r] = np.packbits((probs[j, :L, :L] >= CONFIND_THRESHOLD).numpy().reshape(-1))
        order = [r for r in rows if r in bits]  # the chain's (TM, round, draw) order
        byte_off = np.concatenate([[0], np.cumsum([bits[r].size for r in order])]).astype(np.int64)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp.npz"
        np.savez(tmp, rounds=np.asarray([tp.round_of(r) for r in order]),
                 draws=np.asarray([int(col["draw"][r]) for r in order], dtype=np.int32),
                 L=np.asarray([int(col["L"][r]) for r in order], dtype=np.int32), byte_off=byte_off,
                 packed=np.concatenate([bits[r] for r in order]),
                 origin=np.asarray(["t8_reuse" if r in reuse else "computed" for r in order]),
                 threshold=np.float32(CONFIND_THRESHOLD), checkpoint=str(args.checkpoint),
                 checkpoint_md5=args.checkpoint_md5, batch_size=np.int32(args.batch_size), amp_dtype=args.amp_dtype,
                 pack_tag=tag, pack_root=_SOURCES[src][2])
        os.replace(tmp, out)
        n_done += 1
        n_maps += len(todo)
        if n_done % 200 == 0:
            print(f"  {n_done} chains, {n_maps} maps computed this run, {t_infer / max(n_maps, 1):.4f} s/map", flush=True)
    for k, v in why.items():
        counts[f"not_reused:{k}"] = v
    args.plan_counts = counts
    return n_done, n_maps, t_infer


def cmd_f2c_plan(args):
    """CPU, no model: per pack, how many final-selection maps the `f2c` run would copy from T8 vs compute."""
    if not args.final_selection:
        raise SystemExit("f2c-plan applies to --final-selection mode")
    mine, jobs, _ = _enumerate_jobs(args)
    _load_reuse_idx(args)
    skip_lines = []
    _final_f2c(args, jobs, None, skip_lines)
    counts = args.plan_counts
    for tag, tp, _root in _SOURCES:
        counts[f"{tag}:templates_missing_N_CA_C"] += 0  # information only: in-window templates with a residue lacking N/CA/C
        n_all = len(tp.col["draw"])
        counts[f"{tag}:pack_templates"] = n_all
        counts[f"{tag}:pack_templates_round_base"] = int((tp.col["round"] == "base").sum()) if tp.has_round else n_all
    os.makedirs(args.plan_out, exist_ok=True)
    with open(os.path.join(args.plan_out, f"f2c_plan_{args.part:04d}.tsv"), "w") as fh:
        fh.write("".join(f"{k}\t{v}\n" for k, v in sorted(counts.items())))
    with open(os.path.join(args.plan_out, f"f2c_plan_skips_{args.part:04d}.tsv"), "w") as fh:
        fh.write("\n".join(skip_lines) + ("\n" if skip_lines else ""))
    for k, v in sorted(counts.items()):
        print(f"  {k:<60} {v}", flush=True)
    print("F2C_PLAN_DONE", flush=True)


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
    elem_sum = elem_sumsq = 0.0
    elem_count = 0
    n_tpl_stats = {2: 0, 4: 0}  # template rows with / without element-length stats: never a mix
    eligible = []
    n_chains_with_tpl = 0
    lo, hi = args.tm_range

    def push(runs, ref_bytes, t_he, feat_bytes, stats, is_tpl):
        nonlocal feat_count, elem_sum, elem_sumsq, elem_count
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
            sums, cnt = stats[0], stats[1]
            feat_sum.add_(torch.tensor(sums[0], dtype=torch.float64))
            feat_sumsq.add_(torch.tensor(sums[1], dtype=torch.float64))
            feat_count += cnt
            n_tpl_stats[len(stats)] += 1
            if len(stats) == 4:  # parts written before the element-length stats existed carry 2 entries
                elem_sum += stats[2][0]
                elem_sumsq += stats[2][1]
                elem_count += stats[3]

    min_len, contact_def, dssp_defs, elem_defs, gate_defs, src_defs = None, None, set(), set(), set(), set()
    nat_defs = set()
    for part in range(args.n_parts):
        p = os.path.join(args.parts_dir, f"part_{part:04d}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p} -- rebuild that part before merging")
        d = torch.load(p, map_location="cpu", weights_only=False)
        min_len, contact_def = d["min_len"], d["contact_def"]
        dssp_defs.add(d.get("dssp_def", DSSP_DEF))  # parts written before the key existed used DSSP_DEF
        elem_defs.add((tuple(d.get("element_types", (DSSP_HELIX, DSSP_STRAND))), d.get("loop_axis", None)))
        gate_defs.add(d.get("template_gate", "none"))  # parts written before the key existed had no gate
        src_defs.add(d.get("template_source"))  # None = not final-selection mode
        nat_defs.add(d.get("native_source"))  # None = parts without the stamp (T8 / pre-stamp final-selection)
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
    if len(dssp_defs) != 1:  # parts built with and without the donor mask must never be merged
        raise ValueError(f"parts disagree on the DSSP definition: {sorted(dssp_defs)}")
    if n_tpl_stats[2] and n_tpl_stats[4]:
        raise ValueError(f"parts mix template rows with and without element-length stats: {n_tpl_stats}")
    if len(elem_defs) != 1:  # element sets / loop axes differ -> element tensors and alignment targets would not line up
        raise ValueError(f"parts disagree on (element_types, loop_axis): {sorted(elem_defs, key=str)}")
    element_types, loop_axis = elem_defs.pop()
    if len(gate_defs) != 1:
        raise ValueError(f"parts disagree on the template gate: {sorted(gate_defs)}")
    if len(src_defs) != 1:
        raise ValueError(f"parts disagree on the template source: {sorted(src_defs, key=str)}")
    if len(nat_defs) != 1:
        raise ValueError(f"parts disagree on the native source: {sorted(nat_defs, key=str)}")
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
        "dssp_def": dssp_defs.pop(),
        "reference_source": "synthetic",
        "element_types": element_types,
        "loop_axis": loop_axis,
        "template_gate": gate_defs.pop(),
    }
    template_source = src_defs.pop()
    if template_source is not None:
        out["template_source"] = template_source
    native_source = nat_defs.pop()
    if native_source is not None:
        out["native_source"] = native_source
    if elem_count:
        em = elem_sum / elem_count
        out["elem_feature_names"] = ["length"]
        out["elem_feature_mean"] = torch.tensor(em, dtype=torch.float32)
        out["elem_feature_std"] = torch.tensor(max(elem_sumsq / elem_count - em ** 2, 0.0) ** 0.5, dtype=torch.float32)
        print(f"  element length            mean={em:10.4f} std={float(out['elem_feature_std']):10.4f}", flush=True)
    if args.stats_from:
        # XRUN eval index (2026-10-28): a few-row index must standardise with the PRODUCTION index's statistics (the
        # ones the model trained on), not with stats of its own handful of rows
        prod = torch.load(args.stats_from, map_location="cpu", weights_only=False, mmap=True)
        assert list(prod["pair_feature_names"]) == list(PAIR_FEATURE_NAMES), "pair feature names differ from production"
        for k, dflt in (("contact_def", None), ("dssp_def", DSSP_DEF), ("element_types", (DSSP_HELIX, DSSP_STRAND)),
                        ("loop_axis", None), ("structural_feature_names", None), ("min_len", None)):
            if k not in prod and dflt is None:
                continue  # a key production predates (no default to compare against)
            got, want = out.get(k), prod.get(k, dflt)
            assert (list(got) if isinstance(got, (list, tuple)) else got) == \
                (list(want) if isinstance(want, (list, tuple)) else want), f"{k}: {got!r} != production {want!r}"
        # mirror production's stat keys: an index built before the element-length stats (v4) has none, and the
        # transform reads them only with elem_features on
        for k in ("elem_feature_names", "elem_feature_mean", "elem_feature_std"):
            out.pop(k, None)
        for k in ("pair_feature_mean", "pair_feature_std", "elem_feature_names", "elem_feature_mean", "elem_feature_std"):
            if k in prod:
                out[k] = prod[k].clone() if torch.is_tensor(prod[k]) else prod[k]
        out["stats_source"] = os.path.abspath(args.stats_from)
        print(f"  standardisation stats REPLACED by {args.stats_from}", flush=True)
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
    sel.add_argument("--base-index", default="", help="existing topology_index.pt; its ids define the chain list")
    sel.add_argument("--chain-list", default="", help="chain ids, one per line (instead of --base-index)")
    sel.add_argument("--template-pack", default="", help="t2 template pack root (TemplatePack; instead of --templates)")
    sel.add_argument("--max-templates", type=int, default=None, help="pack mode: templates per chain, spread over TM "
                                                                      "(default 64, user 2026-10-13: cap 64); not with "
                                                                      "--final-selection (no cap)")
    sel.add_argument("--final-selection", default="",
                     help="T8b: final_selection.npz of --template-pack; templates = that selection (TM window + clash "
                          "gate, no cap; user 2026-10-20 'Rebuild from final_selection')")
    sel.add_argument("--template-pack-tag", default="", help="final-selection mode: identity of --template-pack in "
                                                             "row ids and map paths")
    sel.add_argument("--override-pack", default="", help="final-selection mode: second pack (T8b regen) whose final "
                                                         "selection replaces the first for --override-chains")
    sel.add_argument("--override-selection", default="", help="final_selection.npz of --override-pack")
    sel.add_argument("--override-chains", default="", help="chains taking templates ONLY from --override-pack "
                                                           "(none there = no templates)")
    sel.add_argument("--override-pack-tag", default="", help="identity of --override-pack in row ids and map paths")
    sel.add_argument("--processed-dir", required=True)
    sel.add_argument("--manifest", default="", help="shard_manifest.json of the processed dir")
    sel.add_argument("--templates", action="append", default=[], help="ROOT:index_band.npz, repeatable")
    sel.add_argument("--chain-alias", default="",
                     help="TSV with label_id and auth_id columns. The template pool is keyed by "
                          "auth_asym_id and the dataset by label_asym_id; without this they join by "
                          "string and silently select a different polymer. Fails closed: a stem with "
                          "no alias entry gets no template.")
    sel.add_argument("--part", type=int, required=True)
    sel.add_argument("--n-parts", type=int, required=True)
    sel.add_argument("--min-len", type=int, default=1)
    sel.add_argument("--tm-min", type=float, default=None, help="build-time filter on template rows (default: keep the whole band)")
    sel.add_argument("--tm-max", type=float, default=None)
    sel.add_argument("--limit", type=int, default=0, help="debug: only the first N chains of the part")
    sel.add_argument("--selftest", action="store_true", help="also USalign every native to itself and record mismatches (debug)")
    b = sub.add_parser("build", parents=[sel])
    b.add_argument("--out-dir", required=True)
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--usalign", default=_USALIGN)
    b.add_argument("--contact-source", choices=("cb8", "confind"), default="cb8")
    b.add_argument("--native-pack", default="", help="read natives from this pack (+ .idx.npz) instead of --processed-dir")
    b.add_argument("--element-types", default=f"{DSSP_HELIX},{DSSP_STRAND}",
                   help="comma list of DSSP types that are reference elements (0 loop, 1 helix, 2 strand); default = v4")
    b.add_argument("--loop-axis", choices=LOOP_AXIS_MODES, default=None,
                   help="loop element axis when loops are elements (required then): zero | end_to_end")
    b.add_argument("--proline-donor-mask", action="store_true",
                   help="recompute native + template DSSP with proline as a non-donor (default: v4 behaviour)")
    b.add_argument("--f2c-maps", default="", help="maps root written by `f2c` (required for --contact-source confind)")
    b.add_argument("--f2c-batch-size", type=int, default=None,
                   help="final-selection confind: the f2c run's --batch-size; every maps file must carry it")
    b.add_argument("--f2c-checkpoint", default="~/Frame2ConFind/runs/f2s_v384_smoothap-pearson/best.pt",
                   help="final-selection confind: the f2c checkpoint (path string and md5 must match the maps files)")
    b.add_argument("--f2c-allow-reuse", action="store_true",
                   help="final-selection confind: accept maps files holding T8 copies (f2c --reuse-maps)")
    b.add_argument("--max-ca-clashes", type=int, default=None,
                   help="pack mode: drop templates with more than this many non-bonded CA-CA clashes, before the cap")
    b.add_argument("--ca-clash-a", type=float, default=None, help="clash cutoff in A (with --max-ca-clashes)")
    f = sub.add_parser("f2c", parents=[sel])
    f.add_argument("--maps-out", required=True)
    # the production "winner" that wrote every stored contact_map_confind (overwrite of 2026-06-03, verified 100%)
    f.add_argument("--checkpoint", default="~/Frame2ConFind/runs/f2s_v384_smoothap-pearson/best.pt")
    f.add_argument("--f2c-parent", default="~", help="directory containing the Frame2ConFind package")
    f.add_argument("--amp-dtype", default="bf16")
    f.add_argument("--batch-size", type=int, default=8, help="rungs per forward (winner rerun: 8 up to L 512)")
    f.add_argument("--max-len", type=int, default=F2C_MAX_LEN)
    f.add_argument("--native-check", type=int, default=0, help="compare F2C on N natives vs their stored maps first")
    f.add_argument("--native-batch", type=int, default=1, help="natives per forward in the check (backfill used 4)")
    f.add_argument("--reuse-maps", default="", help="final-selection mode: T8 maps root (pack-mode layout) to copy from")
    f.add_argument("--reuse-pack-idx", default="", help="the pre-append tmpl.idx.npz the --reuse-maps were computed from")
    fp = sub.add_parser("f2c-plan", parents=[sel])
    fp.add_argument("--maps-out", required=True, help="the maps root the f2c run will write (written chains are counted)")
    fp.add_argument("--plan-out", required=True)
    fp.add_argument("--checkpoint", default="~/Frame2ConFind/runs/f2s_v384_smoothap-pearson/best.pt")
    fp.add_argument("--max-len", type=int, default=F2C_MAX_LEN)
    fp.add_argument("--reuse-maps", default="")
    fp.add_argument("--reuse-pack-idx", default="")
    m = sub.add_parser("merge")
    m.add_argument("--parts-dir", required=True)
    m.add_argument("--n-parts", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--eligible-out", default="")
    m.add_argument("--tm-range", type=float, nargs=2, default=(0.5, 0.9))
    m.add_argument("--stats-from", default="", help="copy pair / element-length standardisation stats from this "
                                                    "(production) index instead of computing them from the parts")
    args = ap.parse_args()
    if hasattr(args, "tm_min"):  # pack mode requires them explicitly; the band-tree path keeps its 0.0-1.0 defaults
        args.tm_min_given, args.tm_max_given = args.tm_min, args.tm_max
        args.tm_min = 0.0 if args.tm_min is None else args.tm_min
        args.tm_max = 1.0 if args.tm_max is None else args.tm_max
    if hasattr(args, "max_templates"):
        if args.final_selection and args.max_templates is not None:
            raise SystemExit("--final-selection has no cap: drop --max-templates")
        if not args.final_selection:
            if any((args.override_pack, args.override_selection, args.override_chains)):
                raise SystemExit("--override-* need --final-selection")
            args.max_templates = 64 if args.max_templates is None else args.max_templates
    if getattr(args, "reuse_maps", "") and not args.final_selection:
        raise SystemExit("--reuse-maps applies to --final-selection mode")
    {"build": cmd_build, "f2c": cmd_f2c, "f2c-plan": cmd_f2c_plan, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
