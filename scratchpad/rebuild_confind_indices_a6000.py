#!/usr/bin/env python3
"""Regenerate ConFind synthetic topology indices with:
1. `tmalign_smooth` secondary structure classification (`topology_index_confind_synth_tmalign_smooth.pt`)
2. Fixed 9-residue ProteinMPNN chunks (`topology_index_confind_synth_mpnn_chunk9.pt`)

Reads the existing ConFind synthetic index (`topology_index_confind_synth.pt`), the native pack
(`tri_v2.pack`), and the template pool (`t2_pool_auth`), recomputing native and template secondary
structure with `tmalign_smooth` and projecting the ConFind contacts, structural pair features, and
USalign alignments onto both (a) `tmalign_smooth` SSE elements and (b) fixed 9-residue chunks.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import mmap
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proteinfoundation.datasets.sse_topology import (  # noqa: E402
    DSSP_HELIX,
    DSSP_STRAND,
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    STRUCTURAL_PAIR_FEATURES,
    _element_axes,
    assemble_pair_features,
    dssp_to_runs,
    element_lengths,
    runs_to_spans,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.nn.mpnn_topology_encoder import CHUNK_SIZE, chunk9_runs  # noqa: E402
from proteinfoundation.utils.dssp_utils import compute_tmalign_dssp, smooth_tmalign_dssp  # noqa: E402

ALIGN_NONE = -1
CONFIND_THRESHOLD = 0.01

_SRC_IDX = None
_PACK_IDX = None
_PACK_MM = None
_PACK_FH = None
_ALIAS = None
_POOL_DIR = ""
_MIN_LEN = 1


def _recompute_orientation_cos(
    ca: torch.Tensor,
    ca_mask: torch.Tensor,
    runs,
    keep,
) -> torch.Tensor:
    """Compute [T, T] orientation cosine matrix directly from CA coordinates and new runs/keep."""
    T = len(keep)
    if T == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    spans = runs_to_spans(runs)
    L = int(ca.shape[0])
    elem = torch.full((L,), -1, dtype=torch.long)
    for a, ia in enumerate(keep):
        s, e = spans[ia]
        elem[s:e] = a
    valid = elem >= 0
    ca_ok = valid & ca_mask.bool()
    if not ca_ok.any():
        return torch.zeros((T, T), dtype=torch.float32)
    axes = _element_axes(ca.float(), elem, ca_ok, T)
    cos = (axes @ axes.T).clamp(-1.0, 1.0)
    return 0.5 * (cos + cos.transpose(0, 1))


def _project_element_features_and_align(
    old_runs,
    old_keep,
    new_runs,
    new_keep,
    he_contact: torch.Tensor,
    structural: torch.Tensor,
    align_old: torch.Tensor,
    require_same_type: bool = True,
):
    """Project (he_contact, structural, align_old) from (old_runs, old_keep) onto (new_runs, new_keep)
    using exact interval overlap matrix O[a, b] = max(0, min(e_a, e_b) - max(s_a, s_b))."""
    T_old, T_new = len(old_keep), len(new_keep)
    n_struct = len(STRUCTURAL_PAIR_FEATURES)

    if T_new == 0:
        return (
            torch.zeros(0, 0, dtype=torch.uint8),
            torch.zeros(0, 0, n_struct, dtype=torch.float16),
            torch.full_like(align_old, ALIGN_NONE, dtype=torch.int16),
        )
    if T_old == 0:
        return (
            torch.zeros(T_new, T_new, dtype=torch.uint8),
            torch.zeros(T_new, T_new, n_struct, dtype=torch.float16),
            torch.full_like(align_old, ALIGN_NONE, dtype=torch.int16),
        )

    old_spans = runs_to_spans(old_runs)
    new_spans = runs_to_spans(new_runs)

    new_s = torch.tensor([new_spans[i][0] for i in new_keep], dtype=torch.long)
    new_e = torch.tensor([new_spans[i][1] for i in new_keep], dtype=torch.long)
    new_t = torch.tensor([int(new_runs[i][0]) for i in new_keep], dtype=torch.long)

    old_s = torch.tensor([old_spans[i][0] for i in old_keep], dtype=torch.long)
    old_e = torch.tensor([old_spans[i][1] for i in old_keep], dtype=torch.long)
    old_t = torch.tensor([int(old_runs[i][0]) for i in old_keep], dtype=torch.long)

    # [T_new, T_old] residue overlap counts between new elements and old elements
    O = (torch.minimum(new_e[:, None], old_e[None, :]) - torch.maximum(new_s[:, None], old_s[None, :])).clamp(min=0).float()

    max_ov, best_new = O.max(dim=0)
    if require_same_type:
        valid_map = (max_ov > 0) & (new_t[best_new] == old_t)
    else:
        valid_map = max_ov > 0
    old_to_new_elem = torch.where(valid_map, best_new, torch.full_like(best_new, ALIGN_NONE))

    align_new = torch.full_like(align_old, ALIGN_NONE, dtype=torch.int16)
    if align_old.numel() > 0:
        valid_al = (align_old >= 0) & (align_old < T_old)
        align_new[valid_al] = old_to_new_elem[align_old[valid_al].long()].to(torch.int16)

    if he_contact.numel() == 0:
        return (
            torch.zeros(T_new, T_new, dtype=torch.uint8),
            torch.zeros(T_new, T_new, n_struct, dtype=torch.float16),
            align_new,
        )

    if require_same_type:
        same_t = (new_t[:, None] == old_t[None, :]).float()
        W_same = O * same_t
        W = torch.where(W_same.sum(dim=-1, keepdim=True) > 0, W_same, O)
    else:
        W = O

    row_sum = W.sum(dim=-1, keepdim=True)
    has_overlap = row_sum.squeeze(-1) > 0
    W_norm = W / row_sum.clamp(min=1.0)
    M = (W > 0).float()

    new_contact = ((M @ he_contact.float() @ M.T) > 0).float()
    new_structural = torch.einsum("ai,ijk,bj->abk", W_norm, structural.float(), W_norm)

    diag_idx = torch.arange(T_new)
    if float(he_contact.diagonal().max().item()) > 0.5:
        new_contact[diag_idx[has_overlap], diag_idx[has_overlap]] = 1.0
    if n_struct >= 2:
        new_structural[diag_idx, diag_idx, 0] = 0.0
        new_structural[diag_idx[has_overlap], diag_idx[has_overlap], 1] = 1.0

    new_structural = 0.5 * (new_structural + new_structural.transpose(0, 1))
    return (
        new_contact.to(torch.uint8),
        new_structural.to(torch.float16),
        align_new,
    )


def _stats_for_row(
    contact_u8: torch.Tensor,
    structural_f16: torch.Tensor,
    runs,
    keep,
):
    """Compute standardization accumulators (feat_sum, feat_sumsq, feat_cnt, elem_sum, elem_sumsq, elem_cnt)."""
    T = len(keep)
    if T == 0:
        return (
            torch.zeros(N_PAIR_FEATURES, dtype=torch.float64),
            torch.zeros(N_PAIR_FEATURES, dtype=torch.float64),
            0,
            0.0,
            0.0,
            0,
        )
    feat = assemble_pair_features(contact_u8.float(), structural_f16.float(), runs, keep)
    flat = feat.reshape(-1, N_PAIR_FEATURES).to(torch.float64)
    el = element_lengths(runs, keep).to(torch.float64)
    return (
        flat.sum(0),
        (flat ** 2).sum(0),
        int(flat.shape[0]),
        float(el.sum().item()),
        float((el ** 2).sum().item()),
        int(el.numel()),
    )


def _get_old_row(idx, r: int):
    ra, rb = int(idx["runs_offset"][r]), int(idx["runs_offset"][r + 1])
    old_runs = [(int(t), int(n)) for t, n in idx["runs_flat"][ra:rb].tolist()] if rb > ra else []
    old_keep = [i for i, (t, _) in enumerate(old_runs) if t in (DSSP_HELIX, DSSP_STRAND)]
    T_old = int(idx["he_size"][r])
    ha, hb = int(idx["he_offset"][r]), int(idx["he_offset"][r + 1])
    he_old = idx["he_flat"][ha:hb].reshape(T_old, T_old).float() if (T_old > 0 and hb > ha) else torch.zeros(0, 0)
    fa, fb = int(idx["feat_offset"][r]), int(idx["feat_offset"][r + 1])
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    struct_old = (
        idx["feat_flat"][fa:fb].reshape(T_old, T_old, n_struct).float()
        if (T_old > 0 and fb > fa)
        else torch.zeros(0, 0, n_struct)
    )
    aa, ab = int(idx["align_offset"][r]), int(idx["align_offset"][r + 1])
    align_old = idx["align_flat"][aa:ab].to(torch.int16) if ab > aa else torch.zeros(0, dtype=torch.int16)
    return old_runs, old_keep, he_old, struct_old, align_old


def _runs_to_dssp_tensor(runs) -> torch.Tensor:
    if not runs:
        return torch.zeros(0, dtype=torch.long)
    parts = [torch.full((int(n),), int(t), dtype=torch.long) for t, n in runs if int(n) > 0]
    return torch.cat(parts, dim=0) if parts else torch.zeros(0, dtype=torch.long)


def _process_row_range(row_range):
    """Process a contiguous range of rows [r_start, r_end) from _SRC_IDX."""
    r_start, r_end = row_range
    idx = _SRC_IDX
    ids = idx["ids"]
    is_native_arr = idx["row_is_native"]

    # Smooth accumulators
    s_runs_flat = []
    s_runs_len = []
    s_he_size = []
    s_he_flat = []
    s_feat_flat = []
    s_align_flat = []
    s_feat_sum = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    s_feat_sumsq = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    s_feat_cnt = 0
    s_elem_sum = 0.0
    s_elem_sumsq = 0.0
    s_elem_cnt = 0

    # Chunk9 accumulators
    c_runs_flat = []
    c_runs_len = []
    c_he_size = []
    c_he_flat = []
    c_feat_flat = []
    c_align_flat = []
    c_feat_sum = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    c_feat_sumsq = torch.zeros(N_PAIR_FEATURES, dtype=torch.float64)
    c_feat_cnt = 0
    c_elem_sum = 0.0
    c_elem_sumsq = 0.0
    c_elem_cnt = 0

    cached_stem = None
    cached_dssp_all = None
    cached_L_t = None

    for r in range(r_start, r_end):
        row_id = str(ids[r])
        is_nat = bool(is_native_arr[r].item())
        old_runs, old_keep, he_old, struct_old, align_old = _get_old_row(idx, r)
        L_old = sum(n for _, n in old_runs)

        if is_nat:
            stem = row_id
            g = None
            if _PACK_IDX is not None and stem in _PACK_IDX:
                off, ln = _PACK_IDX[stem]
                try:
                    g = torch.load(io.BytesIO(_PACK_MM[off : off + ln]), map_location="cpu", weights_only=False)
                except Exception:
                    g = None

            if g is not None and getattr(g, "coords", None) is not None and getattr(g, "coord_mask", None) is not None:
                coords = g.coords.float()
                cmask = (g.coord_mask.float() > 0.5)
                L_n = int(coords.shape[0])
                ca = coords[:, 1, :]
                ca_mask = cmask[:, 1]
                dssp_s = compute_tmalign_dssp(ca[None], ca_mask[None], smooth=True)[0]
                runs_s = dssp_to_runs(dssp_s, min_len=_MIN_LEN)
                keep_s = [i for i, (t, _) in enumerate(runs_s) if t in (DSSP_HELIX, DSSP_STRAND)]

                runs_c = chunk9_runs(L_n, chunk_size=CHUNK_SIZE)
                keep_c = list(range(len(runs_c)))

                ref_s_u8, str_s_f16, al_s = _project_element_features_and_align(
                    old_runs, old_keep, runs_s, keep_s, he_old, struct_old, align_old,
                    require_same_type=True,
                )
                ref_c_u8, str_c_f16, al_c = _project_element_features_and_align(
                    old_runs, old_keep, runs_c, keep_c, he_old, struct_old, align_old,
                    require_same_type=False,
                )
            else:
                dssp_old = _runs_to_dssp_tensor(old_runs)
                dssp_s = smooth_tmalign_dssp(dssp_old[None])[0] if dssp_old.numel() > 0 else dssp_old
                runs_s = dssp_to_runs(dssp_s, min_len=_MIN_LEN) if dssp_s.numel() > 0 else []
                keep_s = [i for i, (t, _) in enumerate(runs_s) if t in (DSSP_HELIX, DSSP_STRAND)]
                ref_s_u8, str_s_f16, al_s = _project_element_features_and_align(
                    old_runs, old_keep, runs_s, keep_s, he_old, struct_old, align_old, require_same_type=True
                )

                runs_c = chunk9_runs(L_old, chunk_size=CHUNK_SIZE)
                keep_c = list(range(len(runs_c)))
                ref_c_u8, str_c_f16, al_c = _project_element_features_and_align(
                    old_runs, old_keep, runs_c, keep_c, he_old, struct_old, align_old, require_same_type=False
                )
        else:
            # Template row: stem@rw{rewind}#{k}
            dssp_s = None
            L_t = L_old
            if "@" in row_id and "#" in row_id and _POOL_DIR:
                stem, rest = row_id.split("@", 1)
                k = int(rest.split("#", 1)[1])
                if stem != cached_stem:
                    cached_stem = stem
                    cached_dssp_all = None
                    cached_L_t = None
                    auth_stem = _ALIAS.get(stem, stem) if _ALIAS is not None else stem
                    npz_path = os.path.join(
                        _POOL_DIR,
                        f"shard{zlib.crc32(auth_stem.encode()) % 1000:04d}",
                        f"{auth_stem}.npz",
                    )
                    if os.path.exists(npz_path):
                        z = np.load(npz_path)
                        amask_np = z["atom_mask"].astype(bool)
                        tcoords_np = z["coords"]
                        L_npz = int(amask_np.shape[0])
                        n_rungs = int(tcoords_np.shape[0])
                        ca_present = amask_np[:, 1]
                        ca_packed_idx = np.cumsum(amask_np.reshape(-1), dtype=np.int64).reshape(L_npz, 37)[:, 1] - 1
                        ca_all = torch.zeros((n_rungs, L_npz, 3), dtype=torch.float32)
                        if ca_present.any():
                            ca_all[:, ca_present, :] = torch.from_numpy(
                                tcoords_np[:, ca_packed_idx[ca_present], :]
                            ).float()
                        ca_mask_all = torch.from_numpy(ca_present)[None].expand(n_rungs, L_npz)
                        cached_dssp_all = compute_tmalign_dssp(ca_all, ca_mask_all, smooth=True)
                        cached_L_t = L_npz
                if cached_dssp_all is not None and 0 <= k < int(cached_dssp_all.shape[0]):
                    dssp_s = cached_dssp_all[k]
                    L_t = cached_L_t

            if dssp_s is not None:
                runs_s = dssp_to_runs(dssp_s, min_len=_MIN_LEN)
            else:
                dssp_old = _runs_to_dssp_tensor(old_runs)
                dssp_s = smooth_tmalign_dssp(dssp_old[None])[0] if dssp_old.numel() > 0 else dssp_old
                runs_s = dssp_to_runs(dssp_s, min_len=_MIN_LEN) if dssp_s.numel() > 0 else []

            keep_s = [i for i, (t, _) in enumerate(runs_s) if t in (DSSP_HELIX, DSSP_STRAND)]
            ref_s_u8, str_s_f16, al_s = _project_element_features_and_align(
                old_runs, old_keep, runs_s, keep_s, he_old, struct_old, align_old,
                require_same_type=True,
            )

            runs_c = chunk9_runs(L_t, chunk_size=CHUNK_SIZE)
            keep_c = list(range(len(runs_c)))
            ref_c_u8, str_c_f16, al_c = _project_element_features_and_align(
                old_runs, old_keep, runs_c, keep_c, he_old, struct_old, align_old,
                require_same_type=False,
            )

            fs, fsq, fcnt, es, esq, ecnt = _stats_for_row(ref_s_u8, str_s_f16, runs_s, keep_s)
            s_feat_sum.add_(fs)
            s_feat_sumsq.add_(fsq)
            s_feat_cnt += fcnt
            s_elem_sum += es
            s_elem_sumsq += esq
            s_elem_cnt += ecnt

            fs_c, fsq_c, fcnt_c, es_c, esq_c, ecnt_c = _stats_for_row(ref_c_u8, str_c_f16, runs_c, keep_c)
            c_feat_sum.add_(fs_c)
            c_feat_sumsq.add_(fsq_c)
            c_feat_cnt += fcnt_c
            c_elem_sum += es_c
            c_elem_sumsq += esq_c
            c_elem_cnt += ecnt_c

        # Append smooth row
        s_runs_flat.extend(runs_s)
        s_runs_len.append(len(runs_s))
        s_he_size.append(len(keep_s))
        s_he_flat.append(ref_s_u8.reshape(-1))
        s_feat_flat.append(str_s_f16.reshape(-1))
        s_align_flat.append(al_s.reshape(-1))

        # Append chunk9 row
        c_runs_flat.extend(runs_c)
        c_runs_len.append(len(runs_c))
        c_he_size.append(len(keep_c))
        c_he_flat.append(ref_c_u8.reshape(-1))
        c_feat_flat.append(str_c_f16.reshape(-1))
        c_align_flat.append(al_c.reshape(-1))

    smooth_chunk = {
        "runs_flat": np.asarray(s_runs_flat, dtype=np.int16) if s_runs_flat else np.zeros((0, 2), dtype=np.int16),
        "runs_len": np.asarray(s_runs_len, dtype=np.int64),
        "he_size": np.asarray(s_he_size, dtype=np.int16),
        "he_flat": torch.cat(s_he_flat).numpy() if s_he_flat else np.zeros(0, dtype=np.uint8),
        "feat_flat": torch.cat(s_feat_flat).numpy() if s_feat_flat else np.zeros(0, dtype=np.float16),
        "align_flat": torch.cat(s_align_flat).numpy() if s_align_flat else np.zeros(0, dtype=np.int16),
        "feat_sum": s_feat_sum.numpy(),
        "feat_sumsq": s_feat_sumsq.numpy(),
        "feat_cnt": int(s_feat_cnt),
        "elem_sum": float(s_elem_sum),
        "elem_sumsq": float(s_elem_sumsq),
        "elem_cnt": int(s_elem_cnt),
    }
    chunk9_chunk = {
        "runs_flat": np.asarray(c_runs_flat, dtype=np.int16) if c_runs_flat else np.zeros((0, 2), dtype=np.int16),
        "runs_len": np.asarray(c_runs_len, dtype=np.int64),
        "he_size": np.asarray(c_he_size, dtype=np.int16),
        "he_flat": torch.cat(c_he_flat).numpy() if c_he_flat else np.zeros(0, dtype=np.uint8),
        "feat_flat": torch.cat(c_feat_flat).numpy() if c_feat_flat else np.zeros(0, dtype=np.float16),
        "align_flat": torch.cat(c_align_flat).numpy() if c_align_flat else np.zeros(0, dtype=np.int16),
        "feat_sum": c_feat_sum.numpy(),
        "feat_sumsq": c_feat_sumsq.numpy(),
        "feat_cnt": int(c_feat_cnt),
        "elem_sum": float(c_elem_sum),
        "elem_sumsq": float(c_elem_sumsq),
        "elem_cnt": int(c_elem_cnt),
    }
    return smooth_chunk, chunk9_chunk


def _assemble_index_from_chunks(src_idx, chunks, dssp_def: str, element_types):
    runs_len = torch.from_numpy(np.concatenate([c["runs_len"] for c in chunks]))
    runs_offset = torch.zeros(runs_len.numel() + 1, dtype=torch.int64)
    runs_offset[1:] = torch.cumsum(runs_len, dim=0)

    he_size = torch.from_numpy(np.concatenate([c["he_size"] for c in chunks]))
    he_counts = he_size.to(torch.int64) ** 2
    he_offset = torch.zeros(he_size.numel() + 1, dtype=torch.int64)
    he_offset[1:] = torch.cumsum(he_counts, dim=0)

    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    feat_counts = he_counts * n_struct
    feat_offset = torch.zeros(he_size.numel() + 1, dtype=torch.int64)
    feat_offset[1:] = torch.cumsum(feat_counts, dim=0)

    runs_flat = torch.from_numpy(np.concatenate([c["runs_flat"] for c in chunks], axis=0))
    he_flat = torch.from_numpy(np.concatenate([c["he_flat"] for c in chunks], axis=0))
    feat_flat = torch.from_numpy(np.concatenate([c["feat_flat"] for c in chunks], axis=0))
    align_flat = torch.from_numpy(np.concatenate([c["align_flat"] for c in chunks], axis=0))

    feat_sum = sum((torch.from_numpy(c["feat_sum"]) for c in chunks), torch.zeros(N_PAIR_FEATURES, dtype=torch.float64))
    feat_sumsq = sum((torch.from_numpy(c["feat_sumsq"]) for c in chunks), torch.zeros(N_PAIR_FEATURES, dtype=torch.float64))
    feat_cnt = sum(c["feat_cnt"] for c in chunks)
    elem_sum = sum(c["elem_sum"] for c in chunks)
    elem_sumsq = sum(c["elem_sumsq"] for c in chunks)
    elem_cnt = sum(c["elem_cnt"] for c in chunks)

    mean = feat_sum / max(feat_cnt, 1)
    std = (feat_sumsq / max(feat_cnt, 1) - mean ** 2).clamp(min=0.0).sqrt().clamp(min=1e-6)

    out = dict(src_idx)
    out["runs_flat"] = runs_flat
    out["runs_offset"] = runs_offset
    out["he_size"] = he_size
    out["he_offset"] = he_offset
    out["he_flat"] = he_flat
    out["feat_offset"] = feat_offset
    out["feat_flat"] = feat_flat
    out["align_flat"] = align_flat
    out["pair_feature_mean"] = mean.to(torch.float32)
    out["pair_feature_std"] = std.to(torch.float32)
    out["dssp_def"] = dssp_def
    out["element_types"] = tuple(element_types)
    if elem_cnt > 0:
        em = elem_sum / elem_cnt
        out["elem_feature_names"] = ["length"]
        out["elem_feature_mean"] = torch.tensor(em, dtype=torch.float32)
        out["elem_feature_std"] = torch.tensor(max(elem_sumsq / elem_cnt - em ** 2, 0.0) ** 0.5, dtype=torch.float32)
    return out


def main():
    global _SRC_IDX, _PACK_IDX, _PACK_MM, _PACK_FH, _ALIAS, _POOL_DIR, _MIN_LEN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-index", required=True, help="Path to source topology_index_confind_synth.pt")
    ap.add_argument("--src-eligible", default="", help="Path to source eligible_ids_tm0.5-0.9.txt")
    ap.add_argument("--native-pack", required=True, help="Path to tri_v2.pack")
    ap.add_argument("--template-pool", required=True, help="Path to t2_pool_auth directory")
    ap.add_argument("--chain-alias", required=True, help="Path to auth_key.tsv")
    ap.add_argument("--out-smooth", required=True, help="Output path for topology_index_confind_synth_tmalign_smooth.pt")
    ap.add_argument("--out-chunk9", required=True, help="Output path for topology_index_confind_synth_mpnn_chunk9.pt")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--num-chunks", type=int, default=128)
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"Loading source index {args.src_index} (mmap=True)...", flush=True)
    _SRC_IDX = torch.load(args.src_index, map_location="cpu", weights_only=False, mmap=True)
    _MIN_LEN = int(_SRC_IDX.get("min_len", 1))
    _POOL_DIR = args.template_pool

    if os.path.exists(args.chain_alias):
        alias = {}
        with open(args.chain_alias) as fh:
            head = fh.readline().rstrip("\n").split("\t")
            il, ia = head.index("label_id"), head.index("auth_id")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) > max(il, ia) and f[ia]:
                    alias[f[il]] = f[ia]
        _ALIAS = alias
        print(f"Loaded {len(_ALIAS)} chain aliases from {args.chain_alias}", flush=True)

    if os.path.exists(args.native_pack) and os.path.exists(args.native_pack + ".idx.npz"):
        z = np.load(args.native_pack + ".idx.npz", allow_pickle=True)
        _PACK_IDX = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
        _PACK_FH = open(args.native_pack, "rb")
        _PACK_MM = mmap.mmap(_PACK_FH.fileno(), 0, access=mmap.ACCESS_READ)
        print(f"Loaded native pack {args.native_pack} with {len(_PACK_IDX)} stems", flush=True)

    n_rows = len(_SRC_IDX["ids"])
    # Partition by cluster boundaries so all template rows of a chain stay in the same worker chunk
    members_offset = _SRC_IDX["members_offset"]
    n_clusters = int(members_offset.numel()) - 1
    cluster_of = _SRC_IDX["cluster_of"].numpy()
    # Find start row of each cluster
    _, cluster_start_rows = np.unique(cluster_of, return_index=True)
    cluster_start_rows = list(map(int, cluster_start_rows)) + [n_rows]

    step = max(1, n_clusters // max(args.num_chunks, 1))
    row_ranges = []
    for c0 in range(0, n_clusters, step):
        c1 = min(n_clusters, c0 + step)
        row_ranges.append((cluster_start_rows[c0], cluster_start_rows[c1]))

    print(
        f"Processing {n_rows} rows across {n_clusters} clusters in {len(row_ranges)} chunks "
        f"using {args.workers} workers...",
        flush=True,
    )

    smooth_chunks = []
    chunk9_chunks = []
    if args.workers <= 1:
        for i, rr in enumerate(row_ranges):
            sc, cc = _process_row_range(rr)
            smooth_chunks.append(sc)
            chunk9_chunks.append(cc)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for i, (sc, cc) in enumerate(ex.map(_process_row_range, row_ranges)):
                smooth_chunks.append(sc)
                chunk9_chunks.append(cc)
                if (i + 1) % 16 == 0 or (i + 1) == len(row_ranges):
                    print(f"  completed {i + 1}/{len(row_ranges)} chunks ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Assembling tmalign_smooth index...", flush=True)
    out_smooth = _assemble_index_from_chunks(
        _SRC_IDX, smooth_chunks, dssp_def="tmalign_smooth", element_types=(DSSP_HELIX, DSSP_STRAND)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_smooth)), exist_ok=True)
    torch.save(out_smooth, args.out_smooth)
    print(
        f"Saved {args.out_smooth} (pair_feature_mean={out_smooth['pair_feature_mean'].tolist()})",
        flush=True,
    )
    del out_smooth, smooth_chunks

    print("Assembling mpnn_chunk9 index...", flush=True)
    out_chunk9 = _assemble_index_from_chunks(
        _SRC_IDX, chunk9_chunks, dssp_def="mpnn_chunk9", element_types=(DSSP_HELIX,)
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_chunk9)), exist_ok=True)
    torch.save(out_chunk9, args.out_chunk9)
    print(
        f"Saved {args.out_chunk9} (pair_feature_mean={out_chunk9['pair_feature_mean'].tolist()})",
        flush=True,
    )
    del out_chunk9, chunk9_chunks

    if args.src_eligible and os.path.exists(args.src_eligible):
        for dst_idx in (args.out_smooth, args.out_chunk9):
            dst_elig = os.path.join(os.path.dirname(os.path.abspath(dst_idx)), "eligible_ids_tm0.5-0.9.txt")
            if os.path.abspath(args.src_eligible) != os.path.abspath(dst_elig):
                shutil.copy2(args.src_eligible, dst_elig)
                print(f"Copied eligible IDs -> {dst_elig}", flush=True)

    print(f"REBUILD_INDICES_DONE in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
