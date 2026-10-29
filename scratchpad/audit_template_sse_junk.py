#!/usr/bin/env python3
"""Audit non-consecutive / fragmented SSE classifications in the existing template training data,
and measure how much the TMalign/USalign CA-based SSE classifier (raw & smooth) fixes them.

Two complementary audits:
1. Full-index run-length audit across ALL rows in `topology_index_confind_synth.pt` (70,404 native chains
   + 1,656,113 synthetic template rows, plus the training-eligible subset in TM [0.5, 0.9]):
   - Distribution of H/E run lengths (specifically 1-residue H/E, <=2-residue H/E, and canonical
     impossible / non-consecutive fragments: Helix len < 4 or Strand len < 2).
   - Frequency of interrupted / fragmented elements (`H - L(1..2) - H` and `E - L(1..2) - E`) where a
     1-2 residue hydrogen-bond dropout splits a continuous secondary structure element into two separate
     tokens + a spurious loop.
   - Comparison of what `tmalign_smooth`'s morphological smoothing pass (`HEH -> HHH`, `EHE -> EEE`,
     `H L H -> H H H`, `E L E -> E E E` at residue level) plus canonical minimum-length gating
     (`H >= 4`, `E >= 2`) eliminates directly on the index runs.
2. Direct coordinate-level re-featurization audit on a stratified sample of real synthetic template 3D
   structures from `/home/gridsan/cou/pp1c_work/templates_band` (paired with their native chains):
   - Computes exact per-residue DSSP on both native and synthetic templates under:
     (a) `h_bond` (current PyDSSP + proline donor mask),
     (b) `tmalign` (USalign `sec_str` + `make_sec`),
     (c) `tmalign_smooth` (`tmalign` + 2-pass morphological smoothing + min-len H>=3, E>=2).
   - Quantifies:
     - Junk / non-consecutive H/E run rate (`len == 1`, `len <= 2`, `H < 4 or E < 2`)
     - Split-element rate (`X - L(1..2) - X` for `X in {H, E}`)
     - Per-residue SSE agreement (Q3 accuracy and H/E F1) of the noisy synthetic template against the
       CLEAN NATIVE structure's SSE assignment!
     - Element-count inflation / deflation (`|n_el(template) - n_el(native)|`).
"""

import argparse
import glob
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from proteinfoundation.datasets.sse_topology import DSSP_HELIX, DSSP_LOOP, DSSP_STRAND, dssp_to_runs
from proteinfoundation.utils.dssp_utils import compute_dssp_target, smooth_tmalign_dssp

PRO_IDX = 14
CA_IDX = 1


def analyze_runs_subset(runs_flat: np.ndarray, runs_off: np.ndarray, row_indices: np.ndarray, label: str):
    """Vectorized + fast per-row analysis of SSE runs in `topology_index_confind_synth.pt`."""
    t0 = time.time()
    lens = runs_off[row_indices + 1] - runs_off[row_indices]
    valid_mask = lens > 0
    valid_rows = row_indices[valid_mask]
    n_rows = len(valid_rows)

    # Extract slices for valid_rows
    starts = runs_off[valid_rows]
    ends = runs_off[valid_rows + 1]
    lengths = ends - starts

    # Build flat array of runs for these rows
    total_runs = int(lengths.sum())
    idx_flat = np.empty(total_runs, dtype=np.int64)
    pos = 0
    # Fast chunked or numba-free slice gather
    row_of_run = np.repeat(np.arange(n_rows, dtype=np.int32), lengths)
    offsets_within = np.arange(total_runs, dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    idx_flat = np.repeat(starts, lengths) + offsets_within

    sub_runs = runs_flat[idx_flat]
    t = sub_runs[:, 0]
    n = sub_runs[:, 1]

    he_m = (t == DSSP_HELIX) | (t == DSSP_STRAND)
    h_m = t == DSSP_HELIX
    e_m = t == DSSP_STRAND
    l_m = t == DSSP_LOOP

    total_he = int(he_m.sum())
    total_h = int(h_m.sum())
    total_e = int(e_m.sum())

    h_len1 = int((h_m & (n == 1)).sum())
    h_len2 = int((h_m & (n == 2)).sum())
    h_len3 = int((h_m & (n == 3)).sum())
    h_lt3 = h_len1 + h_len2
    h_lt4 = h_lt3 + h_len3

    e_len1 = int((e_m & (n == 1)).sum())
    e_len2 = int((e_m & (n == 2)).sum())

    he_len1 = h_len1 + e_len1
    he_le2 = h_lt3 + e_len1 + e_len2
    canon_junk_tmalign = h_lt3 + e_len1  # H < 3 or E < 2 (impossible even in TMalign / 3_10)
    canon_junk_alpha = h_lt4 + e_len1    # H < 4 or E < 2 (shorter than 1 alpha-helical turn or 1 beta-bridge)

    # Detect split elements: X - L(1..2) - X within the same row
    same_row_next2 = (row_of_run[:-2] == row_of_run[2:])
    t0_arr, t1_arr, t2_arr = t[:-2], t[1:-1], t[2:]
    n1_arr = n[1:-1]
    split_h_l1 = same_row_next2 & (t0_arr == DSSP_HELIX) & (t1_arr == DSSP_LOOP) & (n1_arr == 1) & (t2_arr == DSSP_HELIX)
    split_h_l2 = same_row_next2 & (t0_arr == DSSP_HELIX) & (t1_arr == DSSP_LOOP) & (n1_arr <= 2) & (t2_arr == DSSP_HELIX)
    split_e_l1 = same_row_next2 & (t0_arr == DSSP_STRAND) & (t1_arr == DSSP_LOOP) & (n1_arr == 1) & (t2_arr == DSSP_STRAND)
    split_e_l2 = same_row_next2 & (t0_arr == DSSP_STRAND) & (t1_arr == DSSP_LOOP) & (n1_arr <= 2) & (t2_arr == DSSP_STRAND)
    # Also X - Y(1) - X (e.g. H - E(1) - H or E - H(1) - E)
    split_he_1 = same_row_next2 & ((t0_arr == DSSP_HELIX) | (t0_arr == DSSP_STRAND)) & (t0_arr == t2_arr) & (n1_arr == 1)
    split_he_le2 = same_row_next2 & ((t0_arr == DSSP_HELIX) | (t0_arr == DSSP_STRAND)) & (t0_arr == t2_arr) & (n1_arr <= 2)

    n_split_h_l1 = int(split_h_l1.sum())
    n_split_h_l2 = int(split_h_l2.sum())
    n_split_e_l1 = int(split_e_l1.sum())
    n_split_e_l2 = int(split_e_l2.sum())
    n_split_any_1 = int(split_he_1.sum())
    n_split_any_le2 = int(split_he_le2.sum())

    # Per-row aggregations
    rr_he = row_of_run[he_m]
    he_per_row = np.bincount(rr_he, minlength=n_rows)
    len1_per_row = np.bincount(row_of_run[he_m & (n == 1)], minlength=n_rows)
    tm_junk_per_row = np.bincount(row_of_run[(h_m & (n < 3)) | (e_m & (n < 2))], minlength=n_rows)
    alpha_junk_per_row = np.bincount(row_of_run[(h_m & (n < 4)) | (e_m & (n < 2))], minlength=n_rows)
    split1_per_row = np.bincount(row_of_run[:-2][split_he_1], minlength=n_rows)
    splitle2_per_row = np.bincount(row_of_run[:-2][split_he_le2], minlength=n_rows)

    print(f"\n================================================================================")
    print(f"INDEX AUDIT: {label} (n_rows={n_rows:,}, elapsed={time.time()-t0:.2f}s)")
    print(f"================================================================================")
    print(f"  Total H/E elements: {total_he:,} (mean {total_he/max(n_rows,1):.2f} / chain) | H={total_h:,}, E={total_e:,}")
    print(f"  1. Element Length Breakdown:")
    print(f"     - Helices (total {total_h:,}):")
    print(f"         len == 1 : {h_len1:9,d} ({100*h_len1/max(total_h,1):5.2f}% of H)")
    print(f"         len == 2 : {h_len2:9,d} ({100*h_len2/max(total_h,1):5.2f}% of H)")
    print(f"         len == 3 : {h_len3:9,d} ({100*h_len3/max(total_h,1):5.2f}% of H) [3_10 or clipped turn]")
    print(f"         len <  3 : {h_lt3:9,d} ({100*h_lt3/max(total_h,1):5.2f}% of H) [impossible helix fragment]")
    print(f"         len <  4 : {h_lt4:9,d} ({100*h_lt4/max(total_h,1):5.2f}% of H) [< 1 alpha-helical turn]")
    print(f"     - Strands (total {total_e:,}):")
    print(f"         len == 1 : {e_len1:9,d} ({100*e_len1/max(total_e,1):5.2f}% of E) [isolated beta-bridge B / 1-res E]")
    print(f"         len == 2 : {e_len2:9,d} ({100*e_len2/max(total_e,1):5.2f}% of E)")
    print(f"     - Combined H/E (total {total_he:,}):")
    print(f"         len == 1 (1-res H or E)      : {he_len1:9,d} ({100*he_len1/max(total_he,1):5.2f}% of all H/E)")
    print(f"         H<3 or E<2 (impossible frag) : {canon_junk_tmalign:9,d} ({100*canon_junk_tmalign/max(total_he,1):5.2f}% of all H/E)")
    print(f"         H<4 or E<2 (sub-turn/bridge) : {canon_junk_alpha:9,d} ({100*canon_junk_alpha/max(total_he,1):5.2f}% of all H/E)")
    print(f"  2. Interrupted / Split Elements (X - gap - X within same chain):")
    print(f"     - H - L(1) - H   (1-res loop puncture in helix)  : {n_split_h_l1:8,d} ({100*n_split_h_l1/max(total_h,1):5.2f}% of H)")
    print(f"     - H - L(<=2) - H (<=2-res loop puncture in helix): {n_split_h_l2:8,d} ({100*n_split_h_l2/max(total_h,1):5.2f}% of H)")
    print(f"     - E - L(1) - E   (1-res loop puncture in strand) : {n_split_e_l1:8,d} ({100*n_split_e_l1/max(total_e,1):5.2f}% of E)")
    print(f"     - E - L(<=2) - E (<=2-res loop puncture in strand): {n_split_e_l2:8,d} ({100*n_split_e_l2/max(total_e,1):5.2f}% of E)")
    print(f"     - Any X - *(1) - X   (healed by 1-res smoothing) : {n_split_any_1:8,d}")
    print(f"     - Any X - *(<=2) - X (<=2-res interruption)      : {n_split_any_le2:8,d}")
    print(f"  3. Chain / Template Row Prevalence (out of {n_rows:,} rows):")
    print(f"     - Rows with >=1 1-residue H/E element    : {int((len1_per_row > 0).sum()):8,d} ({100*(len1_per_row > 0).mean():5.2f}%)")
    print(f"     - Rows with >=1 (H<3 or E<2) fragment    : {int((tm_junk_per_row > 0).sum()):8,d} ({100*(tm_junk_per_row > 0).mean():5.2f}%)")
    print(f"     - Rows with >=1 (H<4 or E<2) fragment    : {int((alpha_junk_per_row > 0).sum()):8,d} ({100*(alpha_junk_per_row > 0).mean():5.2f}%)")
    print(f"     - Rows with >=1 1-res split (X-*(1)-X)   : {int((split1_per_row > 0).sum()):8,d} ({100*(split1_per_row > 0).mean():5.2f}%)")
    print(f"     - Rows with >=1 <=2-res split (X-*(<=2)-X): {int((splitle2_per_row > 0).sum()):8,d} ({100*(splitle2_per_row > 0).mean():5.2f}%)")
    any_defect = (tm_junk_per_row > 0) | (splitle2_per_row > 0)
    print(f"     - Rows with ANY (H<3 or E<2 or <=2-split): {int(any_defect.sum()):8,d} ({100*any_defect.mean():5.2f}%)")
    print(f"     - Mean (H<3 or E<2) fragments per row    : {tm_junk_per_row.mean():.2f} (p50={np.median(tm_junk_per_row):.0f}, p90={np.percentile(tm_junk_per_row,90):.0f}, p99={np.percentile(tm_junk_per_row,99):.0f})")
    print(f"     - Mean (H<4 or E<2) fragments per row    : {alpha_junk_per_row.mean():.2f} (p50={np.median(alpha_junk_per_row):.0f}, p90={np.percentile(alpha_junk_per_row,90):.0f}, p99={np.percentile(alpha_junk_per_row,99):.0f})")


def run_native_vs_synth_index_pair_audit(ids, runs_flat: np.ndarray, runs_off: np.ndarray, row_tm: np.ndarray):
    """Compare each training-band synthetic template row against its own parent native row in the index."""
    t0 = time.time()
    stem_to_nat_row = {}
    for i, s in enumerate(ids):
        s_str = str(s)
        if "@rw" not in s_str:
            stem_to_nat_row[s_str] = i

    nat_nel_map = {}
    nat_junk_map = {}
    for stem, r in stem_to_nat_row.items():
        a, b = runs_off[r], runs_off[r + 1]
        if b <= a:
            continue
        rr = runs_flat[a:b]
        he = rr[(rr[:, 0] == DSSP_HELIX) | (rr[:, 0] == DSSP_STRAND)]
        nat_nel_map[stem] = len(he)
        nat_junk_map[stem] = int(((rr[:, 0] == DSSP_HELIX) & (rr[:, 1] < 4) | (rr[:, 0] == DSSP_STRAND) & (rr[:, 1] < 2)).sum())

    excess_junk = []
    nel_diffs = []
    tmpl_has_more_junk = 0
    n_paired = 0
    for i, s in enumerate(ids):
        s_str = str(s)
        if "@rw" not in s_str:
            continue
        if row_tm is not None and not (0.5 <= float(row_tm[i]) <= 0.9):
            continue
        stem = s_str.split("@rw")[0]
        if stem not in nat_nel_map:
            continue
        a, b = runs_off[i], runs_off[i + 1]
        if b <= a:
            continue
        rr = runs_flat[a:b]
        he = rr[(rr[:, 0] == DSSP_HELIX) | (rr[:, 0] == DSSP_STRAND)]
        t_junk = int(((rr[:, 0] == DSSP_HELIX) & (rr[:, 1] < 4) | (rr[:, 0] == DSSP_STRAND) & (rr[:, 1] < 2)).sum())
        n_junk = nat_junk_map[stem]
        excess_junk.append(t_junk - n_junk)
        nel_diffs.append(len(he) - nat_nel_map[stem])
        if t_junk > n_junk:
            tmpl_has_more_junk += 1
        n_paired += 1

    if n_paired > 0:
        ex = np.asarray(excess_junk)
        nd = np.asarray(nel_diffs)
        print(f"\n================================================================================")
        print(f"PAIRED TEMPLATE-VS-NATIVE INDEX COMPARISON (n_paired={n_paired:,}, {time.time()-t0:.2f}s)")
        print(f"================================================================================")
        print(f"  Templates with MORE (H<4 or E<2) junk than their native: {tmpl_has_more_junk:,}/{n_paired:,} ({100*tmpl_has_more_junk/n_paired:.2f}%)")
        print(f"  Mean excess (H<4 or E<2) junk (template - native)     : {ex.mean():+.2f} (|diff|={np.abs(ex).mean():.2f}, p90={np.percentile(ex,90):+.0f}, p99={np.percentile(ex,99):+.0f})")
        print(f"  Mean H/E element count shift (template - native)      : {nd.mean():+.2f} (|diff|={np.abs(nd).mean():.2f})")


def run_coordinate_sample_audit(refs_tsv: str, native_npz: str, templates_root: str, n_chains_sample: int = 500):
    """Compare PyDSSP (`h_bond`) vs `tmalign` vs `tmalign_smooth` on real synthetic template 3D coordinates."""
    if not (os.path.exists(refs_tsv) and os.path.exists(templates_root)):
        print(f"[coord_audit] skipping coordinate audit: missing {refs_tsv} or {templates_root}")
        return

    t0 = time.time()
    # Read unique (stem, slot, rewind, tm, target_npz) from refs.tsv
    records = []
    seen_stems = set()
    with open(refs_tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_set = header.index("set")
        i_stem = header.index("stem")
        i_slot = header.index("slot")
        i_rw = header.index("rewind")
        i_tm = header.index("tm")
        i_tgt = header.index("target")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if parts[i_set] != "train":
                continue
            stem = parts[i_stem]
            if stem in seen_stems:
                continue
            orig_path = parts[i_tgt]
            local_path = orig_path.replace(
                "/orcd/compute/so3/002/chenxi/of_run/pp1c_work/templates_band",
                templates_root,
            )
            if os.path.exists(local_path):
                seen_stems.add(stem)
                records.append((stem, int(parts[i_slot]), int(parts[i_rw]), float(parts[i_tm]), local_path))
                if len(records) >= n_chains_sample:
                    break

    print(f"\n================================================================================")
    print(f"COORDINATE-LEVEL AUDIT ON {len(records)} SYNTHETIC TEMPLATES (from {templates_root})")
    print(f"================================================================================")

    stats = {
        k: {
            "total_he": 0,
            "total_h": 0,
            "total_e": 0,
            "he_len1": 0,
            "h_lt3": 0,
            "h_lt4": 0,
            "e_lt2": 0,
            "split_1": 0,
            "split_le2": 0,
            "rows_with_junk_tm": 0,
            "rows_with_junk_alpha": 0,
            "rows_with_split_le2": 0,
            "n_el_list": [],
            "he_frac_list": [],
            "q3_vs_r0": [],
            "he_f1_vs_r0": [],
            "nel_diff_vs_r0": [],
        }
        for k in ("h_bond", "tmalign", "tmalign_smooth")
    }

    n_evaluated = 0
    for stem, slot, rw, tm_val, npz_path in records:
        z = np.load(npz_path)
        tcoords = z["coords"]
        amask = torch.from_numpy(z["atom_mask"].astype(bool))
        aatype = z["aatype"]
        rewinds = z["rewind_steps"] if "rewind_steps" in z else None
        Lt = int(amask.shape[0])
        if slot >= tcoords.shape[0]:
            continue

        # Slot 0 in templates_band (or minimum rewind slot) is the clean/lowest-rewind reference if present
        r0_slot = int(np.argmin(rewinds)) if rewinds is not None else 0
        full_t = torch.zeros(Lt, 37, 3)
        full_t[amask] = torch.from_numpy(tcoords[slot]).float()
        full_0 = torch.zeros(Lt, 37, 3)
        full_0[amask] = torch.from_numpy(tcoords[r0_slot]).float()
        donor = torch.from_numpy(np.asarray(aatype) != PRO_IDX)[None]
        ca_mask = amask[:, CA_IDX][None]

        # Native/cleanest reference DSSP under h_bond and tmalign
        dssp_0_hb = compute_dssp_target(full_0[None], ca_mask, coord_mask=amask[None], coord_layout="atom37", donor_mask=donor, dssp_def="h_bond")[0]
        dssp_0_tm = compute_dssp_target(full_0[None], ca_mask, coord_mask=amask[None], coord_layout="atom37", donor_mask=donor, dssp_def="tmalign_smooth")[0]
        runs_0_hb = dssp_to_runs(dssp_0_hb, min_len=1)
        nel_0_hb = sum(1 for t, _ in runs_0_hb if t in (DSSP_HELIX, DSSP_STRAND))

        for mode in ("h_bond", "tmalign", "tmalign_smooth"):
            dssp_t = compute_dssp_target(
                full_t[None], ca_mask, coord_mask=amask[None], coord_layout="atom37", donor_mask=donor, dssp_def=mode
            )[0]
            runs = dssp_to_runs(dssp_t, min_len=1)
            st = stats[mode]
            he_runs = [(t, n) for t, n in runs if t in (DSSP_HELIX, DSSP_STRAND)]
            h_runs = [n for t, n in runs if t == DSSP_HELIX]
            e_runs = [n for t, n in runs if t == DSSP_STRAND]

            st["total_he"] += len(he_runs)
            st["total_h"] += len(h_runs)
            st["total_e"] += len(e_runs)
            len1 = sum(1 for _, n in he_runs if n == 1)
            hlt3 = sum(1 for n in h_runs if n < 3)
            hlt4 = sum(1 for n in h_runs if n < 4)
            elt2 = sum(1 for n in e_runs if n < 2)
            st["he_len1"] += len1
            st["h_lt3"] += hlt3
            st["h_lt4"] += hlt4
            st["e_lt2"] += elt2

            sp1 = 0
            sple2 = 0
            for idx_r in range(len(runs) - 2):
                t_a, _ = runs[idx_r]
                _, n_b = runs[idx_r + 1]
                t_c, _ = runs[idx_r + 2]
                if t_a in (DSSP_HELIX, DSSP_STRAND) and t_a == t_c:
                    if n_b == 1:
                        sp1 += 1
                    if n_b <= 2:
                        sple2 += 1
            st["split_1"] += sp1
            st["split_le2"] += sple2
            if (hlt3 + elt2) > 0:
                st["rows_with_junk_tm"] += 1
            if (hlt4 + elt2) > 0:
                st["rows_with_junk_alpha"] += 1
            if sple2 > 0:
                st["rows_with_split_le2"] += 1

            st["n_el_list"].append(len(he_runs))
            valid_res = (dssp_t >= 0) & (dssp_0_hb >= 0)
            n_val = int(valid_res.sum().item())
            he_frac = float(((dssp_t == DSSP_HELIX) | (dssp_t == DSSP_STRAND)).sum().item()) / max(n_val, 1)
            st["he_frac_list"].append(he_frac)
            if n_val > 0:
                q3 = float((dssp_t[valid_res] == dssp_0_hb[valid_res]).float().mean().item())
                st["q3_vs_r0"].append(q3)
                pred_he = ((dssp_t[valid_res] == DSSP_HELIX) | (dssp_t[valid_res] == DSSP_STRAND)).float()
                gt_he = ((dssp_0_hb[valid_res] == DSSP_HELIX) | (dssp_0_hb[valid_res] == DSSP_STRAND)).float()
                tp = float((pred_he * gt_he).sum().item())
                fp = float((pred_he * (1.0 - gt_he)).sum().item())
                fn = float(((1.0 - pred_he) * gt_he).sum().item())
                f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-8)
                st["he_f1_vs_r0"].append(f1)
                st["nel_diff_vs_r0"].append(abs(len(he_runs) - nel_0_hb))
        n_evaluated += 1

    print(f"Evaluated {n_evaluated} synthetic templates in {time.time()-t0:.2f}s:")
    print(f"{'Metric':<44} | {'h_bond (PyDSSP)':>16} | {'tmalign (raw)':>16} | {'tmalign_smooth':>16}")
    print("-" * 102)
    for label, fn in [
        ("Mean H/E elements per template", lambda s: f"{np.mean(s['n_el_list']):16.2f}"),
        ("Mean H/E residue fraction", lambda s: f"{100*np.mean(s['he_frac_list']):15.2f}%"),
        ("1-res H/E elements (% of all H/E)", lambda s: f"{s['he_len1']:6d} ({100*s['he_len1']/max(s['total_he'],1):5.2f}%)"),
        ("Impossible H<3 or E<2 (% of all H/E)", lambda s: f"{s['h_lt3']+s['e_lt2']:6d} ({100*(s['h_lt3']+s['e_lt2'])/max(s['total_he'],1):5.2f}%)"),
        ("Sub-turn H<4 or E<2 (% of all H/E)", lambda s: f"{s['h_lt4']+s['e_lt2']:6d} ({100*(s['h_lt4']+s['e_lt2'])/max(s['total_he'],1):5.2f}%)"),
        ("1-res split X-*(1)-X count", lambda s: f"{s['split_1']:16d}"),
        ("<=2-res split X-*(<=2)-X count", lambda s: f"{s['split_le2']:16d}"),
        ("Templates w/ >=1 (H<3 or E<2) (%)", lambda s: f"{100*s['rows_with_junk_tm']/max(n_evaluated,1):15.2f}%"),
        ("Templates w/ >=1 (H<4 or E<2) (%)", lambda s: f"{100*s['rows_with_junk_alpha']/max(n_evaluated,1):15.2f}%"),
        ("Templates w/ >=1 <=2-res split (%)", lambda s: f"{100*s['rows_with_split_le2']/max(n_evaluated,1):15.2f}%"),
        ("Q3 agreement vs lowest-rewind slot (%)", lambda s: f"{100*np.mean(s['q3_vs_r0']):15.2f}%"),
        ("H/E binary F1 vs lowest-rewind slot", lambda s: f"{np.mean(s['he_f1_vs_r0']):16.4f}"),
        ("Mean |n_el(tmpl) - n_el(low_rw)|", lambda s: f"{np.mean(s['nel_diff_vs_r0']):16.2f}"),
    ]:
        print(f"{label:<44} | {fn(stats['h_bond']):>16} | {fn(stats['tmalign']):>16} | {fn(stats['tmalign_smooth']):>16}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/home/gridsan/cou/tri_synth_bench/inputs/topology_index_confind_synth.pt")
    ap.add_argument("--refs-tsv", default="/home/gridsan/cou/geom_audit/refs.tsv")
    ap.add_argument("--native-npz", default="/home/gridsan/cou/geom_audit/native.npz")
    ap.add_argument("--templates-root", default="/home/gridsan/cou/pp1c_work/templates_band")
    ap.add_argument("--n-coord-samples", type=int, default=500)
    args = ap.parse_args()

    idx = torch.load(args.index, map_location="cpu", weights_only=False, mmap=True)
    ids = idx["ids"]
    runs_flat = idx["runs_flat"].numpy()
    runs_off = idx["runs_offset"].numpy()
    row_tm = idx["row_tm"].numpy() if "row_tm" in idx else None

    is_tmpl = np.array(["@rw" in str(x) for x in ids], dtype=bool)
    nat_rows = np.flatnonzero(~is_tmpl)
    tmpl_rows = np.flatnonzero(is_tmpl)

    analyze_runs_subset(runs_flat, runs_off, nat_rows, "Native Chains (stored PyDSSP)")
    analyze_runs_subset(runs_flat, runs_off, tmpl_rows, "All Synthetic Templates in Index (PyDSSP)")
    if row_tm is not None:
        in_band = is_tmpl & (row_tm >= 0.5) & (row_tm <= 0.9)
        analyze_runs_subset(runs_flat, runs_off, np.flatnonzero(in_band), "Training-Band Synthetic Templates (TM 0.5-0.9)")
    run_native_vs_synth_index_pair_audit(ids, runs_flat, runs_off, row_tm)

    run_coordinate_sample_audit(args.refs_tsv, args.native_npz, args.templates_root, args.n_coord_samples)


if __name__ == "__main__":
    main()
