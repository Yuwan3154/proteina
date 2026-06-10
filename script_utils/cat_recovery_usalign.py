"""Structural CAT-label recovery via USalign TM-align.

E1: build a per-CAT representative library -- for each CAT (3-level CATH) the
    largest seq-sim cluster's single-domain member (chain_to_cat list length 1),
    reconstructed from its processed .pt to a CA-only PDB at <out>/reps/<CAT>.pdb.
E2: queries = the representatives of no-CAT clusters (no labeled member), each
    reconstructed to a CA-only PDB at <out>/queries/<stem>.pdb.
E3: each query (structure 1) is aligned against the whole rep library via
    `USalign <query> -dir2 reps/ rep_list.txt -outfmt 2` in DEFAULT TMalign mode
    (no -TMscore). TM1 (normalized by the query, the chain lacking a CAT label)
    is used. multiprocessing.Pool over queries; resumable via done.txt.
E4/E5: per-query best CAT + multi-hit counts written to results.tsv; --report-only
    prints the TM and multi-hit distributions. Labels are NOT auto-applied.

Run on Engaging (DATA_PATH set), CPU-only salloc:
    conda activate cue_openfold
    python script_utils/cat_recovery_usalign.py --limit-queries 20   # smoke
    python script_utils/cat_recovery_usalign.py                      # full
    python script_utils/cat_recovery_usalign.py --report-only
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import shutil
import subprocess
from collections import defaultdict

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.openfold_stub.np import residue_constants as rc
from proteinfoundation.prediction_pipeline.usalign_tabular import (
    iter_usalign_outfmt2_rows,
    normalize_usalign_structure_name,
)
from proteinfoundation.utils.cluster_utils import read_cluster_tsv
from proteinfoundation.utils.coors_utils import trans_ang_to_atom37
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb

DEFAULT_CLUSTER_TSV_NAME = (
    "cluster_seqid_0.25_df_pdb_f1_minl50_maxl256_mtprotein_etdiffractionEM_"
    "minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.tsv"
)
DEFAULT_RES_CSV_NAME = (
    "df_pdb_f1_minl50_maxl256_mtprotein_etdiffractionEM_"
    "minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.csv"
)
DEFAULT_THRESHOLDS = [0.4, 0.5, 0.6]
_CA_IDX = rc.atom_order["CA"]
_UNIT_CHECKED = False


def reconstruct_ca_pdb(stem, processed_dir, manifest, out_pdb):
    """Reconstruct a CA-only PDB (Angstrom) from a processed .pt. Returns path or None."""
    global _UNIT_CHECKED
    if os.path.exists(out_pdb):
        return out_pdb
    pt_path = _processed_path_sharded(processed_dir, stem, manifest)
    if not pt_path.exists():
        return None
    graph = torch.load(pt_path, map_location="cpu", weights_only=False)
    coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
    ca = coords[:, 1, :].float()
    ca_mask = coord_mask[:, 1].float()
    if float(ca_mask.sum()) < 1.0:
        return None
    if not _UNIT_CHECKED:
        valid = ca[ca_mask > 0]
        span = float((valid.max(dim=0).values - valid.min(dim=0).values).max())
        assert span > 15.0, (
            f"CA span {span:.1f} for {stem}; coords do not look like Angstrom "
            "(nm would give ~3-30). Reconstruction would corrupt TM-scores."
        )
        print(f"[unit-check] {stem} CA span = {span:.1f} A (Angstrom OK)")
        _UNIT_CHECKED = True
    atom37 = trans_ang_to_atom37(ca).detach().cpu().numpy()
    aatype = graph.residue_type.long().detach().cpu().numpy()
    atom37_mask = np.zeros((aatype.shape[0], 37), dtype=np.float32)
    atom37_mask[:, _CA_IDX] = ca_mask.detach().cpu().numpy()
    os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
    write_prot_to_pdb(
        atom37, out_pdb, aatype=aatype, atom37_mask=atom37_mask,
        overwrite=True, no_indexing=True,
    )
    return out_pdb


def load_resolution_map(csv_path):
    """id -> resolution (float). Skips empty / 'None' values."""
    res = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("resolution")
            if v in (None, "", "None"):
                continue
            res[row["id"]] = float(v)
    return res


def _pick_rep(ms, cname, rep_pick, res_map):
    """Within-cluster representative for a CAT. Returns (rep_seqid, used_fallback).
    'lexmin'=lowest id; 'mmseqs'=cluster mmseqs rep (=cluster name) if it is a
    single-domain member of the CAT else fall back to lexmin; 'hires'=lowest
    resolution value (best quality), ties broken by id."""
    if rep_pick == "mmseqs":
        if cname in ms:
            return cname, False
        return min(ms), True
    if rep_pick == "hires":
        return min(ms, key=lambda x: (res_map.get(x, float("inf")), x)), False
    return min(ms), False


def select_cat_reps(cluster_dict, chain_to_cat, rep_pick="lexmin", res_map=None):
    """CAT -> (cluster_size, cluster_name, rep_seqid) from the largest cluster
    holding a single-domain member of that CAT (deterministic tie-breaks).
    The winning cluster is chosen by (size, cluster_name) independent of rep_pick;
    rep_pick only selects which member within that cluster becomes the rep."""
    if rep_pick == "hires":
        assert res_map is not None, "rep_pick='hires' requires res_map."
    best = {}
    winner_fb = {}
    for cname in sorted(cluster_dict):
        members = cluster_dict[cname]
        size = len(members)
        per_cat = defaultdict(list)
        for m in members:
            cats = chain_to_cat.get(m.lower())
            if cats and len(cats) == 1:
                per_cat[cats[0]].append(m)
        for cat, ms in per_cat.items():
            cur = best.get(cat)
            if cur is None or size > cur[0] or (size == cur[0] and cname < cur[1]):
                rep, fb = _pick_rep(ms, cname, rep_pick, res_map)
                best[cat] = (size, cname, rep)
                winner_fb[cat] = fb
    if rep_pick == "mmseqs":
        n_fb = sum(1 for v in winner_fb.values() if v)
        print(f"[rep_pick=mmseqs] winning clusters whose mmseqs rep is NOT a single-domain "
              f"member of the CAT (fell back to lexmin): {n_fb}/{len(best)}")
    return best


def find_nocat_cluster_reps(cluster_dict, chain_to_cat):
    """Representatives (cluster_name) of clusters with no labeled member."""
    reps = []
    for cname in sorted(cluster_dict):
        if any(chain_to_cat.get(m.lower()) for m in cluster_dict[cname]):
            continue
        reps.append(cname)
    return reps


def align_one_query(args):
    """Worker: align one query (structure 1) vs the rep library. Returns a result dict."""
    query_stem, query_pdb, usalign_bin, reps_dir, rep_list_path, thresholds = args
    cmd = [usalign_bin, query_pdb, "-dir2", reps_dir.rstrip("/") + "/", rep_list_path, "-outfmt", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cat_tm = []
    if proc.returncode == 0:
        for parts in iter_usalign_outfmt2_rows(proc.stdout):
            if len(parts) <= 4:
                continue
            cat = os.path.splitext(normalize_usalign_structure_name(parts[1]))[0]
            cat_tm.append((cat, float(parts[2])))
    cat_tm.sort(key=lambda x: (-x[1], x[0]))
    n_ge = [sum(1 for _, tm in cat_tm if tm >= t) for t in thresholds]
    if cat_tm:
        best_cat, best_tm = cat_tm[0]
    else:
        best_cat, best_tm = "NA", 0.0
    top5 = "|".join(f"{c}:{tm:.4f}" for c, tm in cat_tm[:5])
    return {
        "query": query_stem,
        "best_cat": best_cat,
        "best_tm": best_tm,
        "n_ge": n_ge,
        "top5": top5,
        "ok": proc.returncode == 0,
    }


def read_done(done_path):
    if not os.path.exists(done_path):
        return set()
    with open(done_path) as f:
        return {line.strip() for line in f if line.strip()}


def run_report(results_tsv):
    if not os.path.exists(results_tsv):
        print(f"[report] no results at {results_tsv}")
        return
    with open(results_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        ge_cols = [(i, c) for i, c in enumerate(header) if c.startswith("n_ge_")]
        best_idx = header.index("best_tm")
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    n = len(rows)
    print(f"\n=== CAT recovery report ({n} queries) ===")
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
    hist = [0] * (len(bins) - 1)
    for r in rows:
        v = float(r[best_idx])
        for b in range(len(bins) - 1):
            if bins[b] <= v < bins[b + 1]:
                hist[b] += 1
                break
    print("best_tm histogram:")
    for b in range(len(bins) - 1):
        print(f"  [{bins[b]:.2f}, {bins[b+1]:.2f}): {hist[b]}")
    for idx, col in ge_cols:
        ge = sum(1 for r in rows if int(r[idx]) >= 1)
        print(f"recovered (>=1 rep at {col.replace('n_ge_','TM>=')}): {ge}  ({100.0*ge/n:.1f}%)" if n else f"{col}: 0")
    print("multi-hit (in-between) distribution by n_ge_0.5:")
    mh_idx = None
    for idx, col in ge_cols:
        if col == "n_ge_0.5":
            mh_idx = idx
    if mh_idx is not None:
        dist = defaultdict(int)
        for r in rows:
            k = int(r[mh_idx])
            dist[min(k, 4)] += 1
        for k in sorted(dist):
            label = f"{k}" if k < 4 else ">=4"
            print(f"  hits={label}: {dist[k]}")
    print("Note: labeled-side analogue = 133 genuinely-impure clusters (different chains, different CAT).")
    print("Labels are NOT auto-applied; pick a TM cutoff and decide whether to fold into chain_to_cat.")


def main():
    parser = argparse.ArgumentParser(description="Structural CAT-label recovery via USalign.")
    parser.add_argument("--chain-to-cat", default=None, help="pickle (default: $DATA_PATH/cath_shared/pdb_chain_to_cat.pkl)")
    parser.add_argument("--cluster-tsv", default=None, help="training cluster TSV (default: $DATA_PATH/pdb_train/<known name>)")
    parser.add_argument("--data-dir", default=None, help="dataset dir with processed/ + shard_manifest.json (default: $DATA_PATH/pdb_train)")
    parser.add_argument("--out-dir", default=None, help="output dir (default: $DATA_PATH/cath_shared/cat_recovery)")
    parser.add_argument("--usalign-bin", default=None, help="USalign binary (default: from PATH)")
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit-queries", type=int, default=None, help="cap #queries (smoke test)")
    parser.add_argument("--tm-thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rep-pick", choices=["lexmin", "mmseqs", "hires"], default="lexmin",
                        help="within-cluster CAT representative selection")
    parser.add_argument("--resolution-csv", default=None,
                        help="df CSV with id,resolution (default: $DATA_PATH/pdb_train/<known name>)")
    parser.add_argument("--reps-only", action="store_true",
                        help="build rep library + reps_manifest.tsv, then exit before alignment")
    args = parser.parse_args()

    data_path = os.environ.get("DATA_PATH")
    if data_path is None and None in (args.chain_to_cat, args.cluster_tsv, args.data_dir, args.out_dir):
        parser.error("DATA_PATH env not set; pass --chain-to-cat, --cluster-tsv, --data-dir, --out-dir explicitly.")
    chain_to_cat_path = args.chain_to_cat or os.path.join(data_path, "cath_shared", "pdb_chain_to_cat.pkl")
    cluster_tsv = args.cluster_tsv or os.path.join(data_path, "pdb_train", DEFAULT_CLUSTER_TSV_NAME)
    data_dir = pathlib.Path(args.data_dir or os.path.join(data_path, "pdb_train"))
    out_dir = args.out_dir or os.path.join(data_path, "cath_shared", "cat_recovery")
    results_tsv = os.path.join(out_dir, "results.tsv")

    if args.report_only:
        run_report(results_tsv)
        return

    usalign_bin = args.usalign_bin or shutil.which("USalign")
    assert usalign_bin and os.path.exists(usalign_bin), "USalign not found; pass --usalign-bin."

    processed_dir = data_dir / "processed"
    manifest_path = data_dir / "shard_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else None

    with open(chain_to_cat_path, "rb") as f:
        chain_to_cat = pickle.load(f)
    cluster_dict = read_cluster_tsv(cluster_tsv)
    print(f"clusters={len(cluster_dict)} chains_labeled={len(chain_to_cat)} sharded={manifest is not None}")

    res_csv = args.resolution_csv or (os.path.join(data_path, "pdb_train", DEFAULT_RES_CSV_NAME) if data_path else None)
    res_map = load_resolution_map(res_csv) if (res_csv and os.path.exists(res_csv)) else {}
    if args.rep_pick == "hires" and not res_map:
        parser.error(f"--rep-pick hires needs a resolution CSV; not found at {res_csv}")
    if res_map:
        print(f"resolution map: {len(res_map)} chains from {res_csv}")

    reps_dir = os.path.join(out_dir, "reps")
    queries_dir = os.path.join(out_dir, "queries")
    rep_list_path = os.path.join(out_dir, "rep_list.txt")
    os.makedirs(reps_dir, exist_ok=True)
    os.makedirs(queries_dir, exist_ok=True)

    # E1: representative library
    cat_reps = select_cat_reps(cluster_dict, chain_to_cat, rep_pick=args.rep_pick, res_map=res_map)
    all_cats = {cat for cats in chain_to_cat.values() for cat in cats}
    print(f"distinct CAT (any chain in labeled set)={len(all_cats)}; CAT with single-domain rep={len(cat_reps)}; "
          f"CAT without single-domain rep={len(all_cats - set(cat_reps))}")
    manifest_rows = []
    rep_names = []
    for cat in sorted(cat_reps):
        size, cname, seqid = cat_reps[cat]
        manifest_rows.append((cat, size, cname, seqid, res_map.get(seqid, "")))
        out_pdb = os.path.join(reps_dir, f"{cat}.pdb")
        if reconstruct_ca_pdb(seqid, processed_dir, manifest, out_pdb) is not None:
            rep_names.append(f"{cat}.pdb")
    with open(rep_list_path, "w") as f:
        f.write("\n".join(rep_names) + "\n")
    reps_manifest_path = os.path.join(out_dir, "reps_manifest.tsv")
    with open(reps_manifest_path, "w") as f:
        f.write("cat\tcluster_size\tcluster\trep_seqid\tresolution\n")
        for row in manifest_rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"rep library: {len(rep_names)} reps -> {reps_dir}")
    print(f"reps manifest -> {reps_manifest_path}")
    if args.reps_only:
        return

    # E2: no-CAT cluster representatives as queries
    nocat_reps = find_nocat_cluster_reps(cluster_dict, chain_to_cat)
    print(f"no-CAT cluster reps (queries): {len(nocat_reps)}")
    query_items = []
    for seqid in nocat_reps:
        out_pdb = os.path.join(queries_dir, f"{seqid}.pdb")
        if reconstruct_ca_pdb(seqid, processed_dir, manifest, out_pdb) is not None:
            query_items.append((seqid, out_pdb))
    print(f"queries reconstructed: {len(query_items)}")

    # E3: resumable alignment
    done = read_done(os.path.join(out_dir, "done.txt"))
    pending = [(s, p) for (s, p) in query_items if s not in done]
    if args.limit_queries is not None:
        pending = pending[: args.limit_queries]
    print(f"aligning {len(pending)} queries ({len(done)} already done) with nproc={args.nproc}")

    write_header = not os.path.exists(results_tsv)
    cols = ["query", "best_cat", "best_tm"] + [f"n_ge_{t}" for t in args.tm_thresholds] + ["top5"]
    work = [(s, p, usalign_bin, reps_dir, rep_list_path, args.tm_thresholds) for (s, p) in pending]
    done_f = open(os.path.join(out_dir, "done.txt"), "a")
    res_f = open(results_tsv, "a")
    if write_header:
        res_f.write("\t".join(cols) + "\n")
    n_fail = 0
    with mp.Pool(args.nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(align_one_query, work, chunksize=1), 1):
            row = [r["query"], r["best_cat"], f"{r['best_tm']:.4f}"] + [str(x) for x in r["n_ge"]] + [r["top5"]]
            res_f.write("\t".join(row) + "\n")
            res_f.flush()
            done_f.write(r["query"] + "\n")
            done_f.flush()
            if not r["ok"]:
                n_fail += 1
            if i % 100 == 0:
                print(f"  ...{i}/{len(work)} (usalign failures so far: {n_fail})")
    done_f.close()
    res_f.close()
    print(f"done: {len(work)} aligned, {n_fail} USalign failures. results -> {results_tsv}")
    run_report(results_tsv)


if __name__ == "__main__":
    main()
