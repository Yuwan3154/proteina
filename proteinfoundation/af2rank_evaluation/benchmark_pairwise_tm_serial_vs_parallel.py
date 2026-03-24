#!/usr/bin/env python3
"""
Benchmark all-to-all pairwise TMscore (USalign):

  1) Serial: one ``USalign pdb_i pdb_j -TMscore 5`` per pair (pair_workers=1).
  2) Parallel: ThreadPoolExecutor over pairs (same as diversity fallback).
  3) USalign -dir: single ``USalign -dir <folder> <chain_list> -TMscore 5`` run
     (same as proteina_diversity default path).

All modes use on-disk PDBs (no temp CA-only files). Pair order may differ
between modes; comparisons use sorted TM values.

Run with the cue_openfold conda interpreter (adjust path if your conda root differs):

  cd /path/to/proteina
  PYTHONPATH=. /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \\
    proteinfoundation/af2rank_evaluation/benchmark_pairwise_tm_serial_vs_parallel.py \\
    --protein_dir inference/.../6U1O_A --protein_id 6U1O_A
"""

import argparse
import glob
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

from proteinfoundation.af2rank_evaluation.proteina_diversity import (
    _pairwise_tm_via_usalign_dir,
    resolve_num_workers,
)
from proteinfoundation.af2rank_evaluation.proteinebm_scorer import tmscore_pdb_paths

_USALIGN_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def discover_pdb_paths(protein_dir: str, protein_id: str) -> tuple:
    """Sorted decoy paths and basenames (same glob order as diversity)."""
    pattern = os.path.join(protein_dir, f"{protein_id}_*.pdb")
    pdb_paths = sorted(glob.glob(pattern))
    if len(pdb_paths) < 2:
        print(f"Need >= 2 PDBs matching {protein_id}_*.pdb, found {len(pdb_paths)}", file=sys.stderr)
        sys.exit(1)
    basenames = [os.path.basename(p) for p in pdb_paths]
    return pdb_paths, basenames


def run_all_pairs_serial(pdb_paths: list) -> list:
    n = len(pdb_paths)
    pair_indices = list(combinations(range(n), 2))
    out = []
    for i, j in pair_indices:
        r = tmscore_pdb_paths(pdb_paths[i], pdb_paths[j], env=None)
        out.append(float(r["tms"]))
    return out


def run_all_pairs_parallel(pdb_paths: list, pair_workers: int) -> list:
    n = len(pdb_paths)
    pair_indices = list(combinations(range(n), 2))
    pw = max(1, int(pair_workers))
    tm_env = _USALIGN_PARALLEL_ENV if pw > 1 else None

    def _one(ij):
        i, j = ij
        r = tmscore_pdb_paths(pdb_paths[i], pdb_paths[j], env=tm_env)
        return float(r["tms"])

    with ThreadPoolExecutor(max_workers=pw) as ex:
        return list(ex.map(_one, pair_indices))


def main():
    default_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "inference",
        "inference_seq_cond_sampling_ca_dssp_beta-2.5-2.0_finetune-all_v1.6_default-fold_21-seq-S25_128-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise",
        "6U1O_A",
    )
    default_dir = os.path.normpath(default_dir)

    p = argparse.ArgumentParser(description="Benchmark pairwise TM: serial vs parallel vs USalign -dir")
    p.add_argument("--protein_dir", default=default_dir, help="Directory with decoy PDBs")
    p.add_argument("--protein_id", default="6U1O_A", help="Prefix for PDB glob")
    p.add_argument(
        "--parallel_workers",
        type=int,
        default=None,
        help="Thread pool size for parallel run (default: resolve_num_workers(None))",
    )
    p.add_argument(
        "--skip_serial",
        action="store_true",
        help="Skip serial per-pair run",
    )
    p.add_argument(
        "--skip_parallel",
        action="store_true",
        help="Skip ThreadPoolExecutor per-pair run",
    )
    p.add_argument(
        "--skip_usalign_dir",
        action="store_true",
        help="Skip USalign -dir all-against-all run",
    )
    args = p.parse_args()

    protein_dir = os.path.abspath(os.path.expanduser(args.protein_dir))
    if not os.path.isdir(protein_dir):
        print(f"Not a directory: {protein_dir}", file=sys.stderr)
        sys.exit(1)

    pw = resolve_num_workers(args.parallel_workers)

    print(f"protein_dir: {protein_dir}")
    print(f"protein_id:  {args.protein_id}")
    print("Discovering PDB files...")
    t0 = time.perf_counter()
    pdb_paths, basenames = discover_pdb_paths(protein_dir, args.protein_id)
    t_discover = time.perf_counter() - t0
    n = len(pdb_paths)
    n_pairs = n * (n - 1) // 2
    print(f"  structures: {n}  pairs: {n_pairs}  discover_time: {t_discover:.3f}s")
    print(f"parallel_workers (for threaded per-pair run): {pw}")
    print()

    _ = tmscore_pdb_paths(pdb_paths[0], pdb_paths[1], env=None)

    serial_vals = None
    t_serial = None
    if not args.skip_serial:
        t1 = time.perf_counter()
        serial_vals = run_all_pairs_serial(pdb_paths)
        t_serial = time.perf_counter() - t1
        print(f"Serial (pair_workers=1):       {t_serial:.3f}s  mean_tm={float(np.mean(serial_vals)):.6f}")

    parallel_vals = None
    t_parallel = None
    if not args.skip_parallel:
        t2 = time.perf_counter()
        parallel_vals = run_all_pairs_parallel(pdb_paths, pw)
        t_parallel = time.perf_counter() - t2
        print(f"Parallel (pair_workers={pw}):  {t_parallel:.3f}s  mean_tm={float(np.mean(parallel_vals)):.6f}")

    dir_vals = None
    t_dir = None
    if not args.skip_usalign_dir:
        if shutil.which("USalign") is None:
            print("USalign not on PATH; skipping -dir mode.", file=sys.stderr)
        else:
            t3 = time.perf_counter()
            dir_vals = _pairwise_tm_via_usalign_dir(protein_dir, basenames, env=None)
            t_dir = time.perf_counter() - t3
            if dir_vals is None or len(dir_vals) != n_pairs:
                print(
                    f"USalign -dir failed or wrong count (got {len(dir_vals or [])}, expected {n_pairs})",
                    file=sys.stderr,
                )
                dir_vals = None
                t_dir = None
            else:
                print(
                    f"USalign -dir (single pass):    {t_dir:.3f}s  "
                    f"mean_tm={float(np.mean(dir_vals)):.6f}"
                )

    print()

    def _sorted_match(a: list, b: list) -> bool:
        if a is None or b is None or len(a) != len(b):
            return False
        return bool(np.allclose(np.sort(a), np.sort(b), rtol=0.0, atol=1e-5))

    if serial_vals is not None and parallel_vals is not None:
        same_sp = len(serial_vals) == len(parallel_vals) and np.allclose(
            serial_vals, parallel_vals, rtol=0.0, atol=1e-9
        )
        print(f"Values match (serial vs parallel, same pair order): {same_sp}")
        if t_serial and t_parallel and t_parallel > 0:
            print(f"Speedup serial / parallel: {t_serial / t_parallel:.2f}x")

    if serial_vals is not None and dir_vals is not None:
        same_sd = _sorted_match(serial_vals, dir_vals)
        print(f"Values match (serial vs -dir, sorted TM lists): {same_sd}")
        if t_serial and t_dir and t_dir > 0:
            print(f"Speedup serial / USalign -dir: {t_serial / t_dir:.2f}x")

    if parallel_vals is not None and dir_vals is not None and t_parallel and t_dir:
        same_pd = _sorted_match(parallel_vals, dir_vals)
        print(f"Values match (parallel vs -dir, sorted TM lists): {same_pd}")
        if t_dir > 0:
            print(f"Speedup parallel / USalign -dir: {t_parallel / t_dir:.2f}x")


if __name__ == "__main__":
    main()
