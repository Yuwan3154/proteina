#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Benchmark ext_lig preprocessing performance.

Compares the production cKDTree implementation against a brute-force oracle
implementation over synthetic deposits of varying sizes, and optionally over
real cached .pt + raw PDB files.

Usage
-----
# Synthetic only (no data required)
python script_utils/benchmark_ext_lig.py

# Include real PDB files (requires DATA_PATH env var and processed PDB cache)
python script_utils/benchmark_ext_lig.py --real-pdb-dir $DATA_PATH/pdb_processed \
    --raw-pdb-dir $DATA_PATH/pdb_raw --n-real 50

# Adjust synthetic sizes
python script_utils/benchmark_ext_lig.py --reps 20

Output
------
Prints a table with p50/p95 latency, peak estimated memory, and label
distribution for each configuration. Asserts exact oracle agreement on all
samples.
"""

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure proteinfoundation is importable when run from the repo root
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from proteinfoundation.datasets.ext_lig_utils import (
    EXT_LIG_ABSENT,
    EXT_LIG_PRESENT,
    EXT_LIG_UNKNOWN,
    WATER_RESIDUE_NAMES,
    _CA_CUTOFF,
    compute_ext_lig_from_df,
)


# ===========================================================================
# Brute-force oracle
# ===========================================================================


def _brute_force_ext_lig(full_df, self_chains, graph_residue_ids, cutoff=_CA_CUTOFF):
    """Numpy pairwise-distance oracle – used to verify cKDTree output."""
    import torch

    L = len(graph_residue_ids)
    import torch
    ext_lig = torch.full((L,), EXT_LIG_ABSENT, dtype=torch.long)

    if isinstance(self_chains, str):
        self_chain_set = {self_chains}
    else:
        self_chain_set = set(self_chains)

    external_mask = ~full_df["chain_id"].isin(self_chain_set)
    non_water_mask = ~full_df["residue_name"].isin(WATER_RESIDUE_NAMES)
    ext_atoms = full_df[external_mask & non_water_mask]
    if len(ext_atoms) == 0:
        return ext_lig
    ext_coords = ext_atoms[["x_coord", "y_coord", "z_coord"]].values.astype(np.float64)

    self_atoms = full_df[full_df["chain_id"].isin(self_chain_set)]
    self_ca = self_atoms[self_atoms["atom_name"] == "CA"]
    ca_map = {}
    for _, row in self_ca.iterrows():
        ins = row.get("insertion", "")
        if pd.isna(ins):
            ins = ""
        key = f"{row['chain_id']}:{row['residue_name']}:{int(row['residue_number'])}:{ins}"
        ca_map[key] = np.array(
            [row["x_coord"], row["y_coord"], row["z_coord"]], dtype=np.float64
        )

    for i, resid in enumerate(graph_residue_ids):
        ca = ca_map.get(resid)
        if ca is None:
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        if np.any(np.abs(ca) < 1e-4) and np.allclose(ca, 0.0, atol=1e-4):
            ext_lig[i] = EXT_LIG_UNKNOWN
            continue
        dists = np.linalg.norm(ext_coords - ca, axis=1)
        if np.any(dists <= cutoff):
            ext_lig[i] = EXT_LIG_PRESENT
    return ext_lig


# ===========================================================================
# Synthetic deposit generation
# ===========================================================================


def _make_synthetic_deposit(
    n_self: int,
    n_ext: int,
    n_ext_chains: int = 2,
    box: float = 50.0,
    seed: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a minimal synthetic deposit DataFrame with one self-chain (A) and
    several external chains.

    Returns
    -------
    df : deposit DataFrame
    graph_resids : list of residue ID strings aligned to chain A CAs
    """
    rng = np.random.default_rng(seed)
    rows = []
    graph_resids = []

    # Self chain A: CA atoms at random positions (well away from origin)
    for i in range(n_self):
        ca = rng.uniform(20.0, 20.0 + box, 3)
        resid = f"A:ALA:{i + 1}:"
        rows.append(
            dict(
                chain_id="A",
                residue_number=i + 1,
                residue_name="ALA",
                atom_name="CA",
                x_coord=float(ca[0]),
                y_coord=float(ca[1]),
                z_coord=float(ca[2]),
                insertion="",
            )
        )
        graph_resids.append(resid)

    # External chains: random atom positions
    atoms_per_chain = max(1, n_ext // max(1, n_ext_chains))
    for ci in range(n_ext_chains):
        chain = chr(ord("B") + ci)
        for j in range(atoms_per_chain):
            xyz = rng.uniform(20.0, 20.0 + box, 3)
            rows.append(
                dict(
                    chain_id=chain,
                    residue_number=j + 1,
                    residue_name="LIG",
                    atom_name="C1",
                    x_coord=float(xyz[0]),
                    y_coord=float(xyz[1]),
                    z_coord=float(xyz[2]),
                    insertion="",
                )
            )

    return pd.DataFrame(rows), graph_resids


# ===========================================================================
# Timing helpers
# ===========================================================================


def _time_fn(fn, *args, **kwargs) -> Tuple[float, object]:
    """Return (wall_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def _peak_memory_bytes(fn, *args, **kwargs) -> Tuple[int, object]:
    """Return (peak_bytes, result) using tracemalloc."""
    tracemalloc.start()
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result


def _label_distribution(ext_lig) -> dict:
    import torch
    t = ext_lig if isinstance(ext_lig, torch.Tensor) else torch.tensor(ext_lig)
    n = t.numel()
    return {
        "present_%": 100.0 * (t == EXT_LIG_PRESENT).sum().item() / max(n, 1),
        "absent_%": 100.0 * (t == EXT_LIG_ABSENT).sum().item() / max(n, 1),
        "unknown_%": 100.0 * (t == EXT_LIG_UNKNOWN).sum().item() / max(n, 1),
    }


# ===========================================================================
# Benchmark runners
# ===========================================================================


def _run_synthetic_benchmark(configs, reps: int = 10, verify_oracle: bool = True):
    """
    Run benchmark over a list of (n_self, n_ext) configurations.

    Parameters
    ----------
    configs : list of (label, n_self, n_ext) tuples
    reps : number of repetitions per configuration
    verify_oracle : if True, check exact agreement with brute-force oracle

    Returns
    -------
    list of result dicts
    """
    import torch
    results = []
    for label, n_self, n_ext in configs:
        prod_times = []
        oracle_times = []
        peak_mems = []
        all_label_dist = []
        n_agree = 0

        for rep in range(reps):
            df, graph_resids = _make_synthetic_deposit(n_self, n_ext, seed=rep)

            # Production (cKDTree)
            t_prod, prod_result = _time_fn(
                compute_ext_lig_from_df, df, "A", graph_resids
            )
            prod_times.append(t_prod)

            # Oracle (brute-force)
            t_oracle, oracle_result = _time_fn(
                _brute_force_ext_lig, df, "A", graph_resids
            )
            oracle_times.append(t_oracle)

            # Agreement check
            if verify_oracle:
                agree = torch.equal(prod_result, oracle_result)
                n_agree += int(agree)
                if not agree:
                    mismatch_idx = (prod_result != oracle_result).nonzero(as_tuple=True)[0]
                    print(
                        f"  MISMATCH in {label} rep={rep}: "
                        f"{len(mismatch_idx)} positions differ"
                    )

            # Peak memory for production
            peak, _ = _peak_memory_bytes(
                compute_ext_lig_from_df, df, "A", graph_resids
            )
            peak_mems.append(peak)
            all_label_dist.append(_label_distribution(prod_result))

        prod_arr = np.array(prod_times) * 1000  # ms
        oracle_arr = np.array(oracle_times) * 1000  # ms

        # Average label distribution
        avg_present = np.mean([d["present_%"] for d in all_label_dist])
        avg_absent = np.mean([d["absent_%"] for d in all_label_dist])
        avg_unknown = np.mean([d["unknown_%"] for d in all_label_dist])

        results.append(
            dict(
                label=label,
                n_self=n_self,
                n_ext=n_ext,
                prod_p50_ms=np.percentile(prod_arr, 50),
                prod_p95_ms=np.percentile(prod_arr, 95),
                oracle_p50_ms=np.percentile(oracle_arr, 50),
                oracle_p95_ms=np.percentile(oracle_arr, 95),
                speedup_p50=np.percentile(oracle_arr, 50) / max(np.percentile(prod_arr, 50), 1e-9),
                peak_mem_kb=np.mean(peak_mems) / 1024,
                oracle_agree_pct=100.0 * n_agree / reps,
                avg_present_pct=avg_present,
                avg_absent_pct=avg_absent,
                avg_unknown_pct=avg_unknown,
            )
        )
    return results


def _print_results(results: list):
    print(
        f"\n{'Config':<30} {'n_self':>6} {'n_ext':>6} "
        f"{'prod p50 (ms)':>14} {'prod p95 (ms)':>14} "
        f"{'oracle p50 (ms)':>16} {'speedup p50':>12} "
        f"{'peak mem (KB)':>14} {'oracle agree%':>14} "
        f"{'present%':>9} {'absent%':>8} {'unknown%':>9}"
    )
    print("-" * 180)
    for r in results:
        print(
            f"{r['label']:<30} {r['n_self']:>6} {r['n_ext']:>6} "
            f"{r['prod_p50_ms']:>14.3f} {r['prod_p95_ms']:>14.3f} "
            f"{r['oracle_p50_ms']:>16.3f} {r['speedup_p50']:>12.2f}x "
            f"{r['peak_mem_kb']:>14.1f} {r['oracle_agree_pct']:>14.1f}% "
            f"{r['avg_present_pct']:>8.1f}% {r['avg_absent_pct']:>7.1f}% "
            f"{r['avg_unknown_pct']:>8.1f}%"
        )
    print()

    # Final assertion
    failed = [r for r in results if r["oracle_agree_pct"] < 100.0]
    if failed:
        print(f"ERROR: {len(failed)} configuration(s) did NOT achieve 100% oracle agreement:")
        for r in failed:
            print(f"  {r['label']}: {r['oracle_agree_pct']:.1f}%")
        sys.exit(1)
    else:
        print("All configurations achieve 100% oracle agreement.")


# ===========================================================================
# Real PDB benchmark (optional)
# ===========================================================================


def _run_real_pdb_benchmark(
    processed_dir: str,
    raw_dir: str,
    n_samples: int = 30,
    reps: int = 3,
) -> None:
    """
    Benchmark on real .pt + raw PDB files.

    Expects:
      - processed_dir: directory of .pt files produced by pdb_data.py
      - raw_dir: directory of raw PDB/CIF files

    Skipped gracefully if files cannot be found.
    """
    import torch
    from pathlib import Path

    processed = Path(processed_dir)
    raw = Path(raw_dir)

    pt_files = sorted(processed.glob("*.pt"))[:n_samples]
    if not pt_files:
        print(f"No .pt files found in {processed_dir}, skipping real-PDB benchmark.")
        return

    times = []
    distributions = []
    skipped = 0

    for pt_path in pt_files:
        stem = pt_path.stem
        raw_path = None
        for ext in (".pdb", ".cif", ".ent"):
            candidate = raw / f"{stem}{ext}"
            if candidate.exists():
                raw_path = candidate
                break
        if raw_path is None:
            skipped += 1
            continue

        try:
            graph = torch.load(pt_path, map_location="cpu")
            chains = list(getattr(graph, "chains", ["A"]))
            resids = list(getattr(graph, "residue_id", []))

            # Read raw deposit
            import sys
            sys.path.insert(0, str(_repo_root))
            from proteinfoundation.graphein_utils.graphein_utils import read_pdb_to_dataframe
            full_df = read_pdb_to_dataframe(path=str(raw_path))

            rep_times = []
            for _ in range(reps):
                t, result = _time_fn(
                    compute_ext_lig_from_df,
                    full_df,
                    chains,
                    resids,
                )
                rep_times.append(t * 1000)

            times.append(np.mean(rep_times))
            distributions.append(_label_distribution(result))

        except Exception as e:
            print(f"  Skipping {stem}: {e}")
            skipped += 1

    if not times:
        print("No real-PDB samples could be processed.")
        return

    times = np.array(times)
    print(f"\n--- Real PDB Benchmark ({len(times)} structures, {skipped} skipped) ---")
    print(f"  p50 latency:  {np.percentile(times, 50):.3f} ms")
    print(f"  p95 latency:  {np.percentile(times, 95):.3f} ms")
    print(f"  max latency:  {np.max(times):.3f} ms")
    avg_present = np.mean([d["present_%"] for d in distributions])
    avg_absent = np.mean([d["absent_%"] for d in distributions])
    avg_unknown = np.mean([d["unknown_%"] for d in distributions])
    print(
        f"  label distribution: present={avg_present:.1f}% "
        f"absent={avg_absent:.1f}% unknown={avg_unknown:.1f}%"
    )


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark ext_lig preprocessing.")
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of repetitions per configuration (default: 10)",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip oracle agreement check (faster for very large configs)",
    )
    parser.add_argument(
        "--real-pdb-dir",
        default=None,
        help="Directory of .pt files for real-PDB benchmark (optional)",
    )
    parser.add_argument(
        "--raw-pdb-dir",
        default=None,
        help="Directory of raw PDB/CIF files for real-PDB benchmark (optional)",
    )
    parser.add_argument(
        "--n-real",
        type=int,
        default=30,
        help="Number of real PDB structures to benchmark (default: 30)",
    )
    args = parser.parse_args()

    # Synthetic benchmark configurations
    # (label, n_self_residues, n_external_atoms)
    configs = [
        ("monomer (no ext atoms)",    100,    0),
        ("short chain, small lig",     50,   20),
        ("short chain, large lig",     50, 1000),
        ("medium chain, medium lig",  200,  200),
        ("medium chain, large lig",   200, 5000),
        ("long chain, no ext atoms",  500,    0),
        ("long chain, medium lig",    500,  500),
        ("long chain, large complex", 500, 5000),
        ("XL chain, XL complex",     1000, 10000),
    ]

    print("=" * 180)
    print("ext_lig preprocessing benchmark")
    print(f"  reps per config : {args.reps}")
    print(f"  oracle verify   : {not args.no_oracle}")
    print("=" * 180)

    results = _run_synthetic_benchmark(
        configs,
        reps=args.reps,
        verify_oracle=not args.no_oracle,
    )
    _print_results(results)

    # Real-PDB benchmark (optional)
    if args.real_pdb_dir and args.raw_pdb_dir:
        _run_real_pdb_benchmark(
            processed_dir=args.real_pdb_dir,
            raw_dir=args.raw_pdb_dir,
            n_samples=args.n_real,
            reps=args.reps,
        )


if __name__ == "__main__":
    main()
