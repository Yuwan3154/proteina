#!/usr/bin/env python3
"""
Shared SLURM/LLsub sharding utilities for pipeline scripts.

Provides:
- resolve_shard_args: Single source of truth for SLURM/LLsub env var detection
- add_shard_args: Add --shard_index, --num_shards to argparse
- shard_proteins: Sort by length, round-robin distribute across shards
- build_shard_cli_args: Build CLI args for propagation to sub-scripts
- wait_for_completion: Shard 0 polls until check_fn(name) is True for all names
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def resolve_shard_args(shard_index: Optional[int], num_shards: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """
    Resolve shard_index and num_shards from args or SLURM/LLsub env vars.

    Returns (shard_index, num_shards). Both None when sharding is inactive.
    """
    si = shard_index
    ns = num_shards

    if si is None:
        for var in ("SLURM_ARRAY_TASK_ID", "LLSUB_RANK"):
            if var in os.environ:
                si = int(os.environ[var])
                break
    if ns is None:
        for var in ("SLURM_ARRAY_TASK_COUNT", "LLSUB_SIZE"):
            if var in os.environ:
                ns = int(os.environ[var])
                break

    return (si, ns)


def add_shard_args(parser) -> None:
    """Add --shard_index, --num_shards, and --len_col to an argparse parser."""
    parser.add_argument(
        "--shard_index",
        type=int,
        default=None,
        help="Shard index (0-based). Auto-detected from SLURM_ARRAY_TASK_ID or LLSUB_RANK if not set.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Total number of shards. Auto-detected from SLURM_ARRAY_TASK_COUNT or LLSUB_SIZE if not set.",
    )
    parser.add_argument(
        "--len_col",
        type=str,
        default="length",
        help="CSV column name for protein length used in shard load-balancing (default: length).",
    )


def lengths_from_csv(csv_path: str, id_col: str, len_col: str) -> Optional[Dict[str, int]]:
    """
    Extract a {protein_id: length} mapping from a CSV file.

    Returns None when csv_path is empty/None or len_col is not present in the file,
    allowing callers to fall back to .pt-file-based length loading.
    """
    if not csv_path:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        id_col_stripped = id_col.strip()
        if len_col not in df.columns or id_col_stripped not in df.columns:
            return None
        return dict(
            zip(
                df[id_col_stripped].astype(str).str.strip(),
                df[len_col].fillna(0).astype(int),
            )
        )
    except Exception:
        return None


def load_lengths_from_pt(names: List[str], data_dir: str) -> Dict[str, int]:
    """
    Load protein lengths from pdb_train/processed/{name}.pt files.

    Returns a {name: length} dict; missing entries are silently omitted (length=0 at lookup time).
    """
    import torch
    lengths: Dict[str, int] = {}
    pt_dir = os.path.join(data_dir, "pdb_train", "processed")
    for name in names:
        pt_path = os.path.join(pt_dir, f"{name}.pt")
        if os.path.exists(pt_path):
            try:
                pt = torch.load(pt_path, weights_only=False, map_location="cpu")
                lengths[name] = len(pt.residue_type)
            except Exception:
                pass
    return lengths


def shard_proteins(
    names: List[str],
    shard_index: int,
    num_shards: int,
    lengths: Optional[Dict[str, int]] = None,
    data_dir: Optional[str] = None,
) -> List[str]:
    """
    Partition proteins across shards using Longest Processing Time (LPT) greedy
    bin-packing on a length-squared cost (per-protein work in OpenFold/AF2Rank
    and transformer-based inference scales ~ O(L^2)).

    Algorithm:
      1. Sort proteins by cost descending (longest first), tiebreaker = name.
      2. Iterate sorted proteins; assign each to the bin with the smallest
         running total. Tiebreaker = lowest bin index. This makespan-minimising
         schedule beats round-robin on uneven workloads (LPT bound: 4/3 - 1/3m
         of optimum vs round-robin's worst case ~2x).

    Determinism: same `names` + `num_shards` + `lengths` always yield the same
    partition across all shards.

    Args:
        names: List of protein names.
        shard_index: 0-based shard index.
        num_shards: Total number of shards.
        lengths: Optional pre-computed {name: length}. If None and data_dir given, loads from PT files.
        data_dir: Optional directory containing pdb_train/processed/{name}.pt for length lookup.

    Returns:
        Subset of names assigned to this shard.
    """
    if lengths is None and data_dir:
        lengths = load_lengths_from_pt(names, data_dir)
    if lengths is None:
        lengths = {n: 0 for n in names}

    # Cost ~ const + c * L^2 * log(L), calibrated 2026-05-22 from a clean
    # single-GPU microbenchmark of the AF2Rank/OpenFold model_1_ptm forward pass
    # only (no concurrent shards, no CPU/IO contention, recycles=6, deepspeed
    # evo attn disabled). 45 reps across L = 64, 96, 128, 192, 256, 320, 384,
    # 448, 512 with per-rep std < 0.5%. OLS feature-selection across the basis
    # {1, sqrt L, log L, L, L log L, L^1.5, L^2, L^2 log L, L^3} (manual VIF +
    # forward/backward AIC stepwise) found:
    #   - L^2 * log(L) is the single best basis term: const + L^2 log L gives
    #     R^2 = 0.998 and the lowest single-feature AIC (716).
    #   - Adding more terms hits extreme multicollinearity (VIF ~ 1e15) and only
    #     improves R^2 from 0.998 to 0.9994.
    #   - The fitted forward-pass time (ms): t(L) = 6084 + 0.1101 * L^2 * log(L).
    #   - This empirically matches "FlashAttention-style memory-bandwidth-bound
    #     attention with a logarithmic chunking factor" -- the per-L^2 cost is
    #     amplified by log L because each O(L^2) attention chunk also touches
    #     O(log L) memory tiers per element.
    # Sanity check vs production-scrape predictions: predicted per-protein
    # STEP 4 time (32 forwards) is 4.2 min @ L=62, 12.9 min @ L=178, 91 min @
    # L=492 -- in line with observed wall-clocks.
    import math as _math
    _C0_PER_TEMPLATE = 6084.3         # ms intercept (per-template fixed overhead)
    _C_L2LOGL        = 0.11010        # ms per L^2 * log(L)
    def _cost(name: str) -> float:
        L = lengths.get(name, 0)
        if L <= 0:
            return 0.0
        return _C0_PER_TEMPLATE + _C_L2LOGL * L * L * _math.log(L)

    sorted_names = sorted(names, key=lambda p: (-_cost(p), p))

    bins: List[List[str]] = [[] for _ in range(num_shards)]
    bin_totals: List[int] = [0] * num_shards
    for name in sorted_names:
        # Smallest total wins; ties broken by lowest bin index for determinism.
        target = min(range(num_shards), key=lambda i: (bin_totals[i], i))
        bins[target].append(name)
        bin_totals[target] += _cost(name)

    shard_names = bins[shard_index]
    total_residues = sum(lengths.get(p, 0) for p in shard_names)
    max_residues = max((lengths.get(p, 0) for p in shard_names), default=0)
    logger.info(
        f"Shard {shard_index}/{num_shards}: {len(shard_names)} proteins, "
        f"~{total_residues} total residues, max protein length ~{max_residues}"
    )
    if shard_index == 0 and num_shards:
        nonzero_totals = [t for t in bin_totals if t > 0]
        if nonzero_totals:
            imbalance_ratio = max(nonzero_totals) / min(nonzero_totals)
            logger.info(
                "LPT bin-packing shard cost (const + c*L^2*log(L)) totals: min=%.1f max=%.1f ratio=%.3f",
                min(nonzero_totals), max(nonzero_totals), imbalance_ratio,
            )
    return shard_names


def build_shard_cli_args(
    shard_index: Optional[int],
    num_shards: Optional[int],
    len_col: str = "length",
) -> List[str]:
    """
    Build CLI args for propagation to sub-scripts.

    Returns ['--shard_index', '0', '--num_shards', '4', '--len_col', 'length']
    when sharding active, else [].  len_col is always included so sub-scripts
    use the same length column even when sharding is inactive.
    """
    args = ["--len_col", len_col]
    if shard_index is None or num_shards is None:
        return args
    return ["--shard_index", str(shard_index), "--num_shards", str(num_shards)] + args


def default_progress_check_workers() -> int:
    return min(32, (os.cpu_count() or 1) * 4)


def filter_proteins_threaded(
    protein_ids: List[str],
    is_complete_fn: Callable[[str], bool],
    max_workers: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Return (needing_work, already_done) using threads for I/O-heavy progress checks.

    Any exception raised by ``is_complete_fn`` for a protein is caught, logged at
    WARNING level, and the protein is placed in *needing_work* so the per-protein
    retry logic (with its own error handling) can deal with it.  A single bad
    protein must not crash the entire step.
    """
    from concurrent.futures import ThreadPoolExecutor

    workers = max_workers if max_workers is not None and max_workers > 0 else default_progress_check_workers()
    needing_work: List[str] = []
    already_done: List[str] = []

    def _safe_check(protein_id: str) -> bool:
        try:
            return is_complete_fn(protein_id)
        except Exception as exc:
            logger.warning(
                "Completion check raised for %s (%s: %s) – treating as needing work.",
                protein_id, type(exc).__name__, exc,
            )
            return False

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for protein_id, complete in zip(protein_ids, executor.map(_safe_check, protein_ids)):
            if complete:
                already_done.append(protein_id)
            else:
                needing_work.append(protein_id)
    return needing_work, already_done


def step_sentinel_path(
    output_dir: Union[str, Path],
    step_name: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    return Path(output_dir) / f".step_{step_name}_shard_{shard_index}_of_{num_shards}_complete"


def wait_for_step(
    output_dir: Union[str, Path],
    step_name: str,
    num_shards: int,
    shard_index: int,
    success: bool,
    poll_interval: int = 60,
    timeout: int = 86400,
) -> bool:
    """Write this shard's step sentinel and wait for peer step sentinels."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sentinel = step_sentinel_path(output_path, step_name, shard_index, num_shards)
    sentinel.write_text("0" if success else "1")
    logger.info(
        "Wrote step sentinel %s for shard %d/%d (%s).",
        step_name, shard_index, num_shards, "success" if success else "failure",
    )

    other_shards = [idx for idx in range(num_shards) if idx != shard_index]
    if other_shards:
        def _peer_done(peer_idx: int) -> bool:
            return step_sentinel_path(output_path, step_name, peer_idx, num_shards).exists()

        if not wait_for_completion(
            other_shards,
            _peer_done,
            poll_interval=poll_interval,
            timeout=timeout,
            item_name=f"{step_name} step shards",
        ):
            return False

    failed_shards = []
    for peer_idx in range(num_shards):
        peer_sentinel = step_sentinel_path(output_path, step_name, peer_idx, num_shards)
        if peer_sentinel.read_text().strip() != "0":
            failed_shards.append(peer_idx)
    if failed_shards:
        logger.error("Step %s reported failure on shard(s): %s", step_name, failed_shards)
        return False
    logger.info("All shards completed step %s successfully.", step_name)
    return True


def wait_for_completion(
    names: List[str],
    check_fn: Callable[[str], bool],
    poll_interval: int = 60,
    timeout: int = 86400,
    item_name: str = "items",
) -> bool:
    """
    Poll until check_fn(name) returns True for all names.

    Intended for shard 0 to wait for all shards to finish before running aggregation.

    Args:
        names: Full list of names to wait for (e.g. shard indices or protein IDs).
        check_fn: Function that returns True when an item is complete.
        poll_interval: Seconds between polls.
        timeout: Max seconds to wait.
        item_name: Human-readable label used in log messages (e.g. "shards", "proteins").

    Returns:
        True if all complete within timeout, False otherwise.
    """
    import time
    start = time.time()
    remaining = set(names)
    while remaining and (time.time() - start) < timeout:
        done = [n for n in remaining if check_fn(n)]
        for n in done:
            remaining.discard(n)
        if remaining:
            logger.info(f"Waiting for {len(remaining)} {item_name}... ({len(names) - len(remaining)}/{len(names)} done)")
            time.sleep(poll_interval)
    if remaining:
        logger.error(f"Timeout: {len(remaining)} {item_name} still incomplete: {list(remaining)[:5]}...")
        return False
    logger.info(f"All {item_name} complete.")
    return True
