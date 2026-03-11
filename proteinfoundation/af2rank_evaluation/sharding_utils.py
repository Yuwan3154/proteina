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
from typing import Callable, Dict, List, Optional, Tuple

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
    """Add --shard_index and --num_shards to an argparse parser."""
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


def shard_proteins(
    names: List[str],
    shard_index: int,
    num_shards: int,
    lengths: Optional[Dict[str, int]] = None,
    data_dir: Optional[str] = None,
) -> List[str]:
    """
    Sort proteins by length (ascending) and distribute via round-robin.

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
        import torch
        lengths = {}
        pt_dir = os.path.join(data_dir, "pdb_train", "processed")
        for name in names:
            pt_path = os.path.join(pt_dir, f"{name}.pt")
            if os.path.exists(pt_path):
                pt = torch.load(pt_path, weights_only=False, map_location="cpu")
                lengths[name] = len(pt.residue_type)
            else:
                lengths[name] = 0
    elif lengths is None:
        lengths = {n: 0 for n in names}

    sorted_names = sorted(names, key=lambda p: lengths.get(p, 0))
    shard_names = sorted_names[shard_index::num_shards]
    total_residues = sum(lengths.get(p, 0) for p in shard_names)
    logger.info(
        f"Shard {shard_index}/{num_shards}: {len(shard_names)} proteins, ~{total_residues} total residues"
    )
    return shard_names


def build_shard_cli_args(shard_index: Optional[int], num_shards: Optional[int]) -> List[str]:
    """
    Build CLI args for propagation to sub-scripts.

    Returns ['--shard_index', '0', '--num_shards', '4'] when sharding active, else [].
    """
    if shard_index is None or num_shards is None:
        return []
    return ["--shard_index", str(shard_index), "--num_shards", str(num_shards)]


def wait_for_completion(
    names: List[str],
    check_fn: Callable[[str], bool],
    poll_interval: int = 60,
    timeout: int = 86400,
) -> bool:
    """
    Poll until check_fn(name) returns True for all names.

    Intended for shard 0 to wait for all shards to finish before running aggregation.

    Args:
        names: Full list of protein names (all shards).
        check_fn: Function that returns True when a protein is complete.
        poll_interval: Seconds between polls.
        timeout: Max seconds to wait.

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
            logger.info(f"Waiting for {len(remaining)} proteins... ({len(names) - len(remaining)}/{len(names)} done)")
            time.sleep(poll_interval)
    if remaining:
        logger.error(f"Timeout: {len(remaining)} proteins still incomplete: {list(remaining)[:5]}...")
        return False
    logger.info("All proteins complete.")
    return True
