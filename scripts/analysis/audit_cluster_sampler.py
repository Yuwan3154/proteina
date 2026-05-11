"""Audit ClusterSampler DDP partitioning + epoch-to-epoch behaviour.

Runs in two passes:
  - LEGACY (v2=False, no set_epoch): expected to expose the bug — ranks visit
    overlapping/skipped clusters because torch.randperm is unsynchronized.
  - V2 (v2=True, set_epoch each epoch): expected to partition cleanly and yield
    different cluster-members across epochs in cluster-random mode.

The "DDP" simulation does NOT spawn separate processes (torch.distributed is
finicky). Instead it instantiates one ClusterSampler per fake rank, monkey-
patches `torch.distributed.is_initialized/get_world_size/get_rank`, and
collects the indices each rank yields. This is sufficient to verify the
shuffle synchronization invariant.

Usage:
    python scripts/analysis/audit_cluster_sampler.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from contextlib import contextmanager
from typing import Dict, List
from unittest.mock import patch

import torch

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

from proteinfoundation.utils.cluster_utils import ClusterSampler  # noqa: E402


class _StubDataset:
    """Minimal stub satisfying ClusterSampler's expectations."""

    def __init__(self, ids: List[str]):
        self.database = "pdb"
        self.file_names = [f"{i}.pt" for i in ids]


def _make_mapping(n_clusters: int, members_per_cluster: int) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for c in range(n_clusters):
        rep = f"c{c}_rep"
        members = [rep] + [f"c{c}_m{m}" for m in range(members_per_cluster - 1)]
        mapping[rep] = members
    return mapping


@contextmanager
def _fake_ddp(world_size: int, rank: int):
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=world_size), \
         patch("torch.distributed.get_rank", return_value=rank):
        yield


def _collect_one_rank(
    mapping: Dict[str, List[str]],
    ids: List[str],
    sampling_mode: str,
    rank: int,
    world_size: int,
    epoch: int,
    v2: bool,
) -> List[int]:
    dataset = _StubDataset(ids)
    sampler = ClusterSampler(
        dataset=dataset,
        clusterid_to_seqid_mapping=mapping,
        sampling_mode=sampling_mode,
        v2=v2,
    )
    sampler.set_epoch(epoch)
    with _fake_ddp(world_size, rank):
        return list(iter(sampler))


def _audit_partitioning(
    mapping: Dict[str, List[str]],
    ids: List[str],
    sampling_mode: str,
    world_size: int,
    epoch: int,
    v2: bool,
):
    per_rank = [
        _collect_one_rank(mapping, ids, sampling_mode, r, world_size, epoch, v2)
        for r in range(world_size)
    ]
    flat = [i for r in per_rank for i in r]
    counts = Counter(flat)
    duplicates = {i: c for i, c in counts.items() if c > 1}
    n_unique_clusters = len(mapping)
    return {
        "per_rank_sizes": [len(r) for r in per_rank],
        "total_yielded": len(flat),
        "unique_yielded": len(set(flat)),
        "duplicate_count": len(duplicates),
        "first_duplicate_examples": dict(list(duplicates.items())[:5]),
        "n_unique_clusters": n_unique_clusters,
    }


def _audit_epoch_to_epoch_diversity(
    mapping: Dict[str, List[str]],
    ids: List[str],
    world_size: int,
    n_epochs: int,
    v2: bool,
):
    """For cluster-random: across epochs, rank 0 should pick different members per cluster."""
    per_epoch_seq_ids: List[List[str]] = []
    id_to_idx = {fn.split(".")[0]: i for i, fn in enumerate([f"{x}.pt" for x in ids])}
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    for ep in range(n_epochs):
        idxs = _collect_one_rank(mapping, ids, "cluster-random", 0, world_size, ep, v2)
        per_epoch_seq_ids.append([idx_to_id[i] for i in idxs])
    same_as_epoch_0 = [
        sum(1 for a, b in zip(per_epoch_seq_ids[0], ep) if a == b)
        for ep in per_epoch_seq_ids
    ]
    return {
        "n_epochs": n_epochs,
        "epoch0_size": len(per_epoch_seq_ids[0]),
        "matches_with_epoch_0": same_as_epoch_0,
    }


def _audit_membership(
    mapping: Dict[str, List[str]],
    ids: List[str],
    sampling_mode: str,
    world_size: int,
    epoch: int,
    v2: bool,
):
    """Every yielded sequence must belong to *some* cluster."""
    member_to_rep = {m: r for r, ms in mapping.items() for m in ms}
    id_to_idx = {fn.split(".")[0]: i for i, fn in enumerate([f"{x}.pt" for x in ids])}
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    bad = []
    for r in range(world_size):
        idxs = _collect_one_rank(mapping, ids, sampling_mode, r, world_size, epoch, v2)
        for i in idxs:
            sid = idx_to_id[i]
            if sid not in member_to_rep:
                bad.append((r, sid))
    return {"violations": len(bad), "examples": bad[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clusters", type=int, default=200)
    ap.add_argument("--members_per_cluster", type=int, default=5)
    ap.add_argument("--world_size", type=int, default=4)
    ap.add_argument("--n_epochs", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("audit_cluster_sampler_report.md"))
    args = ap.parse_args()

    mapping = _make_mapping(args.n_clusters, args.members_per_cluster)
    ids = sorted({m for ms in mapping.values() for m in ms})

    lines = ["# ClusterSampler audit\n"]
    lines.append(
        f"- |clusters|={args.n_clusters}, members_per_cluster={args.members_per_cluster}, "
        f"world_size={args.world_size}, n_epochs={args.n_epochs}\n"
    )

    for v2_flag, label in [(False, "LEGACY (v2=False)"), (True, "V2 (v2=True)")]:
        lines.append(f"\n## {label}\n")
        for mode in ["cluster-reps", "cluster-random"]:
            part = _audit_partitioning(mapping, ids, mode, args.world_size, epoch=0, v2=v2_flag)
            mem = _audit_membership(mapping, ids, mode, args.world_size, epoch=0, v2=v2_flag)
            lines.append(f"### {mode} — partitioning\n")
            lines.append(f"  per-rank sizes: {part['per_rank_sizes']}")
            lines.append(f"  total yielded: {part['total_yielded']}")
            lines.append(f"  unique yielded: {part['unique_yielded']} (clusters={part['n_unique_clusters']})")
            lines.append(f"  duplicates across ranks: {part['duplicate_count']}")
            if part["first_duplicate_examples"]:
                lines.append(f"  duplicate examples: {part['first_duplicate_examples']}")
            lines.append(f"  membership violations: {mem['violations']}")
            lines.append("")

        div = _audit_epoch_to_epoch_diversity(
            mapping, ids, args.world_size, args.n_epochs, v2=v2_flag
        )
        lines.append("### cluster-random — epoch-to-epoch diversity (rank 0)\n")
        lines.append(f"  matches_with_epoch_0 across {div['n_epochs']} epochs: {div['matches_with_epoch_0']}")
        lines.append(
            "  (epoch 0 always matches itself; later epochs should differ — "
            "a near-equal count means the per-epoch RNG isn't refreshing)"
        )

    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[audit_cluster_sampler] wrote {args.out}")


if __name__ == "__main__":
    main()
