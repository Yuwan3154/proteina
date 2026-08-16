#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Add ``graph.sequence`` to processed .pt files written before the pipeline emitted it.

The processing pipeline now stores a one-letter sequence on every graph it builds, derived from
``residue_type`` so it agrees with ``coords`` by construction. Graphs processed earlier have no
such attribute, and this backfills them in place.

Two properties matter because this rewrites a tree that other training runs read:
  * each file is replaced by an atomic rename from a temporary written in the same directory, so a
    concurrent reader sees either the old graph or the new one, never a partial file;
  * the pass is idempotent and resumable -- a file whose sequence already matches its residue_type
    is left untouched, and ``--resume`` skips paths a previous run recorded as done without
    reopening them, which is what makes a restart cheap on a tree this size.

Usage:
    python -m proteinfoundation.utils.backfill_graph_sequence \
        --processed-dir $DATA_PATH/pdb_train/processed \
        --done-log $DATA_PATH/pdb_train/backfill_sequence.done --workers 32
"""

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proteinfoundation.utils.constants import residue_type_to_sequence  # noqa: E402

TMP_SUFFIX = ".seqtmp"


def _one(path: str, dry_run: bool) -> str:
    g = torch.load(path, map_location="cpu", weights_only=False)
    rt = getattr(g, "residue_type", None)
    if rt is None:
        return "no_residue_type"
    seq = residue_type_to_sequence(rt)
    coords = getattr(g, "coords", None)
    if coords is not None and int(coords.shape[0]) != len(seq):
        return "length_mismatch"
    if getattr(g, "sequence", None) == seq:
        return "already_correct"
    if dry_run:
        return "would_write"
    g.sequence = seq
    tmp = path + TMP_SUFFIX
    torch.save(g, tmp)
    os.rename(tmp, path)
    return "written"


def _verify_one(path: str) -> str:
    g = torch.load(path, map_location="cpu", weights_only=False)
    rt = getattr(g, "residue_type", None)
    seq = getattr(g, "sequence", None)
    if rt is None:
        return "no_residue_type"
    if seq is None:
        return "missing"
    if not isinstance(seq, str):
        return "not_a_string"
    if seq != residue_type_to_sequence(rt):
        return "wrong"
    coords = getattr(g, "coords", None)
    if coords is not None and int(coords.shape[0]) != len(seq):
        return "length_mismatch"
    return "ok"


def collect_paths(root: Path, resume: set) -> list:
    paths, stale = [], 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(TMP_SUFFIX):
                stale += 1
                continue
            if not f.endswith(".pt"):
                continue
            p = os.path.join(dirpath, f)
            if p not in resume:
                paths.append(p)
    if stale:
        print(f"note: {stale} orphaned {TMP_SUFFIX} files from an interrupted run", flush=True)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--done-log", default="")
    parser.add_argument("--resume", action="store_true", help="skip paths listed in --done-log")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="check only, write nothing")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    done = set()
    if args.resume and args.done_log and os.path.exists(args.done_log):
        done = {line.rstrip("\n") for line in open(args.done_log)}
        print(f"resuming: {len(done)} paths already recorded", flush=True)

    paths = collect_paths(Path(args.processed_dir), done)
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"{len(paths)} files to process, workers={args.workers}, "
          f"dry_run={args.dry_run}, verify={args.verify}", flush=True)

    fn = _verify_one if args.verify else _one
    log = open(args.done_log, "a") if (args.done_log and not args.verify and not args.dry_run) else None
    counts = Counter()
    # Reading and rewriting a few hundred thousand files off the pool is I/O bound, exactly like
    # the topology index build; parallel workers overlap those waits.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        it = ex.map(fn, paths) if args.verify else ex.map(fn, paths, [args.dry_run] * len(paths))
        for j, status in enumerate(it):
            counts[status] += 1
            if log is not None and status in ("written", "already_correct"):
                log.write(paths[j] + "\n")
            if (j + 1) % 20000 == 0:
                print(f"  {j + 1}/{len(paths)}  {dict(counts)}", flush=True)
                if log is not None:
                    log.flush()
    if log is not None:
        log.close()

    print("\nresult:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")
    bad = sum(v for k, v in counts.items() if k not in ("written", "already_correct", "ok", "would_write"))
    print(f"\n{'CLEAN' if bad == 0 else f'{bad} files need attention'}")


if __name__ == "__main__":
    main()
