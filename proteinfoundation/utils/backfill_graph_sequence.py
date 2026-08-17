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
from typing import Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proteinfoundation.utils.constants import residue_type_to_sequence  # noqa: E402

TMP_SUFFIX = ".seqtmp"


SUCCESS = ("written", "already_correct", "ok", "would_write")


def _one(path: str, dry_run: bool) -> Tuple[str, str]:
    # A .pt that cannot be read at all must not take the whole pass down on file 200,000 of
    # 307,000; the status is recorded per path instead, and --audit-skips re-examines every one.
    try:
        g = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return "load_failed", f"{type(e).__name__}: {e}"
    rt = getattr(g, "residue_type", None)
    if rt is None:
        return "no_residue_type", ""
    seq = residue_type_to_sequence(rt)
    coords = getattr(g, "coords", None)
    if coords is not None and int(coords.shape[0]) != len(seq):
        return "length_mismatch", f"coords={int(coords.shape[0])} residue_type={len(seq)}"
    if getattr(g, "sequence", None) == seq:
        return "already_correct", ""
    if dry_run:
        return "would_write", ""
    g.sequence = seq
    tmp = path + TMP_SUFFIX
    try:
        torch.save(g, tmp)
        os.rename(tmp, path)
    except Exception as e:
        # The original is still intact -- nothing is replaced until the rename succeeds.
        return "write_failed", f"{type(e).__name__}: {e}"
    return "written", ""


def _verify_one(path: str) -> Tuple[str, str]:
    try:
        g = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return "load_failed", f"{type(e).__name__}: {e}"
    rt = getattr(g, "residue_type", None)
    seq = getattr(g, "sequence", None)
    if rt is None:
        return "no_residue_type", ""
    if seq is None:
        return "missing", ""
    if not isinstance(seq, str):
        return "not_a_string", type(seq).__name__
    if seq != residue_type_to_sequence(rt):
        return "wrong", f"stored={seq[:24]}... derived={residue_type_to_sequence(rt)[:24]}..."
    coords = getattr(g, "coords", None)
    if coords is not None and int(coords.shape[0]) != len(seq):
        return "length_mismatch", f"coords={int(coords.shape[0])} sequence={len(seq)}"
    return "ok", ""


def _audit_one(path: str) -> Tuple[str, str]:
    """Re-examine one skipped path and report the facts behind the skip.

    Counting skips is not the same as knowing they were correctly skipped, so every recorded skip
    is reopened here and described by what is actually on the graph.
    """
    try:
        g = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return "unreadable", f"{type(e).__name__}: {e}  size={os.path.getsize(path)}B"
    attrs = sorted(k for k in dir(g) if not k.startswith("_"))
    have = [k for k in ("residue_type", "coords", "coord_mask", "sequence",
                        "contact_map_confind", "dssp_target") if getattr(g, k, None) is not None]
    shapes = {k: tuple(getattr(g, k).shape) for k in have
              if isinstance(getattr(g, k), torch.Tensor)}
    seq = getattr(g, "sequence", None)
    return "readable", (f"present={have} shapes={shapes} "
                        f"sequence={'str len ' + str(len(seq)) if isinstance(seq, str) else seq} "
                        f"n_attrs={len(attrs)}")


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
    parser.add_argument("--skip-log", default="", help="where every non-success path is recorded")
    parser.add_argument("--audit-skips", default="",
                        help="re-open every path in this skip log and report why it was skipped")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    if args.audit_skips:
        audit(args.audit_skips, args.workers)
        return

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
    skip_log = open(args.skip_log, "w") if args.skip_log else None
    counts = Counter()
    # Reading and rewriting a few hundred thousand files off the pool is I/O bound, exactly like
    # the topology index build; parallel workers overlap those waits.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        it = ex.map(fn, paths) if args.verify else ex.map(fn, paths, [args.dry_run] * len(paths))
        for j, (status, detail) in enumerate(it):
            counts[status] += 1
            if log is not None and status in ("written", "already_correct"):
                log.write(paths[j] + "\n")
            if skip_log is not None and status not in SUCCESS:
                skip_log.write(f"{paths[j]}\t{status}\t{detail}\n")
            if (j + 1) % 20000 == 0:
                print(f"  {j + 1}/{len(paths)}  {dict(counts)}", flush=True)
                for f in (log, skip_log):
                    if f is not None:
                        f.flush()
    for f in (log, skip_log):
        if f is not None:
            f.close()

    print("\nresult:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v}")
    bad = sum(v for k, v in counts.items() if k not in SUCCESS)
    print(f"\n{'CLEAN' if bad == 0 else f'{bad} files need attention'}")
    if bad and args.skip_log:
        print(f"every one is listed in {args.skip_log}; audit them with "
              f"--audit-skips {args.skip_log}")


def audit(skip_log: str, workers: int) -> None:
    """Reopen every recorded skip and report what is actually on those graphs."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(skip_log) if ln.strip()]
    print(f"auditing {len(rows)} skipped paths from {skip_log}\n")
    by_status = Counter(r[1] for r in rows)
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  recorded status {k:<20} {v}")
    print()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_audit_one, [r[0] for r in rows]))
    for (path, status, detail), (kind, facts) in zip(
        [(r[0], r[1], r[2] if len(r) > 2 else "") for r in rows], results
    ):
        print(f"{status:<18} {kind:<11} {os.path.basename(path)}")
        if detail:
            print(f"    recorded: {detail}")
        print(f"    on disk:  {facts}")
    print(f"\naudited {len(rows)} paths, {sum(1 for k, _ in results if k == 'unreadable')} unreadable")


if __name__ == "__main__":
    main()
