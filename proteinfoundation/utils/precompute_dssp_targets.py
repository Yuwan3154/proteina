#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""
Offline DSSP target pre-computation for Proteina training datasets.

Reads each processed .pt file, runs pydssp on the backbone atoms, and stores
the resulting dssp_target tensor back into the file.  After this script has
run, DSSPTargetTransform short-circuits and skips the pydssp call at training
time, removing per-batch CPU overhead.

Coordinate-ordering note
------------------------
Processed .pt files store coordinates in raw PDB ordering (N=0, CA=1, C=2,
O=3).  The OpenFold reorder (N=0, CA=1, C=2, CB=3, O=4) is applied in
pdb_data.py:get() at load time, *after* this script runs.  Therefore this
script extracts N/CA/C/O using indices [0, 1, 2, 3] — NOT [0, 1, 2, 4].

Usage examples
--------------
# All files in a processed/ directory:
python precompute_dssp_targets.py --processed-dir /data/pdb/processed

# Resolve processed-dir from a training config:
python precompute_dssp_targets.py --config configs/experiment_config/training_*.yaml

# SLURM array job (automatic sharding via SLURM_ARRAY_TASK_ID):
python precompute_dssp_targets.py --processed-dir /data/pdb/processed --use-slurm

# Manual sharding (4-way):
python precompute_dssp_targets.py --processed-dir /data/pdb/processed --shard-id 0 --num-shards 4
"""

import argparse
import csv
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------

def _iter_processed_files(processed_dir: Path) -> Iterable[Path]:
    """Yield all .pt files under ``processed_dir``, walking sub-dirs.

    Originally a flat-dir listing; switched to ``rglob`` so the sharded
    layout (e.g. d_FS processed/<bucket>/*.pt) is handled transparently.
    Flat directories incur no real cost since rglob just enumerates the
    single level.
    """
    for p in Path(processed_dir).rglob("*.pt"):
        if p.is_file():
            yield p


def _read_skip_csv(path: Path) -> Tuple[Set[str], List[Dict[str, str]]]:
    """Read a previous-run output CSV and return its filenames + raw rows.

    Returns:
        (skip_names, inherited_rows)
        - skip_names: set of ``file`` values to filter out before dispatch.
        - inherited_rows: list of full row dicts in original order. Written
          verbatim to the head of a fresh output CSV so the new CSV is a
          strict superset of the input (chains across SLURM allocations).
    """
    if not path.exists():
        raise FileNotFoundError(f"--skip-from-csv path not found: {path}")
    names: Set[str] = set()
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        if "file" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} is missing the 'file' column; "
                "expected the dssp_precompute.csv format."
            )
        for row in reader:
            fname = row.get("file")
            if not fname:
                continue
            names.add(fname)
            rows.append(row)
    return names, rows


# ---------------------------------------------------------------------------
# DSSP computation (PDB-ordered coords: O at index 3)
# ---------------------------------------------------------------------------

def _compute_dssp(graph) -> Optional[torch.Tensor]:
    """Run pydssp on PDB-ordered backbone coords and return [L] long tensor.

    Values: 0=loop, 1=helix, 2=strand, -1=invalid (missing backbone atoms).
    Returns None if coords are absent or have fewer than 4 atoms (CA-only).

    This function is called on raw .pt graphs where coords are in PDB ordering
    (N=0, CA=1, C=2, O=3).  Do NOT call it on OpenFold-reordered coords
    (where O=4); use compute_dssp_target(..., coord_layout="atom37") for that.
    """
    import pydssp  # imported inside worker to avoid pickling issues

    coords = getattr(graph, "coords", None)
    if coords is None or coords.shape[1] < 4:
        return None

    coord_mask = getattr(graph, "coord_mask", None)

    # PDB ordering: N=0, CA=1, C=2, O=3
    ncao = coords[:, [0, 1, 2, 3], :]   # [L, 4, 3]
    ncao_batch = ncao.unsqueeze(0)       # [1, L, 4, 3]

    dssp_out = pydssp.assign(ncao_batch, out_type="index")  # [1, L]
    dssp_target = dssp_out[0].long().clone()                # [L]

    # Mask residues with missing backbone atoms.
    # coord_mask may be float (fill value 1e-5, valid > 0.5) or bool.
    if coord_mask is not None and coord_mask.shape[1] >= 4:
        valid = (
            (coord_mask[:, 0] > 0.5)
            & (coord_mask[:, 1] > 0.5)
            & (coord_mask[:, 2] > 0.5)
            & (coord_mask[:, 3] > 0.5)
        )
        dssp_target = torch.where(valid, dssp_target, torch.full_like(dssp_target, -1))

    return dssp_target


# ---------------------------------------------------------------------------
# Per-file worker (must be a module-level function for multiprocessing)
# ---------------------------------------------------------------------------

def _process_one(path: Path, overwrite: bool) -> Dict:
    """Load one .pt file, compute DSSP, and write it back atomically."""
    start = time.perf_counter()

    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "file": path.name,
            "status": "failed",
            "error": f"load: {exc}",
            "load_sec": time.perf_counter() - start,
            "dssp_sec": 0.0,
            "save_sec": 0.0,
            "total_sec": time.perf_counter() - start,
        }

    load_end = time.perf_counter()

    if getattr(graph, "coords", None) is None:
        return {
            "file": path.name,
            "status": "failed",
            "error": "missing_coords",
            "load_sec": load_end - start,
            "dssp_sec": 0.0,
            "save_sec": 0.0,
            "total_sec": load_end - start,
        }

    if getattr(graph, "dssp_target", None) is not None and not overwrite:
        return {
            "file": path.name,
            "status": "skipped",
            "error": "",
            "load_sec": load_end - start,
            "dssp_sec": 0.0,
            "save_sec": 0.0,
            "total_sec": load_end - start,
        }

    try:
        dssp_start = time.perf_counter()
        dssp_target = _compute_dssp(graph)
        dssp_end = time.perf_counter()

        if dssp_target is None:
            return {
                "file": path.name,
                "status": "skipped",
                "error": "too_few_atoms",
                "load_sec": load_end - start,
                "dssp_sec": dssp_end - dssp_start,
                "save_sec": 0.0,
                "total_sec": dssp_end - start,
            }

        graph.dssp_target = dssp_target

        save_start = time.perf_counter()
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save(graph, tmp_path)
        tmp_path.rename(path)
        save_end = time.perf_counter()

        return {
            "file": path.name,
            "status": "processed",
            "error": "",
            "load_sec": load_end - start,
            "dssp_sec": dssp_end - dssp_start,
            "save_sec": save_end - save_start,
            "total_sec": save_end - start,
        }

    except Exception as exc:
        return {
            "file": path.name,
            "status": "failed",
            "error": str(exc),
            "load_sec": load_end - start,
            "dssp_sec": 0.0,
            "save_sec": 0.0,
            "total_sec": time.perf_counter() - start,
        }


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_message(msg: str, log_path: Optional[Path] = None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Async processing pool
# ---------------------------------------------------------------------------

def _process_async(
    items: List[Path],
    workers: int,
    overwrite: bool,
    log_path: Optional[Path],
    log_every: int,
    log_every_sec: float,
    result_callback=None,
) -> Tuple[List[Dict], Dict]:
    total = len(items)
    completed = processed = skipped = failed = 0
    dssp_sum = total_sum = 0.0
    last_log = start = time.perf_counter()
    results: List[Dict] = []

    _log_message(f"Starting: total={total} workers={workers}", log_path=log_path)

    items_iter = iter(items)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        future_to_path: Dict = {}

        # Seed the pool
        for _ in range(workers):
            try:
                path = next(items_iter)
            except StopIteration:
                break
            f = executor.submit(_process_one, path, overwrite)
            futures.add(f)
            future_to_path[f] = path

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                except Exception as exc:
                    path = future_to_path.get(future)
                    result = {
                        "file": path.name if path is not None else "unknown",
                        "status": "failed",
                        "error": str(exc),
                        "load_sec": 0.0,
                        "dssp_sec": 0.0,
                        "save_sec": 0.0,
                        "total_sec": 0.0,
                    }

                status = result["status"]
                if status == "processed":
                    processed += 1
                    dssp_sum += float(result.get("dssp_sec", 0.0))
                    total_sum += float(result.get("total_sec", 0.0))
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    _log_message(
                        f"Failed {result['file']}: {result.get('error', '')}",
                        log_path=log_path,
                    )

                if result_callback is not None:
                    result_callback(result)
                results.append(result)
                completed += 1

                now = time.perf_counter()
                if (completed % log_every == 0) or (now - last_log >= log_every_sec):
                    elapsed = now - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_sec = (total - completed) / rate if rate > 0 else 0.0
                    _log_message(
                        f"Progress {completed}/{total} processed={processed} "
                        f"skipped={skipped} failed={failed} "
                        f"rate={rate:.2f}/s eta={eta_sec/60.0:.1f}m",
                        log_path=log_path,
                    )
                    last_log = now

                # Submit next item
                try:
                    path = next(items_iter)
                except StopIteration:
                    continue
                f = executor.submit(_process_one, path, overwrite)
                futures.add(f)
                future_to_path[f] = path

    stats = {
        "total": completed,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "dssp_sum": dssp_sum,
        "total_sum": total_sum,
    }
    return results, stats


# ---------------------------------------------------------------------------
# SLURM sharding helper
# ---------------------------------------------------------------------------

def _resolve_shard(
    shard_id: Optional[int],
    num_shards: Optional[int],
    use_slurm: bool,
) -> Tuple[int, int]:
    if num_shards is not None:
        return (shard_id or 0), num_shards
    if shard_id is not None:
        raise ValueError("--shard-id requires --num-shards.")

    if use_slurm:
        array_id = os.getenv("SLURM_ARRAY_TASK_ID")
        array_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if array_id and array_count:
            return int(array_id), int(array_count)
        proc_id = os.getenv("SLURM_PROCID")
        n_tasks = os.getenv("SLURM_NTASKS")
        if proc_id and n_tasks:
            return int(proc_id), int(n_tasks)
        raise ValueError("--use-slurm set but SLURM variables not found.")

    return 0, 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_precompute(
    processed_dir: Path,
    workers: Optional[int] = None,
    overwrite: bool = False,
    limit: int = 0,
    offset: int = 0,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
    use_slurm: bool = False,
    output_csv: str = "dssp_precompute.csv",
    log_path: Optional[Path] = None,
    log_every: int = 200,
    log_every_sec: float = 60.0,
    skip_from_csv: Optional[Path] = None,
) -> Dict:
    if workers is None:
        workers = int(os.getenv("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

    shard_id, num_shards = _resolve_shard(shard_id, num_shards, use_slurm)

    output_csv_path = Path(output_csv)
    if num_shards > 1:
        output_csv_path = output_csv_path.with_name(
            f"{output_csv_path.stem}_shard{shard_id}_of_{num_shards}{output_csv_path.suffix}"
        )
    log_path = log_path or output_csv_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _log_message(
        f"precompute_dssp_targets: processed_dir={processed_dir} "
        f"workers={workers} shard={shard_id}/{num_shards} overwrite={overwrite}",
        log_path=log_path,
    )

    skip_names: Set[str] = set()
    inherited_rows: List[Dict[str, str]] = []
    if skip_from_csv is not None:
        skip_names, inherited_rows = _read_skip_csv(skip_from_csv)
        _log_message(
            f"Skip-from-csv: {skip_from_csv} -> {len(skip_names)} filenames to skip "
            f"({len(inherited_rows)} rows inherited)",
            log_path=log_path,
        )

    files = list(_iter_processed_files(processed_dir))
    _log_message(f"Found {len(files)} .pt files", log_path=log_path)

    if num_shards > 1:
        files = [f for i, f in enumerate(files) if i % num_shards == shard_id]
        _log_message(f"After sharding: {len(files)} files", log_path=log_path)

    if offset:
        files = files[offset:]
    if limit and limit > 0:
        files = files[:limit]

    if skip_names:
        before = len(files)
        files = [f for f in files if f.name not in skip_names]
        _log_message(
            f"After skip-from-csv: {len(files)} files ({before - len(files)} filtered)",
            log_path=log_path,
        )
    _log_message(f"Processing {len(files)} files", log_path=log_path)

    # Open streaming CSV. When --skip-from-csv is given AND the output CSV is
    # fresh (and is a *different* file from the skip CSV), copy the inherited
    # rows in first so the new CSV is a strict superset and can be chained as
    # the next allocation's skip list.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_output = (not output_csv_path.exists()) or output_csv_path.stat().st_size == 0
    same_in_out = (
        skip_from_csv is not None
        and output_csv_path.exists()
        and skip_from_csv.resolve() == output_csv_path.resolve()
    )
    need_inherit = bool(inherited_rows) and fresh_output and not same_in_out

    csv_handle = open(output_csv_path, "a", newline="")
    fieldnames = ["file", "status", "load_sec", "dssp_sec", "save_sec", "total_sec", "error"]
    csv_writer = csv.DictWriter(csv_handle, fieldnames=fieldnames, extrasaction="ignore")
    if fresh_output:
        csv_writer.writeheader()
    if need_inherit:
        for row in inherited_rows:
            csv_writer.writerow(row)
        csv_handle.flush()
        _log_message(
            f"Wrote {len(inherited_rows)} inherited rows into fresh {output_csv_path}",
            log_path=log_path,
        )

    def _write_row(result: Dict) -> None:
        csv_writer.writerow(result)
        csv_handle.flush()

    wall_start = time.perf_counter()
    _, stats = _process_async(
        files,
        workers=workers,
        overwrite=overwrite,
        log_path=log_path,
        log_every=log_every,
        log_every_sec=log_every_sec,
        result_callback=_write_row,
    )
    csv_handle.close()
    wall_sec = time.perf_counter() - wall_start

    n = stats["processed"]
    avg_dssp = stats["dssp_sum"] / n if n else 0.0
    avg_total = stats["total_sum"] / n if n else 0.0

    _log_message(
        f"Done: processed={stats['processed']} skipped={stats['skipped']} "
        f"failed={stats['failed']} / total={stats['total']}",
        log_path=log_path,
    )
    _log_message(f"Wall time: {wall_sec:.1f}s", log_path=log_path)
    _log_message(f"Avg dssp time (processed): {avg_dssp:.3f}s", log_path=log_path)
    _log_message(f"Avg total time (processed): {avg_total:.3f}s", log_path=log_path)
    _log_message(f"Output CSV: {output_csv_path}", log_path=log_path)

    return {
        "processed": stats["processed"],
        "skipped": stats["skipped"],
        "failed": stats["failed"],
        "total": stats["total"],
        "avg_dssp_sec": avg_dssp,
        "avg_total_sec": avg_total,
        "wall_sec": wall_sec,
        "output_csv": str(output_csv_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute DSSP targets and store them into processed .pt files."
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Path to the dataset's processed/ directory.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Hydra experiment config; resolves processed_dir from datamodule.data_dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: all CPUs / SLURM_CPUS_PER_TASK).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if dssp_target already exists in the file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N files (0 = no limit; useful for testing).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N files.",
    )
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Auto-detect shard from SLURM_ARRAY_TASK_ID / SLURM_PROCID.",
    )
    parser.add_argument(
        "--output-csv",
        default="dssp_precompute.csv",
        help="Path for the per-file progress CSV.",
    )
    parser.add_argument("--log-path", default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Log progress every N completions.",
    )
    parser.add_argument(
        "--log-every-sec",
        type=float,
        default=60.0,
        help="Also log every N seconds.",
    )
    parser.add_argument(
        "--skip-from-csv",
        default=None,
        help="Previous run's progress CSV; every file listed there is skipped "
             "this run. If the output CSV is fresh, it is initialized with all "
             "rows from this CSV so the new CSV stays a strict superset and "
             "can chain into the next allocation as --skip-from-csv.",
    )
    args = parser.parse_args()

    if args.config is None and args.processed_dir is None:
        parser.error("Provide --config or --processed-dir.")

    processed_dir = None
    if args.config:
        cfg = OmegaConf.load(Path(args.config))
        data_dir = OmegaConf.select(cfg, "datamodule.data_dir")
        if data_dir is not None:
            processed_dir = Path(data_dir) / "processed"

    if args.processed_dir:
        processed_dir = Path(args.processed_dir)

    if processed_dir is None:
        parser.error("Could not resolve processed_dir. Provide --processed-dir.")

    run_precompute(
        processed_dir=processed_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        limit=args.limit,
        offset=args.offset,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        use_slurm=args.use_slurm,
        output_csv=args.output_csv,
        log_path=Path(args.log_path) if args.log_path else None,
        log_every=args.log_every,
        log_every_sec=args.log_every_sec,
        skip_from_csv=Path(args.skip_from_csv) if args.skip_from_csv else None,
    )


if __name__ == "__main__":
    main()
