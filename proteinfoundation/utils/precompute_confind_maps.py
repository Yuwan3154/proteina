#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import argparse
import csv
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from proteinfoundation.utils.confind_utils import confind_raw_contact_map


def _iter_processed_files(processed_dir: Path) -> Iterable[Path]:
    for name in os.listdir(processed_dir):
        if name.endswith(".pt"):
            yield processed_dir / name


def _load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _get_length(path: Path) -> Tuple[Path, int]:
    graph = _load_graph(path)
    coord_mask = getattr(graph, "coord_mask", None)
    if coord_mask is not None:
        length = int(coord_mask.any(dim=-1).sum().item())
    else:
        length = int(graph.coords.shape[0])
    return path, length


def _log_message(message: str, log_path: Optional[Path] = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as handle:
            handle.write(line + "\n")


def _read_length_cache(cache_path: Path) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    if not cache_path.exists():
        return lengths
    with open(cache_path, "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("file")
            length = row.get("length")
            if name and length is not None:
                lengths[name] = int(length)
    return lengths


def _write_length_cache(cache_path: Path, items: List[Tuple[Path, int]]) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "length"])
        writer.writeheader()
        for path, length in items:
            writer.writerow({"file": path.name, "length": length})
    tmp_path.replace(cache_path)


def _acquire_lock(lock_path: Path, timeout_sec: float) -> bool:
    start = time.time()
    backoff = 0.5
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            return True
        except FileExistsError:
            if (time.time() - start) > timeout_sec:
                return False
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 10.0)


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _collect_lengths_async(paths: List[Path], workers: int) -> List[Tuple[Path, int]]:
    results: List[Tuple[Path, int]] = []
    paths_iter = iter(paths)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        while len(futures) < workers:
            try:
                path = next(paths_iter)
            except StopIteration:
                break
            futures.add(executor.submit(_get_length, path))

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                try:
                    path = next(paths_iter)
                except StopIteration:
                    continue
                futures.add(executor.submit(_get_length, path))
    return results


def _process_one(
    path: Path,
    rotlib_path: str,
    confind_bin: str,
    omp_threads: int,
    overwrite: bool,
) -> Dict[str, float]:
    start = time.perf_counter()
    graph = _load_graph(path)
    load_end = time.perf_counter()

    if hasattr(graph, "contact_map_confind") and graph.contact_map_confind is not None:
        if not overwrite:
            return {
                "file": path.name,
                "status": "skipped",
                "load_sec": load_end - start,
                "confind_sec": 0.0,
                "save_sec": 0.0,
                "total_sec": load_end - start,
            }

    confind_start = time.perf_counter()
    raw_map = confind_raw_contact_map(
        graph,
        rotlib_path=rotlib_path,
        confind_bin=confind_bin,
        omp_threads=omp_threads,
        renumber=True,
    )
    confind_end = time.perf_counter()

    graph.contact_map_confind = torch.as_tensor(raw_map, dtype=torch.float16)

    save_start = time.perf_counter()
    torch.save(graph, path)
    save_end = time.perf_counter()

    return {
        "file": path.name,
        "status": "processed",
        "load_sec": load_end - start,
        "confind_sec": confind_end - confind_start,
        "save_sec": save_end - save_start,
        "total_sec": save_end - start,
    }


def _process_async(
    items: List[Tuple[Path, int]],
    workers: int,
    rotlib_path: str,
    confind_bin: str,
    omp_threads: int,
    overwrite: bool,
    log_path: Optional[Path],
    log_every: int,
    log_every_sec: float,
    result_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    collect_results: bool = True,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    results: List[Dict[str, float]] = [] if collect_results else []
    items_iter = iter(items)
    total = len(items)
    completed = 0
    processed = 0
    skipped = 0
    failed = 0
    confind_sum = 0.0
    total_sum = 0.0
    last_log = time.perf_counter()
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        future_to_path = {}
        while len(futures) < workers:
            try:
                path, _ = next(items_iter)
            except StopIteration:
                break
            future = executor.submit(
                _process_one,
                path,
                rotlib_path,
                confind_bin,
                omp_threads,
                overwrite,
            )
            futures.add(future)
            future_to_path[future] = path

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
                        "confind_sec": 0.0,
                        "save_sec": 0.0,
                        "total_sec": 0.0,
                    }
                    failed += 1
                else:
                    if result["status"] == "processed":
                        processed += 1
                        confind_sum += float(result.get("confind_sec", 0.0))
                        total_sum += float(result.get("total_sec", 0.0))
                    elif result["status"] == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                if result_callback is not None:
                    result_callback(result)
                if collect_results:
                    results.append(result)
                completed += 1
                now = time.perf_counter()
                should_log = (completed % log_every == 0) or (
                    now - last_log >= log_every_sec
                )
                if should_log:
                    elapsed = now - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = total - completed
                    eta_sec = remaining / rate if rate > 0 else 0.0
                    _log_message(
                        f"Progress {completed}/{total} processed={processed} "
                        f"skipped={skipped} failed={failed} "
                        f"rate={rate:.3f}/s eta={eta_sec/60.0:.1f}m",
                        log_path=log_path,
                    )
                    last_log = now
                try:
                    path, _ = next(items_iter)
                except StopIteration:
                    continue
                future = executor.submit(
                    _process_one,
                    path,
                    rotlib_path,
                    confind_bin,
                    omp_threads,
                    overwrite,
                )
                futures.add(future)
                future_to_path[future] = path
    stats = {
        "total": completed,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "confind_sum": confind_sum,
        "total_sum": total_sum,
    }
    return results, stats


def _resolve_from_config(config_path: Path) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    cfg = OmegaConf.load(config_path)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    datamodule = cfg_resolved.get("datamodule", {})
    data_dir = datamodule.get("data_dir")
    dataselector = datamodule.get("dataselector")
    rotlib = None
    for t in datamodule.get("transforms", []) or []:
        if isinstance(t, dict) and t.get("_target_") == "proteinfoundation.datasets.transforms.ContactMapTransform":
            rotlib = t.get("confind_rotamer_lib")
            break
    return data_dir, rotlib, dataselector


def _dataselector_identifier(ds: Dict[str, Any]) -> str:
    def _join(vals):
        return "".join(vals) if vals else ""

    return (
        f"df_pdb_f{ds.get('fraction')}_minl{ds.get('min_length')}_maxl{ds.get('max_length')}"
        f"_mt{ds.get('molecule_type')}"
        f"_et{_join(ds.get('experiment_types'))}"
        f"_mino{ds.get('oligomeric_min')}_maxo{ds.get('oligomeric_max')}"
        f"_minr{ds.get('best_resolution')}_maxr{ds.get('worst_resolution')}"
        f"_hl{_join(ds.get('has_ligands'))}"
        f"_rl{_join(ds.get('remove_ligands'))}"
        f"_rnsr{ds.get('remove_non_standard_residues')}"
        f"_rpu{ds.get('remove_pdb_unavailable')}"
        f"_l{_join(ds.get('labels'))}"
        f"_rcu{ds.get('remove_cath_unavailable')}"
        f"_ex{len(ds.get('exclude_ids') or [])}"
    )


def _read_lengths_from_dataset_csv(csv_path: Path) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    if not csv_path.exists():
        return lengths
    with open(csv_path, "r") as handle:
        reader = csv.DictReader(handle)
        if "length" not in reader.fieldnames:
            return lengths
        for row in reader:
            length = row.get("length")
            if length is None:
                continue
            if row.get("id"):
                name = row["id"]
            elif row.get("pdb") and row.get("chain"):
                name = f"{row['pdb']}_{row['chain']}"
            elif row.get("pdb"):
                name = row["pdb"]
            else:
                continue
            lengths[f"{name}.pt"] = int(length)
    return lengths


def _resolve_shard_from_slurm(
    shard_id: Optional[int], num_shards: Optional[int], require_slurm: bool
) -> Tuple[int, int]:
    if num_shards is not None:
        if shard_id is None:
            shard_id = 0
        return shard_id, num_shards
    if shard_id is not None and num_shards is None:
        raise ValueError("Provide --num-shards with --shard-id.")

    array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    array_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
    if array_id is not None and array_count is not None:
        return int(array_id), int(array_count)

    proc_id = os.getenv("SLURM_PROCID")
    n_tasks = os.getenv("SLURM_NTASKS")
    if proc_id is not None and n_tasks is not None:
        return int(proc_id), int(n_tasks)

    if require_slurm:
        raise ValueError("SLURM variables not found; disable --use-slurm.")
    return 0, 1


def run_precompute(
    processed_dir: Path,
    rotlib: str,
    dataselector: Optional[Dict[str, Any]] = None,
    confind_bin: str = "confind",
    omp_threads: Optional[int] = None,
    workers: Optional[int] = None,
    use_slurm: bool = False,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
    limit: int = 0,
    offset: int = 0,
    sort_by_length: bool = True,
    length_cache: Optional[Path] = None,
    length_cache_readonly: bool = False,
    length_cache_timeout: float = 3600.0,
    overwrite: bool = False,
    output_csv: str = "confind_precompute.csv",
    log_path: Optional[Path] = None,
    log_every: int = 50,
    log_every_sec: float = 60.0,
    stream_csv: bool = True,
) -> Dict[str, Any]:
    if omp_threads is None:
        omp_threads = int(os.getenv("OMP_NUM_THREADS") or 1)
    if workers is None:
        workers = int(os.getenv("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

    shard_id, num_shards = _resolve_shard_from_slurm(shard_id, num_shards, use_slurm)

    files = list(_iter_processed_files(processed_dir))
    length_cache = length_cache or (processed_dir / "length_cache.csv")

    output_csv_path = Path(output_csv)
    if num_shards > 1:
        output_csv_path = output_csv_path.with_name(
            f"{output_csv_path.stem}_shard{shard_id}_of_{num_shards}{output_csv_path.suffix}"
        )
    elif output_csv_path.name == "confind_precompute.csv":
        output_csv_path = output_csv_path.with_name(
            f"confind_precompute_shard{shard_id}_of_{num_shards}{output_csv_path.suffix}"
        )
    output_csv = str(output_csv_path)
    log_path = log_path or Path(output_csv).with_suffix(".log")
    if log_path.parent != Path("."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    _log_message(
        f"Precompute start: processed_dir={processed_dir} workers={workers} "
        f"omp_threads={omp_threads} shard={shard_id}/{num_shards} "
        f"length_cache={length_cache}",
        log_path=log_path,
    )

    items: List[Tuple[Path, int]] = []
    if sort_by_length:
        cache_map = _read_length_cache(length_cache)
        if dataselector is not None and not cache_map:
            data_dir = processed_dir.parent
            csv_name = _dataselector_identifier(dataselector) + ".csv"
            dataset_csv = data_dir / csv_name
            dataset_lengths = _read_lengths_from_dataset_csv(dataset_csv)
            if dataset_lengths:
                cache_map.update(dataset_lengths)
                _log_message(
                    f"Loaded {len(dataset_lengths)} lengths from {dataset_csv}",
                    log_path=log_path,
                )
        missing = [path for path in files if path.name not in cache_map]
        if missing:
            if length_cache_readonly:
                raise ValueError(
                    f"length_cache.csv missing {len(missing)} entries and readonly is set."
                )
            lock_path = length_cache.with_suffix(length_cache.suffix + ".lock")
            if _acquire_lock(lock_path, timeout_sec=length_cache_timeout):
                try:
                    _log_message(
                        f"Building length cache for {len(missing)} entries...",
                        log_path=log_path,
                    )
                    new_items = _collect_lengths_async(missing, workers=workers)
                    merged = [
                        (processed_dir / name, length)
                        for name, length in cache_map.items()
                    ]
                    merged.extend(new_items)
                    _write_length_cache(length_cache, merged)
                    cache_map = _read_length_cache(length_cache)
                finally:
                    _release_lock(lock_path)
            else:
                _log_message(
                    "Length cache lock timeout; waiting for cache to appear.",
                    log_path=log_path,
                )
                while not length_cache.exists():
                    time.sleep(5.0)
                cache_map = _read_length_cache(length_cache)

        for path in files:
            length = cache_map.get(path.name)
            if length is None:
                continue
            items.append((path, length))
        items.sort(key=lambda x: x[1], reverse=True)
    else:
        items = [(path, -1) for path in files]

    if num_shards > 1:
        items = [item for idx, item in enumerate(items) if idx % num_shards == shard_id]

    items = items[offset:] if offset else items
    if limit and limit > 0:
        items = items[:limit]

    csv_writer = None
    csv_handle = None
    if stream_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = open(csv_path, "a", newline="")
        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "file",
                "status",
                "load_sec",
                "confind_sec",
                "save_sec",
                "total_sec",
            ],
        )
        if csv_path.stat().st_size == 0:
            csv_writer.writeheader()

    def _write_row(result: Dict[str, float]) -> None:
        if csv_writer is None:
            return
        csv_writer.writerow(result)
        csv_handle.flush()

    start = time.perf_counter()
    results, stats = _process_async(
        items,
        workers=workers,
        rotlib_path=rotlib,
        confind_bin=confind_bin,
        omp_threads=omp_threads,
        overwrite=overwrite,
        log_path=log_path,
        log_every=log_every,
        log_every_sec=log_every_sec,
        result_callback=_write_row if stream_csv else None,
        collect_results=not stream_csv,
    )
    if csv_handle is not None:
        csv_handle.close()

    total_sec = time.perf_counter() - start
    processed_count = stats["processed"]
    avg_confind = stats["confind_sum"] / processed_count if processed_count else 0.0
    avg_total = stats["total_sum"] / processed_count if processed_count else 0.0
    if not stream_csv:
        with open(output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file",
                    "status",
                    "load_sec",
                    "confind_sec",
                    "save_sec",
                    "total_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    _log_message(
        f"Processed files: {processed_count} / {stats['total']}",
        log_path=log_path,
    )
    _log_message(f"Total wall time: {total_sec:.2f}s", log_path=log_path)
    _log_message(
        f"Avg confind time (processed): {avg_confind:.2f}s", log_path=log_path
    )
    _log_message(
        f"Avg total time (processed): {avg_total:.2f}s", log_path=log_path
    )
    _log_message(f"Wrote: {output_csv}", log_path=log_path)

    return {
        "processed": processed_count,
        "total": stats["total"],
        "avg_confind_sec": avg_confind,
        "avg_total_sec": avg_total,
        "total_sec": total_sec,
        "output_csv": output_csv,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--rotlib", default=None)
    parser.add_argument("--confind-bin", default="confind")
    parser.add_argument("--omp-threads", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--use-slurm", action="store_true")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sort-by-length", action="store_true", default=True)
    parser.add_argument("--no-sort-by-length", dest="sort_by_length", action="store_false")
    parser.add_argument("--length-cache", default=None)
    parser.add_argument("--length-cache-readonly", action="store_true")
    parser.add_argument("--length-cache-timeout", type=float, default=3600.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-csv", default="confind_precompute.csv")
    parser.add_argument("--no-stream-csv", dest="stream_csv", action="store_false")
    parser.set_defaults(stream_csv=True)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-every-sec", type=float, default=60.0)
    args = parser.parse_args()

    if args.config is None and args.processed_dir is None:
        raise ValueError("Provide --config or --processed-dir.")

    processed_dir = None
    rotlib = args.rotlib
    dataselector = None
    if args.config:
        data_dir, cfg_rotlib, dataselector = _resolve_from_config(Path(args.config))
        if processed_dir is None and data_dir is not None:
            processed_dir = Path(data_dir) / "processed"
        if rotlib is None:
            rotlib = cfg_rotlib
    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
    if processed_dir is None:
        raise ValueError("Could not resolve processed dir. Provide --processed-dir.")
    if rotlib is None:
        rotlib = os.getenv("MSL_ROTLIB")
    if rotlib is None:
        candidate = os.path.join(processed_dir.parent, "rotlibs")
        if os.path.isdir(candidate):
            rotlib = candidate
    if rotlib is None:
        raise ValueError(
            "Rotamer library path not set; use --rotlib, MSL_ROTLIB, or data_dir/rotlibs."
        )

    run_precompute(
        processed_dir=processed_dir,
        rotlib=rotlib,
        dataselector=dataselector,
        confind_bin=args.confind_bin,
        omp_threads=args.omp_threads,
        workers=args.workers,
        use_slurm=args.use_slurm,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        limit=args.limit,
        offset=args.offset,
        sort_by_length=args.sort_by_length,
        length_cache=Path(args.length_cache) if args.length_cache else None,
        length_cache_readonly=args.length_cache_readonly,
        length_cache_timeout=args.length_cache_timeout,
        overwrite=args.overwrite,
        output_csv=args.output_csv,
        log_path=Path(args.log_path) if args.log_path else None,
        log_every=args.log_every,
        log_every_sec=args.log_every_sec,
        stream_csv=args.stream_csv,
    )


if __name__ == "__main__":
    main()
