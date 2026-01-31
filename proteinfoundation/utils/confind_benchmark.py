#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

import argparse
import csv
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb


def _load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _effective_length(graph) -> int:
    coord_mask = getattr(graph, "coord_mask", None)
    if coord_mask is None:
        return int(graph.coords.shape[0])
    return int(coord_mask.any(dim=-1).sum().item())


def _write_graph_pdb(graph, pdb_path: Path) -> None:
    coords = graph.coords
    coord_mask = getattr(graph, "coord_mask", None)
    residue_type = getattr(graph, "residue_type", None)
    chain_index = getattr(graph, "chains", None)

    if residue_type is None:
        raise ValueError("Graph missing residue_type; cannot write PDB for confind.")
    if coord_mask is None:
        raise ValueError("Graph missing coord_mask; cannot write PDB for confind.")

    coords = coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    coord_mask = coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]

    write_prot_to_pdb(
        coords.detach().cpu().numpy(),
        str(pdb_path),
        aatype=residue_type.detach().cpu().numpy(),
        atom37_mask=coord_mask.detach().cpu().numpy(),
        chain_index=chain_index.detach().cpu().numpy() if chain_index is not None else None,
        overwrite=True,
        no_indexing=True,
    )


def _run_confind(
    pdb_path: Path,
    output_path: Path,
    rotlib_path: str,
    confind_bin: str,
    omp_threads: int,
) -> float:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    start = time.perf_counter()
    subprocess.run(
        [
            confind_bin,
            "--p",
            str(pdb_path),
            "--o",
            str(output_path),
            "--rLib",
            rotlib_path,
            "--ren",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return time.perf_counter() - start


def _select_stratified_samples(
    processed_dir: Path,
    min_len: int,
    max_len: int,
    bins: int,
    seed: int,
    max_trials: int,
) -> List[Tuple[str, int, Path]]:
    rng = random.Random(seed)
    files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]
    rng.shuffle(files)

    edges = np.linspace(min_len, max_len, bins + 1)
    selected: Dict[int, Tuple[str, int, Path]] = {}
    trials = 0
    for fname in files:
        if len(selected) >= bins or trials >= max_trials:
            break
        trials += 1
        path = processed_dir / fname
        try:
            graph = _load_graph(path)
        except Exception:
            continue
        length = _effective_length(graph)
        if length < min_len or length > max_len:
            continue
        bin_idx = int(np.searchsorted(edges, length, side="right") - 1)
        bin_idx = min(max(bin_idx, 0), bins - 1)
        if bin_idx not in selected:
            selected[bin_idx] = (fname, length, path)

    if len(selected) < bins:
        raise RuntimeError(
            f"Only found {len(selected)} bins after {trials} trials; try raising max_trials."
        )
    return [selected[i] for i in sorted(selected.keys())]


def _benchmark_omp_threads(
    samples: List[Tuple[str, int, Path]],
    rotlib_path: str,
    confind_bin: str,
    omp_threads_list: List[int],
) -> Tuple[int, List[Dict[str, float]]]:
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_paths = {}
        for sample_id, _, sample_path in samples:
            graph = _load_graph(sample_path)
            pdb_path = tmpdir / f"{sample_id}.pdb"
            _write_graph_pdb(graph, pdb_path)
            pdb_paths[sample_id] = pdb_path

        for omp_threads in omp_threads_list:
            for sample_id, length, _ in samples:
                runtime = _run_confind(
                    pdb_paths[sample_id],
                    tmpdir / f"{sample_id}_t{omp_threads}.contacts",
                    rotlib_path,
                    confind_bin,
                    omp_threads,
                )
                results.append(
                    {
                        "sample_id": sample_id,
                        "length": length,
                        "omp_threads": omp_threads,
                        "runtime_sec": runtime,
                    }
                )

    by_threads: Dict[int, List[float]] = {}
    for row in results:
        by_threads.setdefault(int(row["omp_threads"]), []).append(
            float(row["runtime_sec"])
        )
    best_threads = min(by_threads.items(), key=lambda kv: np.mean(kv[1]))[0]
    return best_threads, results


def _run_confind_job(
    sample_id: str,
    length: int,
    sample_path: Path,
    rotlib_path: str,
    confind_bin: str,
    omp_threads: int,
) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        graph = _load_graph(sample_path)
        pdb_path = tmpdir / f"{sample_id}.pdb"
        _write_graph_pdb(graph, pdb_path)
        runtime = _run_confind(
            pdb_path,
            tmpdir / f"{sample_id}.contacts",
            rotlib_path,
            confind_bin,
            omp_threads,
        )
    return {
        "sample_id": sample_id,
        "length": length,
        "omp_threads": omp_threads,
        "runtime_sec": runtime,
    }


def _benchmark_workers(
    samples: List[Tuple[str, int, Path]],
    rotlib_path: str,
    confind_bin: str,
    omp_threads: int,
    workers_list: List[int],
) -> Tuple[int, List[Dict[str, float]]]:
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    def _run_async(worker_count: int) -> Tuple[float, List[Dict[str, float]]]:
        items = sorted(samples, key=lambda x: x[1], reverse=True)
        items_iter = iter(items)
        rows: List[Dict[str, float]] = []
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = set()
            while len(futures) < worker_count:
                try:
                    sample_id, length, sample_path = next(items_iter)
                except StopIteration:
                    break
                futures.add(
                    executor.submit(
                        _run_confind_job,
                        sample_id,
                        length,
                        sample_path,
                        rotlib_path,
                        confind_bin,
                        omp_threads,
                    )
                )

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    rows.append(future.result())
                    try:
                        sample_id, length, sample_path = next(items_iter)
                    except StopIteration:
                        continue
                    futures.add(
                        executor.submit(
                            _run_confind_job,
                            sample_id,
                            length,
                            sample_path,
                            rotlib_path,
                            confind_bin,
                            omp_threads,
                        )
                    )
        wall_time = time.perf_counter() - start
        return wall_time, rows

    results = []
    by_workers = {}
    for workers in workers_list:
        wall_time, rows = _run_async(workers)
        by_workers[workers] = {
            "wall_time_sec": wall_time,
            "samples_per_sec": len(samples) / wall_time if wall_time > 0 else 0.0,
        }
        for row in rows:
            row["workers"] = workers
            row["wall_time_sec"] = wall_time
            results.append(row)

    best_workers = max(by_workers.items(), key=lambda kv: kv[1]["samples_per_sec"])[0]
    return best_workers, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        required=True,
        help="Path to processed .pt directory.",
    )
    parser.add_argument(
        "--rotlib",
        required=True,
        help="Path to rotamer library directory (EBL.out/BEBL.out).",
    )
    parser.add_argument("--confind-bin", default="confind")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-trials", type=int, default=8000)
    parser.add_argument("--omp-threads-list", default="1,2,4")
    parser.add_argument("--workers-list", default="1,2,4,8")
    parser.add_argument(
        "--output-dir",
        default="confind_benchmark",
        help="Output directory for csv/plots.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omp_threads_list = [
        int(x) for x in args.omp_threads_list.split(",") if x.strip()
    ]
    workers_list = [int(x) for x in args.workers_list.split(",") if x.strip()]

    samples = _select_stratified_samples(
        processed_dir=processed_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        bins=args.bins,
        seed=args.seed,
        max_trials=args.max_trials,
    )

    # Use short/mid/long samples for omp benchmark
    thread_samples = [samples[0], samples[len(samples) // 2], samples[-1]]
    best_omp_threads, omp_rows = _benchmark_omp_threads(
        samples=thread_samples,
        rotlib_path=args.rotlib,
        confind_bin=args.confind_bin,
        omp_threads_list=omp_threads_list,
    )

    omp_csv = output_dir / "omp_benchmark.csv"
    with open(omp_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "length", "omp_threads", "runtime_sec"]
        )
        writer.writeheader()
        writer.writerows(omp_rows)

    best_workers, worker_rows = _benchmark_workers(
        samples=thread_samples,
        rotlib_path=args.rotlib,
        confind_bin=args.confind_bin,
        omp_threads=best_omp_threads,
        workers_list=workers_list,
    )

    workers_csv = output_dir / "worker_benchmark.csv"
    with open(workers_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "length",
                "omp_threads",
                "runtime_sec",
                "workers",
                "wall_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(worker_rows)

    # Benchmark runtime vs length with best settings
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    results = []
    items = sorted(samples, key=lambda x: x[1], reverse=True)
    items_iter = iter(items)
    with ProcessPoolExecutor(max_workers=best_workers) as executor:
        futures = set()
        while len(futures) < best_workers:
            try:
                sample_id, length, sample_path = next(items_iter)
            except StopIteration:
                break
            futures.add(
                executor.submit(
                    _run_confind_job,
                    sample_id,
                    length,
                    sample_path,
                    args.rotlib,
                    args.confind_bin,
                    best_omp_threads,
                )
            )
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
                try:
                    sample_id, length, sample_path = next(items_iter)
                except StopIteration:
                    continue
                futures.add(
                    executor.submit(
                        _run_confind_job,
                        sample_id,
                        length,
                        sample_path,
                        args.rotlib,
                        args.confind_bin,
                        best_omp_threads,
                    )
                )

    results_csv = output_dir / "runtime_vs_length.csv"
    with open(results_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "length", "omp_threads", "runtime_sec"]
        )
        writer.writeheader()
        writer.writerows(results)

    lengths = np.array([r["length"] for r in results], dtype=float)
    runtimes = np.array([r["runtime_sec"] for r in results], dtype=float)
    plt.figure(figsize=(6, 4))
    plt.scatter(lengths, runtimes, s=40)
    plt.xlabel("Chain length")
    plt.ylabel("Confind runtime (s)")
    plt.title("Confind runtime vs chain length")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / "runtime_vs_length.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    n_files = len([f for f in os.listdir(processed_dir) if f.endswith(".pt")])
    avg_runtime = float(np.mean(runtimes))
    cpu_count = os.cpu_count() or 1

    # Estimate throughput from worker benchmark (best_workers)
    worker_wall_times = [
        row["wall_time_sec"] for row in worker_rows if row["workers"] == best_workers
    ]
    wall_time = float(np.mean(worker_wall_times)) if worker_wall_times else 1.0
    samples_per_sec = len(thread_samples) / wall_time if wall_time > 0 else 0.0
    est_total_sec = n_files / samples_per_sec if samples_per_sec > 0 else 0.0

    summary = {
        "best_omp_threads": best_omp_threads,
        "best_workers": best_workers,
        "avg_runtime_sec": avg_runtime,
        "num_files": n_files,
        "cpu_count": cpu_count,
        "estimated_workers": best_workers,
        "estimated_total_hours": est_total_sec / 3600.0,
        "samples": results,
    }
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Best OMP threads: {best_omp_threads}")
    print(f"Best workers: {best_workers}")
    print(f"Average runtime: {avg_runtime:.3f}s")
    print(
        f"Estimated total hours (workers={best_workers}): {est_total_sec/3600:.2f}"
    )
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {omp_csv}")
    print(f"Wrote: {workers_csv}")


if __name__ == "__main__":
    main()
