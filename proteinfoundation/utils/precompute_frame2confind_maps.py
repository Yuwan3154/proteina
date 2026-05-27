#!/usr/bin/env python3
"""GPU-accelerated ConFind contact map precompute using Frame2ConFind.

Drop-in replacement for the CPU-based ``precompute_confind_maps.py``.  Stores
continuous probability maps as ``contact_map_confind`` (float16) in each
processed ``.pt`` file — the same attribute and dtype used by the CPU pipeline,
so the existing ``contact_method: confind`` transform reads them unchanged.

Example
-------
::

    python -m proteinfoundation.utils.precompute_frame2confind_maps \\
        --checkpoint ~/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/best.pt \\
        --processed-dir ~/proteina/data/pdb_train/processed \\
        --batch-size 4 \\
        --max-len 384 \\
        --amp-dtype bf16

"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, "/home/ubuntu")

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.frame2confind_utils import _place_cb

_DEFAULT_CHECKPOINT = (
    "~/Frame2ConFind/runs/f2s_ft_max384_pair_ebs16_no-sin-pos-emb/best.pt"
)


def _log(msg: str, log_path: Optional[Path] = None) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def _load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _graph_to_f2s_item(graph) -> Optional[Dict[str, torch.Tensor]]:
    """Extract Frame2seq coords from a proteina graph.

    Returns None if the graph has no valid coordinates.
    """
    coords = getattr(graph, "coords", None)
    coord_mask = getattr(graph, "coord_mask", None)
    if coords is None:
        return None

    idx = PDB_TO_OPENFOLD_INDEX_TENSOR
    coords = coords[:, idx, :]
    if coord_mask is not None:
        coord_mask = coord_mask[:, idx]

    L = coords.shape[0]
    # Frame2seq ordering: N=0, CA=1, C=2, CB=3, O=4
    # OpenFold ordering: N=0, CA=1, C=2, CB=3, O=4 — first 5 atoms match
    x_f2s = coords[:, [0, 1, 2, 3, 4], :].clone()

    if coord_mask is not None:
        mask = coord_mask[:, 1].bool()  # CA
        cb_valid = coord_mask[:, 3].bool()  # CB in OpenFold
        missing_cb = ~cb_valid & mask
    else:
        mask = torch.ones(L, dtype=torch.bool)
        missing_cb = torch.zeros(L, dtype=torch.bool)

    if missing_cb.any():
        cb_placed = _place_cb(
            coords[missing_cb, 0], coords[missing_cb, 1], coords[missing_cb, 2]
        )
        x_f2s[missing_cb, 3, :] = cb_placed

    return {"x_f2s": x_f2s, "mask": mask, "length": L}


def _collate_items(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length items into a padded batch."""
    B = len(items)
    Lmax = max(it["length"] for it in items)
    x_f2s = torch.zeros(B, Lmax, 5, 3, dtype=torch.float32)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, it in enumerate(items):
        L = it["length"]
        x_f2s[i, :L] = it["x_f2s"]
        mask[i, :L] = it["mask"]
        lengths[i] = L
    return {"x_f2s": x_f2s, "mask": mask, "lengths": lengths}


def _iter_processed(processed_dir: Path, max_len: int) -> List[Tuple[Path, int]]:
    """Enumerate processed .pt files with lengths, filtering by max_len.

    Uses ``rglob`` so the sharded layout (d_FS processed/<bucket>/*.pt) is
    walked transparently. Flat dirs incur no extra cost.
    """
    items = []
    for path in sorted(Path(processed_dir).rglob("*.pt")):
        if not path.is_file():
            continue
        try:
            graph = _load_graph(path)
        except Exception:
            continue
        coords = getattr(graph, "coords", None)
        if coords is None:
            continue
        L = coords.shape[0]
        if L > max_len:
            continue
        items.append((path, L))
    return items


def run_precompute(
    checkpoint: str,
    processed_dir: str,
    batch_size: int = 4,
    max_len: int = 384,
    amp_dtype: str = "bf16",
    compile_model: bool = False,
    overwrite: bool = False,
    limit: int = 0,
    status_csv: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, float]:
    """Run GPU-accelerated ConFind precompute.

    Args:
        checkpoint: Path to Frame2ConFind best.pt.
        processed_dir: Path to proteina processed .pt directory.
        batch_size: Batch size for GPU inference.
        max_len: Maximum residue length to process.
        amp_dtype: AMP precision (bf16, fp16, fp32).
        compile_model: Whether to torch.compile the model.
        overwrite: Whether to overwrite existing contact_map_confind.
        limit: Max number of files to process (0 = no limit).
        status_csv: Optional path to write per-file status CSV.
        log_path: Optional path to write log file.

    Returns:
        Dictionary of summary statistics.
    """
    from Frame2ConFind.inference.api import Frame2ConFindPredictor

    processed_dir = Path(processed_dir)
    checkpoint = str(Path(checkpoint).expanduser())
    log_file = Path(log_path) if log_path else None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    _log(f"Loading model from {checkpoint}", log_file)
    predictor = Frame2ConFindPredictor(
        checkpoint=checkpoint,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
    )
    _log(f"Model loaded on {predictor.device}", log_file)

    # Enumerate and filter files
    _log(f"Scanning {processed_dir} for .pt files (max_len={max_len})", log_file)
    all_items = _iter_processed(processed_dir, max_len)
    _log(f"Found {len(all_items)} files within max_len={max_len}", log_file)

    # Filter already-processed files
    if not overwrite:
        pending = []
        skipped = 0
        for path, length in all_items:
            try:
                graph = _load_graph(path)
                if (
                    hasattr(graph, "contact_map_confind")
                    and graph.contact_map_confind is not None
                ):
                    skipped += 1
                    continue
            except Exception:
                pass
            pending.append((path, length))
        _log(f"Skipping {skipped} files with existing contact_map_confind", log_file)
    else:
        pending = all_items

    # Sort by length descending (longest first for efficient batching)
    pending.sort(key=lambda x: x[1], reverse=True)

    if limit > 0:
        pending = pending[:limit]
    _log(f"Processing {len(pending)} files", log_file)

    # CSV writer for status
    csv_writer = None
    csv_handle = None
    if status_csv:
        csv_path = Path(status_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=["file", "length", "status", "infer_sec", "save_sec", "error"],
        )
        csv_writer.writeheader()

    processed_count = 0
    failed_count = 0
    total_infer_sec = 0.0
    total_save_sec = 0.0
    total_residues = 0
    start_time = time.perf_counter()

    # Process in batches
    idx = 0
    while idx < len(pending):
        batch_paths = []
        batch_items = []
        batch_graphs = []

        # Build batch
        while len(batch_items) < batch_size and idx < len(pending):
            path, length = pending[idx]
            idx += 1
            try:
                graph = _load_graph(path)
                item = _graph_to_f2s_item(graph)
                if item is None:
                    if csv_writer:
                        csv_writer.writerow(
                            {
                                "file": path.name,
                                "length": length,
                                "status": "failed",
                                "infer_sec": 0.0,
                                "save_sec": 0.0,
                                "error": "no_coords",
                            }
                        )
                    failed_count += 1
                    continue
                batch_paths.append(path)
                batch_items.append(item)
                batch_graphs.append(graph)
            except Exception as e:
                if csv_writer:
                    csv_writer.writerow(
                        {
                            "file": path.name,
                            "length": length,
                            "status": "failed",
                            "infer_sec": 0.0,
                            "save_sec": 0.0,
                            "error": str(e),
                        }
                    )
                failed_count += 1
                continue

        if not batch_items:
            continue

        # Collate and run inference
        collated = _collate_items(batch_items)
        infer_start = time.perf_counter()
        probs = predictor.predict_batch(
            collated["x_f2s"], collated["mask"]
        )  # [B, Lmax, Lmax]
        infer_sec = time.perf_counter() - infer_start

        # Save results back to .pt files
        save_start = time.perf_counter()
        for i, (path, graph) in enumerate(zip(batch_paths, batch_graphs)):
            L = int(collated["lengths"][i].item())
            # Extract [L, L] and store as float16 (matching CPU precompute format)
            contact_probs = probs[i, :L, :L].contiguous().to(dtype=torch.float16)
            graph.contact_map_confind = contact_probs
            try:
                tmp_path = path.with_suffix(".pt.tmp")
                torch.save(graph, tmp_path)
                tmp_path.rename(path)
                if csv_writer:
                    csv_writer.writerow(
                        {
                            "file": path.name,
                            "length": L,
                            "status": "processed",
                            "infer_sec": infer_sec / len(batch_items),
                            "save_sec": 0.0,  # filled below
                            "error": "",
                        }
                    )
                processed_count += 1
                total_residues += L
            except Exception as e:
                if csv_writer:
                    csv_writer.writerow(
                        {
                            "file": path.name,
                            "length": L,
                            "status": "failed",
                            "infer_sec": infer_sec / len(batch_items),
                            "save_sec": 0.0,
                            "error": str(e),
                        }
                    )
                failed_count += 1

        save_sec = time.perf_counter() - save_start
        total_infer_sec += infer_sec
        total_save_sec += save_sec

        # Progress log
        done = processed_count + failed_count
        elapsed = time.perf_counter() - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = len(pending) - done
        eta = remaining / rate if rate > 0 else 0.0
        if done % max(1, 10 * batch_size) < batch_size or done == len(pending):
            _log(
                f"Progress {done}/{len(pending)} "
                f"processed={processed_count} failed={failed_count} "
                f"rate={rate:.1f}/s eta={eta / 60:.1f}m "
                f"batch_infer={infer_sec:.3f}s batch_save={save_sec:.3f}s",
                log_file,
            )

    if csv_handle is not None:
        csv_handle.close()

    elapsed = time.perf_counter() - start_time
    stats = {
        "processed": processed_count,
        "failed": failed_count,
        "total_infer_sec": total_infer_sec,
        "total_save_sec": total_save_sec,
        "total_residues": total_residues,
        "wall_time_sec": elapsed,
        "proteins_per_sec": processed_count / elapsed if elapsed > 0 else 0.0,
        "residues_per_sec": total_residues / total_infer_sec if total_infer_sec > 0 else 0.0,
    }
    _log(
        f"Done: processed={processed_count} failed={failed_count} "
        f"wall_time={elapsed:.1f}s proteins/sec={stats['proteins_per_sec']:.1f} "
        f"residues/sec={stats['residues_per_sec']:.0f}",
        log_file,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated ConFind contact map precompute using Frame2ConFind."
    )
    parser.add_argument(
        "--checkpoint",
        default=_DEFAULT_CHECKPOINT,
        help="Path to Frame2ConFind best.pt checkpoint.",
    )
    parser.add_argument(
        "--processed-dir",
        default="/home/ubuntu/proteina/data/pdb_train/processed",
        help="Path to proteina processed .pt directory.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument(
        "--amp-dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile for the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing contact_map_confind in .pt files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0 = no limit).",
    )
    parser.add_argument("--status-csv", default=None)
    parser.add_argument("--log", default=None, dest="log_path")
    args = parser.parse_args()

    run_precompute(
        checkpoint=args.checkpoint,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        amp_dtype=args.amp_dtype,
        compile_model=args.compile,
        overwrite=args.overwrite,
        limit=args.limit,
        status_csv=args.status_csv,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
