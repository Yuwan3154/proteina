#!/usr/bin/env python3
"""Base d_FS preprocessing: raw .pdb -> processed .pt on the sharded tree.

For each AFDB stem in --ids, looks up its bucket in --shard-manifest, then
calls PDBLightningDataModule._process_single_pdb (the staticmethod that
the proteina dataset code uses internally) with:

    raw_dir       = <raw-root>/<bucket>/
    processed_dir = <processed-root>/<bucket>/

This is the right entrypoint because:
- ``_process_single_pdb`` is a @staticmethod, so we don't need to instantiate
  the datamodule (which would otherwise try to glob the flat raw/ tree).
- It already wraps ``read_pdb_to_dataframe`` + ``protein_to_pyg`` + the
  AFDB-specific ext_lig=2 (unknown) computation when database_tag="afdb".

Parallelism: ProcessPoolExecutor with --workers; the inner function is
already CPU-bound and side-effect-free so this scales linearly to a few
dozen cores. Skip-existing logic checks for ``<processed-root>/<bucket>/<stem>.pt``
before dispatching so partial reruns are cheap.

Example:
    salloc --partition=mit_preemptable --cpus-per-task=48 --mem=200G --time=6:00:00
    ssh <node>
    module load miniforge && conda activate cue_openfold
    python -m proteinfoundation.utils.precompute_afdb_base \\
        --ids ~/proteina/data/d_FS/effective_ids.txt \\
        --shard-manifest ~/proteina/data/d_FS/shard_manifest.json \\
        --raw-root ~/proteina/data/d_FS/raw \\
        --processed-root /orcd/compute/so3/001/chenxi/d_FS/processed \\
        --format pdb --workers 48
"""

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make proteina importable.
_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS.parent.parent
sys.path.insert(0, str(_ROOT))

from proteinfoundation.datasets.pdb_data import PDBLightningDataModule  # noqa: E402

sys.path.insert(0, str(_THIS.parent))
from shard_dfs import id_to_bucket  # noqa: E402


def _worker(args):
    stem, bucket, raw_root, processed_root, format_type, store_het, store_bfactor = args
    raw_dir = pathlib.Path(raw_root) / bucket
    processed_dir = pathlib.Path(processed_root) / bucket
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Skip if already processed.
    if (processed_dir / f"{stem}.pt").exists():
        return ("skip", stem)
    try:
        out = PDBLightningDataModule._process_single_pdb(
            index_pdb_tuple=(0, stem),
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            format_type=format_type,
            store_het=store_het,
            store_bfactor=store_bfactor,
            database_tag="afdb",
        )
    except Exception as e:
        return ("fail", f"{stem}: {e}")
    if out is None:
        return ("fail", f"{stem}: returned None")
    return ("ok", stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=pathlib.Path, required=True)
    p.add_argument("--shard-manifest", type=pathlib.Path, required=True)
    p.add_argument("--raw-root", type=pathlib.Path, required=True)
    p.add_argument("--processed-root", type=pathlib.Path, required=True)
    p.add_argument("--format", default="pdb")
    p.add_argument("--store-het", action="store_true")
    p.add_argument("--store-bfactor", action="store_true", default=True)
    p.add_argument("--no-store-bfactor", dest="store_bfactor", action="store_false")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N ids (for smoke tests)")
    p.add_argument("--progress-every", type=int, default=5000)
    args = p.parse_args()

    depth_table = json.loads(args.shard_manifest.read_text())["depth_table"]
    ids = [s.strip() for s in args.ids.read_text().splitlines() if s.strip()]
    if args.limit is not None:
        ids = ids[:args.limit]
    total = len(ids)
    print(f"[{time.strftime('%H:%M:%S')}] dispatching {total} ids "
          f"across {args.workers} workers (limit={args.limit})", flush=True)

    payloads = [
        (s, id_to_bucket(s, depth_table), str(args.raw_root),
         str(args.processed_root), args.format,
         args.store_het, args.store_bfactor)
        for s in ids
    ]

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    fail_path = args.processed_root.parent / "preprocess_failures.txt"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_path, "a") as fail_f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, payload): payload[0] for payload in payloads}
        for i, fut in enumerate(as_completed(futs), 1):
            kind, info = fut.result()
            counts[kind] += 1
            if kind == "fail":
                fail_f.write(info + "\n")
                fail_f.flush()
            if i % args.progress_every == 0:
                rate = i / max(1.0, time.time() - t0)
                eta_s = (total - i) / max(rate, 1e-6)
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{total} "
                      f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} "
                      f"({rate:.1f}/s, ETA {eta_s/60:.1f} min)", flush=True)

    print(f"DONE in {(time.time()-t0)/60:.1f} min: {counts}", flush=True)
    print(f"failures recorded to {fail_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
