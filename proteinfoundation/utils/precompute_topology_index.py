#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Build the retrieval + topology index consumed by TopologyReferenceTransform.

Topology conditioning needs, for every chain: its run-length-compressed DSSP, its SSE-by-SSE
contact map, and the pool of same-cluster chains with a DIFFERENT sequence that may serve as its
template. Resolving any of that at __getitem__ time would mean loading a second .pt per sample and
doubling dataloader I/O, so it is precomputed once here.

Everything is stored as FLAT tensors with offsets rather than a dict of per-chain tensors: a dict
with hundreds of thousands of small Python objects is exactly the shape that triggers the
copy-on-write refcount blow-up in forked dataloader workers (see the CoW note in this repo's
history), whereas a handful of large tensors is shared cleanly.

Usage:
    python -m proteinfoundation.utils.precompute_topology_index \
        --dataset pdb_train_contact-confind_S25_max512_purge-test_cutoff-190828 \
        --out $DATA_PATH/pdb_train/topology_index.pt
"""

import argparse
import gc
import hashlib
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proteinfoundation.datasets.sse_topology import (  # noqa: E402
    DSSP_HELIX,
    DSSP_STRAND,
    dssp_to_runs,
    sse_contact_reference,
)


def _read_one(path: str, min_len: int, threshold: float):
    """Worker: one .pt -> (runs, flattened SSE contact reference, T_he). None runs = unusable."""
    if not os.path.exists(path):
        return None, None, 0
    g = torch.load(path, map_location="cpu", weights_only=False)
    dssp = getattr(g, "dssp_target", None)
    raw = getattr(g, "contact_map_confind", None)
    if dssp is None or bool((dssp < 0).all()) or raw is None:
        return None, None, 0
    runs = dssp_to_runs(dssp, min_len=min_len)
    cm = (raw.float() >= threshold).float()
    ref, keep = sse_contact_reference(cm, runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
    # Returned as raw bytes, not a tensor: torch pickles tensors through shared memory, and one
    # segment per chain exhausts it ("unable to mmap ...: Cannot allocate memory") long before
    # the scan finishes.
    return runs, ref.to(torch.uint8).flatten().numpy().tobytes(), len(keep)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-subdir", default="pdb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=os.environ.get("PROTEINA_REPO", "."))
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0, help="debug: only index N chains")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    cfg_path = os.path.join(
        args.repo, "configs/datasets_config", args.dataset_subdir, args.dataset + ".yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup()

    splits = {}
    for split in ("train", "val", "test"):
        mapping = dm.datasplitter.clusterid_to_seqid_mappings.get(split)
        if mapping:
            splits[split] = mapping

    ids, cluster_of = [], []
    id_to_row = {}
    members_flat, members_offset = [], [0]
    for mapping in splits.values():
        for seqids in mapping.values():
            cluster_row = len(members_offset) - 1
            for s in seqids:
                if s in id_to_row:
                    continue
                id_to_row[s] = len(ids)
                ids.append(s)
                cluster_of.append(cluster_row)
            members_flat.extend(id_to_row[s] for s in seqids)
            members_offset.append(len(members_flat))
    print(f"indexed {len(ids)} chains in {len(members_offset) - 1} clusters", flush=True)

    csv_path = Path(cfg.datamodule.dataselector.data_dir) / f"{dm._get_file_identifier(dm.dataselector)}.csv"
    df = pd.read_csv(csv_path)
    id_to_seq = dict(zip(df["id"].astype(str), df["sequence"].astype(str)))
    print(f"sequences from {csv_path.name}: {len(id_to_seq)}", flush=True)
    n_no_seq = 0

    ds = dm.train_ds
    n = len(ids) if args.limit <= 0 else min(args.limit, len(ids))
    seq_hash = torch.zeros(n, dtype=torch.int64)
    runs_flat, runs_offset = [], [0]
    he_flat, he_offset, he_size = [], [0], torch.zeros(n, dtype=torch.int16)
    n_missing = 0

    # Reading ~300k .pt files off the HDD-backed pool is I/O bound, not compute bound (the
    # single-process version sat in uninterruptible I/O wait at ~100k after an hour). Parallel
    # readers overlap those waits; ex.map preserves order, so the offset arrays stay aligned.
    paths = [str(ds._processed_path_for(f"{ids[i]}.pt")) for i in range(n)]
    # Drop the datamodule before forking: workers only need paths, and inheriting the parent's
    # heap is what broke the pool (the same copy-on-write problem the flat storage avoids).
    del ds, dm, df, id_to_row
    gc.collect()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = []
        for j, r in enumerate(ex.map(_read_one, paths, itertools.repeat(args.min_len),
                                     itertools.repeat(args.contact_threshold), chunksize=32)):
            results.append(r)
            if (j + 1) % 20000 == 0:
                print(f"  {j + 1}/{n}", flush=True)

    for i, (runs, ref_bytes, t_he) in enumerate(results):
        if runs is None:
            n_missing += 1
            runs_offset.append(runs_offset[-1])
            he_offset.append(he_offset[-1])
            continue
        runs_flat.extend(runs)
        runs_offset.append(runs_offset[-1] + len(runs))
        he_size[i] = t_he
        # A chain with no helix or strand at all yields a 0x0 reference, hence 0 bytes;
        # torch.frombuffer rejects an empty buffer.
        ref_flat = (
            torch.frombuffer(bytearray(ref_bytes), dtype=torch.uint8)
            if ref_bytes
            else torch.zeros(0, dtype=torch.uint8)
        )
        he_flat.append(ref_flat)
        he_offset.append(he_offset[-1] + ref_flat.numel())

        # Sequences come from the selection CSV, not the graph: if the attribute were absent the
        # hash would fall back to the chain id, every mate would look sequence-distinct, and the
        # "exclude identical-sequence templates" policy would silently do nothing.
        seq = id_to_seq.get(ids[i])
        if seq is None:
            n_no_seq += 1
        key = str(seq) if seq is not None else ids[i]
        seq_hash[i] = int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % (2**62)

    print(f"chains without usable DSSP or contact map: {n_missing}", flush=True)
    print(f"chains with no sequence in the CSV: {n_no_seq}", flush=True)
    runs_tensor = (
        torch.tensor(runs_flat, dtype=torch.int16)
        if runs_flat
        else torch.zeros((0, 2), dtype=torch.int16)
    )

    out = {
        "ids": ids[:n],
        "cluster_of": torch.tensor(cluster_of[:n], dtype=torch.int32),
        "members_flat": torch.tensor(members_flat, dtype=torch.int32),
        "members_offset": torch.tensor(members_offset, dtype=torch.int64),
        "seq_hash": seq_hash,
        "runs_flat": runs_tensor,
        "runs_offset": torch.tensor(runs_offset, dtype=torch.int64),
        "he_offset": torch.tensor(he_offset, dtype=torch.int64),
        "he_size": he_size,
        "he_flat": torch.cat(he_flat) if he_flat else torch.zeros(0, dtype=torch.uint8),
        "min_len": args.min_len,
        "contact_threshold": args.contact_threshold,
    }
    torch.save(out, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
