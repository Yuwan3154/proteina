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
import hashlib
import os
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-subdir", default="pdb")
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default=os.environ.get("PROTEINA_REPO", "."))
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0, help="debug: only index N chains")
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

    for i in range(n):
        stem = ids[i]
        path = ds._processed_path_for(f"{stem}.pt")
        if not os.path.exists(path):
            n_missing += 1
            runs_offset.append(len(runs_flat))
            he_offset.append(len(he_flat))
            continue
        g = torch.load(path, map_location="cpu", weights_only=False)
        dssp = getattr(g, "dssp_target", None)
        raw = getattr(g, "contact_map_confind", None)
        if dssp is None or bool((dssp < 0).all()) or raw is None:
            n_missing += 1
            runs_offset.append(len(runs_flat))
            he_offset.append(len(he_flat))
            continue

        runs = dssp_to_runs(dssp, min_len=args.min_len)
        runs_flat.extend(runs)  # list of (type, length); stacked into [total, 2] below
        runs_offset.append(runs_offset[-1] + len(runs))

        cm = (raw.float() >= args.contact_threshold).float()
        ref, keep = sse_contact_reference(cm, runs, keep_types=(DSSP_HELIX, DSSP_STRAND))
        he_size[i] = len(keep)
        he_flat.append(ref.to(torch.uint8).flatten())
        he_offset.append(he_offset[-1] + ref.numel())

        # Sequences come from the selection CSV, not the graph: if the attribute were absent the
        # hash would fall back to the chain id, every mate would look sequence-distinct, and the
        # "exclude identical-sequence templates" policy would silently do nothing.
        seq = id_to_seq.get(stem)
        if seq is None:
            n_no_seq += 1
        key = str(seq) if seq is not None else stem
        seq_hash[i] = int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % (2**62)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1}/{n}", flush=True)

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
