#!/usr/bin/env python3
"""Joint mmseqs clustering of the combined PDB+AFDB set (Phase 2b step 3).

Reads the combined CSV (id, sequence), writes a FASTA, runs mmseqs easy-cluster at
--seqid (cov 0.8, cov-mode 1) via cluster_sequences, producing
cluster_seqid_<seqid>_<out_dir.name>.tsv (the name the datamodule reads).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from proteinfoundation.utils.cluster_utils import cluster_sequences, df_to_fasta, read_cluster_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqid", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    df = pd.read_csv(args.csv)
    print(f"[combclust] loaded {len(df)} rows from {args.csv}", flush=True)

    fasta = out_dir / "combined_for_clustering.fasta"
    df_to_fasta(df, str(fasta))
    print(f"[combclust] wrote FASTA {fasta} ({len(df)} seqs); starting mmseqs...", flush=True)

    cluster_out = out_dir / f"cluster_seqid_{args.seqid}_{out_dir.name}.fasta"
    cluster_sequences(str(fasta), str(cluster_out), min_seq_id=args.seqid,
                      coverage=0.8, overwrite=True, silence_mmseqs_output=False)

    tsv = cluster_out.with_suffix(".tsv")
    m = read_cluster_tsv(str(tsv))
    n_chains = sum(len(v) for v in m.values())
    print(f"[combclust] {tsv.name}: clusters={len(m)} chains={n_chains}", flush=True)
    print("[combclust] DONE", flush=True)


if __name__ == "__main__":
    main()
