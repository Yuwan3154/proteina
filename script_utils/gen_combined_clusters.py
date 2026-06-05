#!/usr/bin/env python3
"""Joint mmseqs clustering of the combined PDB+AFDB set (Phase 2b step 3).

Reads the combined CSV (id, sequence) -> FASTA -> mmseqs easy-cluster (direct subprocess
with --threads matched to the allocation + tmp on node-local --tmp-dir, to avoid the
384-thread oversubscription + networked-tmp timeout). Writes
cluster_seqid_<seqid>_<out_dir.name>.tsv (the name the datamodule reads) and the rep FASTA
as its .fasta sibling.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from proteinfoundation.utils.cluster_utils import df_to_fasta, read_cluster_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqid", type=float, default=0.25)
    ap.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    df = pd.read_csv(args.csv)
    print(f"[combclust] loaded {len(df)} rows from {args.csv}", flush=True)

    fasta = out_dir / "combined_for_clustering.fasta"
    df_to_fasta(df, str(fasta))
    print(f"[combclust] wrote FASTA ({len(df)} seqs); mmseqs threads={args.threads} tmp={args.tmp_dir}", flush=True)

    work = pathlib.Path(args.tmp_dir) / f"combclust_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = ["mmseqs", "easy-cluster", str(fasta.resolve()), "pdb_cluster", "tmp",
           "--min-seq-id", str(args.seqid), "-c", "0.8", "--cov-mode", "1",
           "--threads", str(args.threads), "--remove-tmp-files", "1"]
    r = subprocess.run(cmd, cwd=str(work))
    assert r.returncode == 0, f"mmseqs failed (exit {r.returncode})"

    tsv = out_dir / f"cluster_seqid_{args.seqid}_{out_dir.name}.tsv"
    rep = out_dir / f"cluster_seqid_{args.seqid}_{out_dir.name}.fasta"
    shutil.move(str(work / "pdb_cluster_cluster.tsv"), str(tsv))
    shutil.move(str(work / "pdb_cluster_rep_seq.fasta"), str(rep))
    shutil.rmtree(str(work), ignore_errors=True)

    m = read_cluster_tsv(str(tsv))
    print(f"[combclust] {tsv.name}: clusters={len(m)} chains={sum(len(v) for v in m.values())}", flush=True)
    print("[combclust] DONE", flush=True)


if __name__ == "__main__":
    main()
