#!/usr/bin/env python
"""Quantify FoldBench-monomer sequence leakage into the AFDB pretraining set.

FoldBench removes targets with <40% sequence identity to a reproduced AlphaFold3
*PDB* training set, but it does NOT filter against AFDB. This model's base
checkpoint (proteina_v1.6_DFS_200M) is pretrained on the AFDB-derived D_FS set, so a
FoldBench monomer whose sequence has a close AFDB homolog may have been seen (as a
predicted structure) during pretraining. This script measures that: for each
FoldBench monomer, the maximum sequence identity to D_FS (or D_21M) via MMseqs2.

Inputs (NGC ``proteina_training_data_indices``):
  * ``seq_d_21M.fasta``  all D_21M sequences (~6-7 GB) — large.
  * ``d_fs_index.txt`` / ``d_21M_index.txt``  AFDB accessions per set.
You provide either a ready ``--ref_fasta`` (e.g. a pre-subset D_FS fasta) OR
``--seq_d21m_fasta`` + ``--ref_index`` to subset on the fly.

Query sequences come from the prepared FoldBench manifest (``--manifest`` +
``--cif_dir``) or a ready ``--query_fasta``.

Output ``afdb_leakage.csv`` (per monomer: max_pident, best_ref_accession, qcov) plus
a leakage-rate summary at 40/50/90% identity.

DISK: ``seq_d_21M.fasta`` is large; subsetting D_FS (~588k seqs) is ~150-250 MB. On a
24 GB-free host prefer passing a pre-subset ``--ref_fasta`` (build it where there is
room). D_21M-scale runs belong on a roomy host (Engaging).

Run where MMseqs2 is available (repo env has ``bioconda::mmseqs2``); needs biopython
only if extracting query seqs from CIFs.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_THIS = Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

log = logging.getLogger("afdb_leakage")

_M8_COLS = ["query", "target", "pident", "qcov", "tcov", "evalue", "bits"]


def _norm_acc(s: str) -> str:
    """Normalize an AFDB accession for matching across index/fasta formats.

    Handles 'AF-A0A023PXA5-F1-model_v6', 'AF-A0A023PXA5-F1', 'A0A023PXA5-F1', etc.
    Reduces to the UniProt+fragment core 'A0A023PXA5-F1'.
    """
    s = s.strip().lstrip(">").split()[0]
    if s.startswith("AF-"):
        s = s[3:]
    for suf in ("-model_v6", "-model_v4", "-model_v3", "-model_v2", "-model_v1"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def subset_ref_fasta(seq_fasta: Path, index_file: Path, out_fasta: Path) -> int:
    """Stream seq_fasta, keep records whose accession is in index_file."""
    keep = {_norm_acc(ln) for ln in index_file.read_text().splitlines() if ln.strip()}
    log.info("ref index: %d accessions", len(keep))
    n_written = 0
    with open(seq_fasta) as fin, open(out_fasta, "w") as fout:
        write = False
        for line in fin:
            if line.startswith(">"):
                write = _norm_acc(line) in keep
                if write:
                    fout.write(line)
                    n_written += 1
            elif write:
                fout.write(line)
    log.info("subset ref fasta: %d sequences -> %s", n_written, out_fasta)
    return n_written


def build_query_fasta(manifest: Path, cif_dir: Path, id_col: str, out_fasta: Path) -> int:
    from proteinfoundation.prediction_pipeline.cif_to_pt_converter import extract_sequence_from_cif
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    n = 0
    with open(out_fasta, "w") as f:
        for pid in df[id_col].astype(str):
            pdb_id = pid.split("_")[0]
            chain = pid.split("_", 1)[1] if "_" in pid else "A"
            cif = cif_dir / f"{pdb_id}.cif"
            if not cif.exists():
                for child in sorted(p for p in cif_dir.iterdir() if p.is_dir()):
                    if (child / f"{pdb_id}.cif").exists():
                        cif = child / f"{pdb_id}.cif"
                        break
            if not cif.exists():
                continue
            seq = extract_sequence_from_cif(str(cif), chain)
            if seq:
                f.write(f">{pid}\n{''.join(seq)}\n")
                n += 1
    log.info("query fasta: %d sequences -> %s", n, out_fasta)
    return n


def run_mmseqs(mmseqs: str, query_fasta: Path, ref_fasta: Path, out_m8: Path,
               tmp_dir: Path, sensitivity: float, threads: int) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    cmd = [
        mmseqs, "easy-search", str(query_fasta), str(ref_fasta), str(out_m8), str(tmp_dir),
        "-s", str(sensitivity), "--max-seqs", "300", "--threads", str(threads),
        "--format-output", ",".join(_M8_COLS),
    ]
    log.info("mmseqs easy-search %s vs %s", query_fasta, ref_fasta)
    subprocess.run(cmd, check=True, env=env)


def reduce_leakage(out_m8: Path, query_ids: list[str]) -> pd.DataFrame:
    if out_m8.exists() and out_m8.stat().st_size > 0:
        hits = pd.read_csv(out_m8, sep="\t", names=_M8_COLS)
        hits["pident"] = pd.to_numeric(hits["pident"], errors="coerce")
        # mmseqs reports pident as a fraction (0-1) or percent depending on version;
        # normalize to percent.
        if hits["pident"].max() is not None and hits["pident"].max() <= 1.0:
            hits["pident"] = hits["pident"] * 100.0
        idx = hits.groupby("query")["pident"].idxmax()
        best = hits.loc[idx].set_index("query")
    else:
        best = pd.DataFrame()
    rows = []
    for qid in query_ids:
        if len(best) and qid in best.index:
            r = best.loc[qid]
            rows.append({"protein_id": qid, "max_pident": float(r["pident"]),
                         "best_ref_accession": str(r["target"]),
                         "qcov": float(pd.to_numeric(r.get("qcov"), errors="coerce"))})
        else:
            rows.append({"protein_id": qid, "max_pident": 0.0, "best_ref_accession": "", "qcov": 0.0})
    return pd.DataFrame(rows).sort_values("max_pident", ascending=False).reset_index(drop=True)


def _resolve_mmseqs(arg: Optional[str]) -> str:
    if arg:
        return arg
    for cand in [shutil.which("mmseqs"),
                 str(Path.home() / "miniforge3/envs/proteina/bin/mmseqs"),
                 str(Path.home() / "miniforge3/envs/foldseek/bin/mmseqs")]:
        if cand and os.path.exists(cand):
            return cand
    raise SystemExit("mmseqs not found; pass --mmseqs_bin (repo env has bioconda::mmseqs2).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", required=True)
    # query
    ap.add_argument("--query_fasta", default=None)
    ap.add_argument("--manifest", default=None, help="FoldBench manifest (with --cif_dir) to build query fasta.")
    ap.add_argument("--cif_dir", default=None)
    ap.add_argument("--id_col", default="natives_rcsb")
    # reference
    ap.add_argument("--ref_fasta", default=None, help="Ready reference fasta (e.g. pre-subset D_FS).")
    ap.add_argument("--seq_d21m_fasta", default=None, help="NGC seq_d_21M.fasta (to subset).")
    ap.add_argument("--ref_index", default=None, help="NGC d_fs_index.txt / d_21M_index.txt to subset by.")
    ap.add_argument("--mmseqs_bin", default=None)
    ap.add_argument("--sensitivity", type=float, default=7.5)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s", stream=sys.stderr)
    work = Path(args.work_dir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    mmseqs = _resolve_mmseqs(args.mmseqs_bin)

    # Query fasta
    if args.query_fasta:
        query_fasta = Path(args.query_fasta).expanduser()
        q_ids = [ln[1:].strip().split()[0] for ln in query_fasta.read_text().splitlines() if ln.startswith(">")]
    else:
        if not (args.manifest and args.cif_dir):
            raise SystemExit("Provide --query_fasta OR (--manifest and --cif_dir).")
        query_fasta = work / "foldbench_query.fasta"
        build_query_fasta(Path(args.manifest), Path(args.cif_dir).expanduser(), args.id_col, query_fasta)
        q_ids = [ln[1:].strip().split()[0] for ln in query_fasta.read_text().splitlines() if ln.startswith(">")]

    # Reference fasta
    if args.ref_fasta:
        ref_fasta = Path(args.ref_fasta).expanduser()
    else:
        if not (args.seq_d21m_fasta and args.ref_index):
            raise SystemExit("Provide --ref_fasta OR (--seq_d21m_fasta and --ref_index).")
        ref_fasta = work / "ref_subset.fasta"
        if not (ref_fasta.exists() and ref_fasta.stat().st_size > 0):
            subset_ref_fasta(Path(args.seq_d21m_fasta).expanduser(), Path(args.ref_index).expanduser(), ref_fasta)

    out_m8 = work / "leakage_hits.m8"
    run_mmseqs(mmseqs, query_fasta, ref_fasta, out_m8, work / "mmseqs_tmp", args.sensitivity, args.threads)

    df = reduce_leakage(out_m8, q_ids)
    out_csv = work / "afdb_leakage.csv"
    df.to_csv(out_csv, index=False)

    n = len(df)
    log.info("=" * 60)
    log.info("AFDB sequence-leakage summary (n=%d FoldBench monomers)", n)
    for thr in (40.0, 50.0, 70.0, 90.0):
        c = int((df["max_pident"] >= thr).sum())
        log.info("  max identity to ref >= %2.0f%%: %d (%.1f%%)", thr, c, 100.0 * c / max(n, 1))
    log.info("  median max-identity: %.1f%%", float(df["max_pident"].median()) if n else 0.0)
    log.info("  -> %s", out_csv)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
