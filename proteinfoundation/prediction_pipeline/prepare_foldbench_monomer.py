#!/usr/bin/env python
"""Prepare the FoldBench protein-monomer set for the proteina prediction pipeline.

Reads FoldBench's ``targets/monomer_protein.csv`` (columns ``pdb_id,chain_id`` where
``pdb_id`` looks like ``5sbj-assembly1``), acquires each ground-truth mmCIF, extracts
the requested chain's sequence, filters to chains usable by the maxlen-256 model, and
emits everything the eval-mode pipeline expects, in the SAME shape as af2rank_single:

  * per-target PT files at ``$DATA_PATH/pdb_train/processed/{pdb_id}_{chain}.pt``
    (the model loads sequences from here — see inference.py),
  * native CIFs copied flat to ``<out_root>/pdb/{pdb_id}.cif`` (the run's --cif_dir),
  * a manifest ``<out_root>/foldbench_monomer_manifest.csv`` with the af2rank_single
    columns (required: natives_rcsb, tms_single, in_train, length).

CIF acquisition order: a local FoldBench ground-truth package (``--gt_cif_dir``) first,
else a direct RCSB assembly download (``--allow_rcsb_fallback``; note RCSB's current
release may differ slightly from FoldBench's frozen copy — flag those).

All FoldBench monomers are deposited 2023-01-13..2024-11-01, i.e. AFTER the model's
2019-08-28 PDB-finetune cutoff, so ``in_train=False`` for all. (Structural proximity
to the train set is measured separately by train_closeness_search.py.)

Run in the ``proteina`` conda env (needs torch + torch_geometric + biopython).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

_THIS = Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

from proteinfoundation.prediction_pipeline.cif_to_pt_converter import (  # noqa: E402
    extract_sequence_from_cif,
    sequence_to_pt_data,
)

log = logging.getLogger("prepare_foldbench")

# af2rank_single manifest schema (only natives_rcsb/tms_single/in_train/length are
# consumed by the pipeline; the rest are placeholders for shape parity).
_MANIFEST_COLS = ["natives_frank", "tms_single", "tms_msa", "natives_rcsb", "denovo", "TMscore", "in_train", "length"]


def _rcsb_url(pdb_id: str) -> str:
    if "-assembly" in pdb_id:
        entry, asm = pdb_id.split("-assembly", 1)
        return f"https://files.rcsb.org/download/{entry.upper()}-assembly{asm}.cif"
    return f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"


def acquire_cif(pdb_id: str, gt_cif_dir: Optional[Path], out_pdb_dir: Path,
                allow_rcsb: bool) -> tuple[Optional[Path], str]:
    """Return (cif_path, source) where source in {local, rcsb, cached, ''}."""
    dest = out_pdb_dir / f"{pdb_id}.cif"
    if dest.exists() and dest.stat().st_size > 0:
        return dest, "cached"
    if gt_cif_dir is not None:
        cand = gt_cif_dir / f"{pdb_id}.cif"
        if cand.exists() and cand.stat().st_size > 0:
            shutil.copy(cand, dest)
            return dest, "local"
    if allow_rcsb:
        url = _rcsb_url(pdb_id)
        req = urllib.request.Request(url, headers={"User-Agent": "prepare_foldbench/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if dest.stat().st_size > 0:
                return dest, "rcsb"
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            log.warning("  RCSB download failed for %s (%s): %s", pdb_id, url, e)
            if dest.exists():
                dest.unlink()
    return None, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets_csv", required=True, help="FoldBench targets/monomer_protein.csv.")
    ap.add_argument("--out_root", required=True, help="Output root (native CIFs + manifest). NOT a charged mount.")
    ap.add_argument("--gt_cif_dir", default=None, help="Local FoldBench ground-truth CIF dir (the gdown package).")
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH") or str(Path.home() / "proteina/data"),
                    help="DATA_PATH; PT files go to <data_path>/pdb_train/processed/.")
    ap.add_argument("--max_len", type=int, default=256, help="Drop chains longer than this (model maxlen).")
    ap.add_argument("--min_len", type=int, default=10, help="Drop chains shorter than this.")
    ap.add_argument("--allow_rcsb_fallback", action="store_true",
                    help="If a CIF is not in --gt_cif_dir, download the assembly from RCSB.")
    ap.add_argument("--max_x_frac", type=float, default=0.5,
                    help="Drop chains whose sequence is >this fraction non-standard (X).")
    ap.add_argument("--manifest_name", default="foldbench_monomer_manifest.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s", stream=sys.stderr)

    out_root = Path(args.out_root).expanduser()
    out_pdb_dir = out_root / "pdb"
    out_pdb_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(args.data_path).expanduser() / "pdb_train" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    gt_cif_dir = Path(args.gt_cif_dir).expanduser() if args.gt_cif_dir else None

    targets = pd.read_csv(args.targets_csv, dtype=str, keep_default_na=False)
    log.info("FoldBench monomer targets: %d rows", len(targets))

    rows, dropped = [], []
    src_counts = {"local": 0, "rcsb": 0, "cached": 0, "": 0}
    for _, t in targets.iterrows():
        pdb_id, chain_id = str(t["pdb_id"]).strip(), str(t["chain_id"]).strip()
        protein_id = f"{pdb_id}_{chain_id}"

        cif, source = acquire_cif(pdb_id, gt_cif_dir, out_pdb_dir, args.allow_rcsb_fallback)
        src_counts[source] = src_counts.get(source, 0) + 1
        if cif is None:
            dropped.append((protein_id, "no_cif"))
            continue

        seq = extract_sequence_from_cif(str(cif), chain_id)
        if not seq:
            dropped.append((protein_id, "no_chain_or_empty_seq"))
            continue
        length = len(seq)
        x_frac = seq.count("X") / max(length, 1)
        if x_frac > args.max_x_frac:
            dropped.append((protein_id, f"nonstd_x_frac={x_frac:.2f}"))
            continue
        if length > args.max_len:
            dropped.append((protein_id, f"too_long={length}"))
            continue
        if length < args.min_len:
            dropped.append((protein_id, f"too_short={length}"))
            continue

        pt_path = processed_dir / f"{protein_id}.pt"
        if not pt_path.exists():
            torch.save(sequence_to_pt_data(seq, protein_id), pt_path)

        rows.append({
            "natives_frank": pdb_id,
            "tms_single": "",        # no AF2 single-seq baseline for FoldBench
            "tms_msa": "",
            "natives_rcsb": protein_id,
            "denovo": "",
            "TMscore": "",
            "in_train": False,       # all FoldBench monomers are post-2019-08-28 cutoff
            "length": length,
        })

    manifest = pd.DataFrame(rows, columns=_MANIFEST_COLS)
    manifest_path = out_root / args.manifest_name
    manifest.to_csv(manifest_path, index=False)

    drop_df = pd.DataFrame(dropped, columns=["protein_id", "reason"])
    drop_df.to_csv(out_root / "foldbench_monomer_dropped.csv", index=False)

    # Reconciliation report
    reasons = drop_df["reason"].str.split("=").str[0].value_counts().to_dict() if len(drop_df) else {}
    log.info("=" * 60)
    log.info("FoldBench monomer prep summary")
    log.info("  input targets         : %d", len(targets))
    log.info("  CIF source            : local=%d rcsb=%d cached=%d missing=%d",
             src_counts.get("local", 0), src_counts.get("rcsb", 0),
             src_counts.get("cached", 0), src_counts.get("", 0))
    log.info("  PREPARED (manifest)   : %d", len(manifest))
    log.info("  dropped               : %d  %s", len(drop_df), reasons)
    if len(manifest):
        ln = manifest["length"].astype(int)
        log.info("  length: min=%d median=%d max=%d", ln.min(), int(ln.median()), ln.max())
    log.info("  manifest -> %s", manifest_path)
    log.info("  natives -> %s", out_pdb_dir)
    log.info("  PT files -> %s", processed_dir)
    log.info("NOTE: FoldBench's final protein-monomer set is 334 (7 de novo designs removed +")
    log.info("      resolution/40%%-identity clustering). This prepared set is 'monomers with a")
    log.info("      usable single chain in [%d,%d] residues' and is NOT de-novo-filtered, so the",
             args.min_len, args.max_len)
    log.info("      count differs from 334; intersect with FoldBench's canonical list if needed.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
