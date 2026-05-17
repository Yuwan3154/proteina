"""Compute pTM for AFDB structures from the published PAE matrices.

AFDB exposes predicted-aligned-error matrices at
``https://alphafold.ebi.ac.uk/files/AF-{UniProt}-F1-predicted_aligned_error_v6.json``
but does **not** publish pTM directly. We compute pTM from the PAE matrix using
the same formula as ``proteinfoundation.openfold_stub.utils.loss.compute_tm``,
specialised to a one-hot bin distribution at the published expected PAE:

    N_eff     = max(N, 19)
    d0        = 1.24 * (N_eff - 15) ** (1/3) - 1.8
    per_pair  = 1 / (1 + (PAE ** 2) / (d0 ** 2))    # shape (N, N)
    per_align = per_pair.mean(axis=1)                # shape (N,)
    pTM       = per_align.max()

This is exact (to 1e-7) versus ``compute_tm`` when the bin distribution is a
delta at the expected value — which is the only thing AFDB publishes. See the
plan file for the equivalence proof.

The CLI mirrors ``msa_analysis.py`` so this script reuses its rate limiter,
session pooling, and retry/Retry-After logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reuse the network and concurrency utilities from msa_analysis.
_THIS = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
import msa_analysis as _ma  # noqa: E402

log = logging.getLogger("compute_afdb_ptm")


# ---------------------------------------------------------------------------
# pTM
# ---------------------------------------------------------------------------


def compute_ptm_from_pae(pae: np.ndarray) -> float:
    """Closed-form pTM from an NxN expected-PAE matrix (Å)."""
    pae = np.asarray(pae, dtype=np.float64)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square 2D, got shape {pae.shape}")
    n = pae.shape[0]
    n_eff = max(n, 19)
    d0 = 1.24 * (n_eff - 15) ** (1.0 / 3.0) - 1.8
    per_pair = 1.0 / (1.0 + (pae ** 2) / (d0 ** 2))
    per_alignment = per_pair.mean(axis=1)
    return float(per_alignment.max())


def _self_check() -> None:
    """Sanity check: verify the closed-form matches ``compute_tm`` for one-hot logits."""
    try:
        import torch
        from proteinfoundation.openfold_stub.utils.loss import (
            compute_tm,
            _calculate_bin_centers,
        )
    except Exception as e:
        log.debug("Skipping self-check (compute_tm not importable): %s", e)
        return
    N, max_bin, no_bins = 100, 31, 64
    boundaries = torch.linspace(0, max_bin, steps=no_bins - 1)
    bc = _calculate_bin_centers(boundaries)
    for pae_val in (1.0, 2.5, 5.0, 10.0):
        bin_idx = int(torch.argmin(torch.abs(bc - pae_val)))
        logits = torch.full((N, N, no_bins), -1e9)
        logits[..., bin_idx] = 0.0
        ptm_lib = float(compute_tm(logits, max_bin=max_bin, no_bins=no_bins).item())
        pae = np.full((N, N), float(bc[bin_idx]))
        ptm_direct = compute_ptm_from_pae(pae)
        assert abs(ptm_lib - ptm_direct) < 1e-5, (
            f"closed-form pTM disagrees with compute_tm: pae={pae_val}, lib={ptm_lib}, "
            f"direct={ptm_direct}"
        )
    log.info("Self-check passed: closed-form pTM matches compute_tm within 1e-5.")


# ---------------------------------------------------------------------------
# PAE acquisition + parsing
# ---------------------------------------------------------------------------


def candidate_local_paths(pae_dir: pathlib.Path, uniprot: str) -> list[pathlib.Path]:
    return [
        pae_dir / f"AF-{uniprot}-F1-predicted_aligned_error_v6.json",
        pae_dir / f"{uniprot}.json",
    ]


def fetch_pae(
    uniprot: str,
    url_template: str,
    cache_path: pathlib.Path,
) -> Optional[pathlib.Path]:
    """Reuse msa_analysis.fetch_msa for the JSON download — same retry semantics."""
    return _ma.fetch_msa(uniprot, url_template, cache_path)


def locate_or_fetch_pae(
    uniprot: str,
    pae_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    url_template: str,
    no_fetch: bool,
) -> tuple[Optional[pathlib.Path], str]:
    for p in candidate_local_paths(pae_dir, uniprot):
        if p.is_file() and p.stat().st_size > 0:
            return p, "local"
    if pae_dir != cache_dir:
        for p in candidate_local_paths(cache_dir, uniprot):
            if p.is_file() and p.stat().st_size > 0:
                return p, "local"
    if no_fetch:
        return None, "missing"
    target = cache_dir / f"AF-{uniprot}-F1-predicted_aligned_error_v6.json"
    fetched = fetch_pae(uniprot, url_template, target)
    return (fetched, "downloaded" if fetched is not None else "missing")


def parse_pae_json(path: pathlib.Path) -> Optional[np.ndarray]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("[%s] parse error: %s", path.name, e)
        return None
    # AFDB v6 wraps the dict in a 1-element list.
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict):
        log.warning("[%s] unexpected JSON shape: %s", path.name, type(data).__name__)
        return None
    pae_raw = data.get("predicted_aligned_error")
    if pae_raw is None:
        log.warning("[%s] no predicted_aligned_error key", path.name)
        return None
    try:
        return np.asarray(pae_raw, dtype=np.float32)
    except (ValueError, TypeError) as e:
        log.warning("[%s] could not convert PAE to array: %s", path.name, e)
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def acquire_all_pae(
    uniprots: list[str],
    pae_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    url_template: str,
    workers: int,
    no_fetch: bool,
    show_progress: bool,
) -> dict[str, tuple[Optional[pathlib.Path], str]]:
    unique = sorted(set(uniprots))
    result: dict[str, tuple[Optional[pathlib.Path], str]] = {}
    pbar = tqdm(total=len(unique), desc="PAE", disable=not show_progress)
    if workers <= 1:
        for u in unique:
            result[u] = locate_or_fetch_pae(u, pae_dir, cache_dir, url_template, no_fetch)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(locate_or_fetch_pae, u, pae_dir, cache_dir, url_template, no_fetch): u
                for u in unique
            }
            for fut in as_completed(futs):
                u = futs[fut]
                result[u] = fut.result()
                pbar.update(1)
    pbar.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute AFDB pTM from PAE JSONs.")
    ap.add_argument("--csv", type=pathlib.Path, required=True, help="Input CSV.")
    ap.add_argument("--id-col", required=True, help="UniProt-style ID column.")
    ap.add_argument(
        "--output-col",
        default="afdb_ptm",
        help="Name of the new pTM column to add (default: afdb_ptm).",
    )
    ap.add_argument(
        "--pae-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/data/afdb/pae/"),
        help="Where to look for / cache PAE JSON files.",
    )
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="Override download destination (default: --pae-dir).",
    )
    ap.add_argument(
        "--url-template",
        default="https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-predicted_aligned_error_v6.json",
        help="URL pattern (use `{uniprot}` placeholder).",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--min-request-interval",
        type=float,
        default=0.7,
        help="Global rate limit (seconds between requests). AFDB returns 429 on bursts.",
    )
    ap.add_argument("--no-fetch", action="store_true", help="Use cache only; no network.")
    ap.add_argument(
        "--length-col",
        default="length",
        help="Optional column with expected sequence length (warns on mismatch).",
    )
    ap.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output CSV path (default: overwrite --csv).",
    )
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--quiet", action="store_true")
    grp.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="[%(name)s] %(message)s", stream=sys.stderr)

    # Install global rate limiter into msa_analysis (the module that owns fetch_msa).
    _ma._RATE_LIMITER = _ma._RateLimiter(args.min_request_interval)

    _self_check()

    cache_dir = args.cache_dir if args.cache_dir is not None else args.pae_dir
    out_csv = args.output if args.output is not None else args.csv

    log.info("Reading %s", args.csv)
    df = pd.read_csv(args.csv, dtype={args.id_col: str}, keep_default_na=False)
    if args.id_col not in df.columns:
        raise SystemExit(f"{args.csv}: column {args.id_col!r} not found.")

    df["__uniprot__"] = df[args.id_col].astype(str).map(_ma.normalize_uniprot)
    uniprots = df["__uniprot__"].tolist()
    log.info("%d rows, %d unique uniprots", len(df), len(set(uniprots)))

    pae_paths = acquire_all_pae(
        uniprots=uniprots,
        pae_dir=args.pae_dir,
        cache_dir=cache_dir,
        url_template=args.url_template,
        workers=args.workers,
        no_fetch=args.no_fetch,
        show_progress=not args.quiet,
    )
    src_counts = {"local": 0, "downloaded": 0, "missing": 0}
    for _u, (_p, s) in pae_paths.items():
        src_counts[s] += 1
    log.info("PAE sources: %s", src_counts)

    expected_len = df[args.length_col] if args.length_col in df.columns else None
    cache: dict[str, float] = {}
    length_warnings = 0
    out = []
    for idx, u in enumerate(tqdm(uniprots, desc="pTM", disable=args.quiet)):
        if u in cache:
            out.append(cache[u])
            continue
        path, _src = pae_paths.get(u, (None, "missing"))
        ptm: float
        if path is None:
            ptm = float("nan")
        else:
            pae = parse_pae_json(path)
            if pae is None or pae.size == 0:
                ptm = float("nan")
            else:
                if expected_len is not None:
                    try:
                        el = int(expected_len.iloc[idx])
                        if el and el != pae.shape[0]:
                            length_warnings += 1
                            log.debug("[%s] length mismatch: csv=%d, pae=%d", u, el, pae.shape[0])
                    except (TypeError, ValueError):
                        pass
                try:
                    ptm = compute_ptm_from_pae(pae)
                except Exception as e:
                    log.warning("[%s] pTM compute failed: %s", u, e)
                    ptm = float("nan")
        cache[u] = ptm
        out.append(ptm)

    df[args.output_col] = out
    df = df.drop(columns="__uniprot__")

    if length_warnings:
        log.warning("%d rows had length mismatch between CSV and PAE matrix", length_warnings)

    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    log.info("Wrote %s (column %r)", out_csv, args.output_col)

    sidecar = out_csv.with_name(out_csv.stem + "_ptm_run.json")
    missing = sorted(u for u, (p, s) in pae_paths.items() if s == "missing")
    sidecar.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "csv": str(args.csv.resolve()),
                "id_col": args.id_col,
                "output_col": args.output_col,
                "pae_dir": str(args.pae_dir),
                "cache_dir": str(cache_dir),
                "url_template": args.url_template,
                "workers": args.workers,
                "min_request_interval": args.min_request_interval,
                "n_rows": int(len(df)),
                "n_unique_uniprots": int(len(set(uniprots))),
                "source_counts": src_counts,
                "length_mismatches": int(length_warnings),
                "missing_uniprots": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Wrote sidecar %s", sidecar)


if __name__ == "__main__":
    main()
