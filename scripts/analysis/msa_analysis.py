"""Compute MSA summary metrics for any CSV with UniProt-style IDs.

Looks up an ``.a3m`` MSA per row (local cache first, then fetch from AFDB),
parses it, and adds five columns to the CSV:

    msa_depth                       total number of records (query + hits)
    msa_mean_coverage_top128        mean per-hit non-gap fraction over the top K hits
    msa_std_coverage_top128         stddev of the same
    msa_mean_identity_top128        mean per-hit identity to query (over its non-gap positions)
    msa_std_identity_top128         stddev of the same

Then writes a 5-panel figure of each metric (x-axis) vs. the AFDB confidence column
(y-axis; default ``mean_plddt``).

Hits are taken to be already sorted (MSA generators typically emit best-first).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

log = logging.getLogger("msa_analysis")


# ---------------------------------------------------------------------------
# A3M parsing
# ---------------------------------------------------------------------------

_KEEP_RE = re.compile(r"[A-Z\-]")


def parse_a3m(path: pathlib.Path) -> list[str]:
    """Return aligned sequences. Record 0 is the query.

    A3M convention: lowercase letters are insertions relative to the query —
    we drop them. Uppercase letters and ``-`` survive. Trailing newlines /
    other whitespace are stripped naturally because we keep only the kept set.
    """
    records: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records.append("".join(current))
                    current = []
            else:
                current.append(line.rstrip("\n"))
        if current:
            records.append("".join(current))
    return ["".join(_KEEP_RE.findall(s)) for s in records]


# ---------------------------------------------------------------------------
# MSA acquisition (local + remote)
# ---------------------------------------------------------------------------

_ACCESSION_SUFFIX_RE = re.compile(r"-F\d+$")


def normalize_uniprot(raw: str) -> str:
    """Strip an AlphaFold ``-F\\d+`` fragment suffix; trim whitespace."""
    return _ACCESSION_SUFFIX_RE.sub("", str(raw).strip())


def candidate_local_paths(msa_dir: pathlib.Path, uniprot: str) -> list[pathlib.Path]:
    return [
        msa_dir / f"{uniprot}.a3m",
        msa_dir / f"AF-{uniprot}-F1-msa_v6.a3m",
    ]


_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "proteina-msa-analysis/1.0"})
        _thread_local.session = s
    return s


class _RateLimiter:
    """Global minimum-interval gate, thread-safe.

    AFDB returns 429s very quickly under burst. Configure ``min_interval`` to
    keep total request rate just under the server's limit (≈ 1 req/s headroom).
    """

    def __init__(self, min_interval: float):
        self._min_interval = float(min_interval)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._next - now
            if wait_for > 0:
                target = self._next
            else:
                target = now
            self._next = target + self._min_interval
        if wait_for > 0:
            time.sleep(wait_for)


_RATE_LIMITER: Optional[_RateLimiter] = None


def fetch_msa(
    uniprot: str,
    url_template: str,
    cache_path: pathlib.Path,
    max_attempts: int = 6,
    timeout_s: float = 60.0,
) -> Optional[pathlib.Path]:
    """Download MSA for ``uniprot`` to ``cache_path`` atomically.

    Returns the path on success, ``None`` on 404 / non-retryable error.
    Retries 5xx, 429, and connection errors with exponential backoff (and
    honours ``Retry-After`` when present).
    """
    url = url_template.format(uniprot=uniprot)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    delay = 2.0
    for attempt in range(1, max_attempts + 1):
        if _RATE_LIMITER is not None:
            _RATE_LIMITER.wait()
        try:
            r = _session().get(url, timeout=timeout_s, stream=True)
        except requests.RequestException as e:
            log.debug("[%s] attempt %d connection error: %s", uniprot, attempt, e)
            if attempt == max_attempts:
                return None
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if r.status_code == 404:
            log.info("[%s] 404 at %s", uniprot, url)
            r.close()
            return None
        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            r.close()
            wait = delay
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            log.debug("[%s] attempt %d HTTP %d, sleeping %.1fs", uniprot, attempt, r.status_code, wait)
            if attempt == max_attempts:
                log.warning("[%s] giving up after HTTP %d", uniprot, r.status_code)
                return None
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
            continue
        if r.status_code != 200:
            log.warning("[%s] HTTP %d at %s", uniprot, r.status_code, url)
            r.close()
            return None
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        r.close()
        os.replace(tmp, cache_path)
        return cache_path
    return None


def locate_or_fetch(
    uniprot: str,
    msa_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    url_template: str,
    no_fetch: bool,
) -> tuple[Optional[pathlib.Path], str]:
    """Return (path, source) where source ∈ {local, downloaded, missing}."""
    for p in candidate_local_paths(msa_dir, uniprot):
        if p.is_file() and p.stat().st_size > 0:
            return p, "local"
    if msa_dir != cache_dir:
        for p in candidate_local_paths(cache_dir, uniprot):
            if p.is_file() and p.stat().st_size > 0:
                return p, "local"
    if no_fetch:
        return None, "missing"
    target = cache_dir / f"AF-{uniprot}-F1-msa_v6.a3m"
    fetched = fetch_msa(uniprot, url_template, target)
    return (fetched, "downloaded" if fetched is not None else "missing")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(path: pathlib.Path, top_k: int) -> dict:
    """Parse the MSA at ``path`` and return the 5 metric dict (NaN on degenerate)."""
    out = {
        "msa_depth": np.nan,
        "msa_mean_coverage_top128": np.nan,
        "msa_std_coverage_top128": np.nan,
        "msa_mean_identity_top128": np.nan,
        "msa_std_identity_top128": np.nan,
    }
    seqs = parse_a3m(path)
    if not seqs:
        return out
    out["msa_depth"] = len(seqs)
    query = seqs[0]
    L = len(query)
    if L == 0:
        return out

    hits = []
    for s in seqs[1 : 1 + top_k]:
        if len(s) != L:
            continue
        hits.append(s)
    if not hits:
        return out

    K = len(hits)
    query_arr = np.frombuffer(query.encode("ascii"), dtype=np.uint8)
    msa = np.frombuffer("".join(hits).encode("ascii"), dtype=np.uint8).reshape(K, L)

    gap = ord("-")
    non_gap = msa != gap
    cov_per_hit = non_gap.sum(axis=1) / L

    match = (msa == query_arr[None, :]) & non_gap
    non_gap_count = non_gap.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ident_per_hit = np.where(non_gap_count > 0, match.sum(axis=1) / non_gap_count, np.nan)

    out["msa_mean_coverage_top128"] = float(np.nanmean(cov_per_hit))
    out["msa_std_coverage_top128"] = float(np.nanstd(cov_per_hit))
    out["msa_mean_identity_top128"] = float(np.nanmean(ident_per_hit))
    out["msa_std_identity_top128"] = float(np.nanstd(ident_per_hit))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    x = x[mask]; y = y[mask]
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan"), len(x)
    r = float(np.corrcoef(rx, ry)[0, 1])
    return r, len(x)


def _plot_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    log_x: bool,
    dense_threshold: int = 400,
) -> None:
    """Single x-vs-y panel. Hexbin for dense data, scatter for sparse, stats box."""
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    r, n = _spearman(x, y)

    if log_x:
        pos = xv > 0
        xv = xv[pos]
        yv = yv[pos]

    if len(xv) >= dense_threshold:
        hb = ax.hexbin(
            xv,
            yv,
            gridsize=40,
            bins="log",
            xscale="log" if log_x else "linear",
            cmap="viridis",
            mincnt=1,
            linewidths=0,
        )
        cb = ax.figure.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label("log10(N)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        if log_x:
            ax.set_xscale("log")
        ax.scatter(xv, yv, s=18, alpha=0.6, edgecolors="none", color="#1f77b4")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.text(
        0.02,
        0.97,
        f"n = {n}\nSpearman r = {r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )


def make_figure(df: pd.DataFrame, plddt_col: str, fig_path: pathlib.Path, title: str) -> None:
    """1 + 2 + 2 layout: depth wide on top, then coverage row, then identity row."""
    import matplotlib.gridspec as gridspec

    y = df[plddt_col].to_numpy(dtype=float)

    import textwrap as _tw
    title_lines = _tw.wrap(title, width=90) or [title]
    n_title_lines = len(title_lines) + 1  # +1 for the "(y axis: ...)" line
    # Reserve ~0.025 of figure height per title line.
    top_margin = max(0.90, 1.0 - 0.028 * n_title_lines)

    fig = plt.figure(figsize=(12, 13), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
                           top=top_margin, bottom=0.06, left=0.08, right=0.97)

    ax_depth = fig.add_subplot(gs[0, :])
    _plot_panel(ax_depth, df["msa_depth"].to_numpy(dtype=float), y,
                xlabel="MSA depth (number of sequences)", ylabel=plddt_col, log_x=True)
    ax_depth.set_title("MSA depth")

    panels = [
        (gs[1, 0], "msa_mean_coverage_top128", "mean coverage (top-128)", "Coverage — mean"),
        (gs[1, 1], "msa_std_coverage_top128",  "coverage stddev (top-128)", "Coverage — stddev"),
        (gs[2, 0], "msa_mean_identity_top128", "mean identity (top-128)",   "Identity — mean"),
        (gs[2, 1], "msa_std_identity_top128",  "identity stddev (top-128)", "Identity — stddev"),
    ]
    for spec, col, xlab, ttl in panels:
        ax = fig.add_subplot(spec)
        _plot_panel(ax, df[col].to_numpy(dtype=float), y, xlabel=xlab, ylabel=plddt_col, log_x=False)
        ax.set_title(ttl)

    wrapped = "\n".join(title_lines)
    fig.suptitle(f"{wrapped}\n(y axis: {plddt_col})", fontsize=11, y=0.995, va="top")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def acquire_all(
    uniprots: list[str],
    msa_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    url_template: str,
    workers: int,
    no_fetch: bool,
    show_progress: bool,
) -> dict[str, tuple[Optional[pathlib.Path], str]]:
    """Locate or download every MSA, returning {uniprot: (path|None, source)}."""
    result: dict[str, tuple[Optional[pathlib.Path], str]] = {}
    unique = sorted(set(uniprots))
    pbar = tqdm(total=len(unique), desc="MSAs", disable=not show_progress)
    if workers <= 1:
        for u in unique:
            result[u] = locate_or_fetch(u, msa_dir, cache_dir, url_template, no_fetch)
            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(locate_or_fetch, u, msa_dir, cache_dir, url_template, no_fetch): u
                for u in unique
            }
            for fut in as_completed(futs):
                u = futs[fut]
                result[u] = fut.result()
                pbar.update(1)
    pbar.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute MSA metrics + AFDB-vs-MSA figures.")
    ap.add_argument("--csv", type=pathlib.Path, required=True, help="Input CSV to annotate.")
    ap.add_argument("--id-col", required=True, help="Column with UniProt-style accession IDs.")
    ap.add_argument(
        "--plddt-col",
        default="mean_plddt",
        help="AFDB confidence column for the figure y-axis (default: mean_plddt).",
    )
    ap.add_argument(
        "--msa-dir",
        type=pathlib.Path,
        required=True,
        help="Directory of pre-existing .a3m files (searched first).",
    )
    ap.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=None,
        help="Where to write fetched MSAs (default: --msa-dir).",
    )
    ap.add_argument(
        "--url-template",
        default="https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot}-F1-msa_v6.a3m",
        help="URL pattern for MSA download (use `{uniprot}` placeholder).",
    )
    ap.add_argument("--top-k", type=int, default=128, help="Use this many hits after query.")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent download workers.")
    ap.add_argument(
        "--min-request-interval",
        type=float,
        default=0.7,
        help=(
            "Minimum seconds between successive HTTP requests across ALL workers "
            "(global rate limiter). AFDB rate-limits short bursts aggressively; "
            "0.7s gives ~85 req/min. Set to 0 to disable."
        ),
    )
    ap.add_argument("--no-fetch", action="store_true", help="Disable network; missing → NaN.")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        default=True,
        help="Default. Missing MSAs leave NaN metrics; pass --no-allow-missing to fail instead.",
    )
    ap.add_argument("--no-allow-missing", dest="allow_missing", action="store_false")
    ap.add_argument(
        "--fig-out",
        type=pathlib.Path,
        default=None,
        help="Output figure path (default: <csv stem>_msa_metrics.png next to CSV).",
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

    global _RATE_LIMITER
    _RATE_LIMITER = _RateLimiter(args.min_request_interval)

    cache_dir = args.cache_dir if args.cache_dir is not None else args.msa_dir
    fig_out = args.fig_out if args.fig_out is not None else args.csv.with_name(
        args.csv.stem + "_msa_metrics.png"
    )
    out_csv = args.output if args.output is not None else args.csv

    log.info("Reading %s", args.csv)
    df = pd.read_csv(args.csv, dtype={args.id_col: str}, keep_default_na=False)
    if args.id_col not in df.columns:
        raise SystemExit(f"{args.csv}: column {args.id_col!r} not found.")
    if args.plddt_col not in df.columns:
        log.warning(
            "%s: pLDDT column %r not in CSV; figure y-axis will be all-NaN.",
            args.csv, args.plddt_col,
        )

    df["__uniprot__"] = df[args.id_col].astype(str).map(normalize_uniprot)
    uniprots = df["__uniprot__"].tolist()
    log.info("%d rows, %d unique uniprots", len(df), len(set(uniprots)))

    msa_paths = acquire_all(
        uniprots=uniprots,
        msa_dir=args.msa_dir,
        cache_dir=cache_dir,
        url_template=args.url_template,
        workers=args.workers,
        no_fetch=args.no_fetch,
        show_progress=not args.quiet,
    )
    src_counts = {"local": 0, "downloaded": 0, "missing": 0}
    for _, (_p, s) in msa_paths.items():
        src_counts[s] += 1
    log.info("MSA sources: %s", src_counts)

    metric_keys = [
        "msa_depth",
        "msa_mean_coverage_top128",
        "msa_std_coverage_top128",
        "msa_mean_identity_top128",
        "msa_std_identity_top128",
    ]
    cache: dict[str, dict] = {}
    metrics_rows = []
    iterator = tqdm(uniprots, desc="parsing", disable=args.quiet)
    for u in iterator:
        if u in cache:
            metrics_rows.append(cache[u])
            continue
        path, _src = msa_paths.get(u, (None, "missing"))
        if path is None:
            m = {k: np.nan for k in metric_keys}
        else:
            try:
                m = compute_metrics(path, args.top_k)
            except Exception as e:
                log.warning("[%s] parse error: %s", u, e)
                m = {k: np.nan for k in metric_keys}
        cache[u] = m
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows, index=df.index)
    for k in metric_keys:
        df[k] = metrics_df[k]
    df = df.drop(columns="__uniprot__")

    missing_unis = sorted(u for u, (p, s) in msa_paths.items() if s == "missing")
    if missing_unis and not args.allow_missing:
        raise SystemExit(f"{len(missing_unis)} MSAs missing — pass --allow-missing to ignore.")

    # Write CSV atomically.
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_csv)
    log.info("Wrote %s", out_csv)

    # Side-car JSON with run params.
    sidecar = out_csv.with_name(out_csv.stem + "_msa_run.json")
    sidecar.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "csv": str(args.csv.resolve()),
                "id_col": args.id_col,
                "plddt_col": args.plddt_col,
                "msa_dir": str(args.msa_dir),
                "cache_dir": str(cache_dir),
                "url_template": args.url_template,
                "top_k": args.top_k,
                "workers": args.workers,
                "n_rows": int(len(df)),
                "n_unique_uniprots": int(len(set(uniprots))),
                "source_counts": src_counts,
                "missing_uniprots": missing_unis,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Wrote sidecar %s", sidecar)

    # Figure.
    if args.plddt_col in df.columns:
        make_figure(df, args.plddt_col, fig_out, title=args.csv.stem)
        log.info("Wrote %s", fig_out)
    else:
        log.warning("Skipping figure: %r not in CSV.", args.plddt_col)


if __name__ == "__main__":
    main()
