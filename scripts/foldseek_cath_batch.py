#!/usr/bin/env python3
"""Batch-run Foldseek-server searches over a list of PDB chains and analyse CATH50 hits.

For each `<PDBID>_<CHAIN>` row in the input CSV:
  1. Download `https://files.rcsb.org/download/<PDBID>.cif`
  2. Extract the requested chain to PDB
  3. Invoke foldseek_server_search.py against all 9 non-FoldDisco DBs in mode `tmalign`
  4. Parse CATH50 hits; map targets to C.A.T.; compute counts + top-k distribution
  5. Touch a per-chain `done.flag` (resume sentinel)

Outputs per-chain JSON analyses plus an aggregated cath50_summary.tsv.
"""

import argparse
import collections
import csv
import json
import pathlib
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

from Bio.PDB import MMCIFParser, PDBIO, Select


CHAIN_ID_RE = re.compile(r"^([A-Za-z0-9]{4})_([A-Za-z0-9]+)$")
AF_TARGET_RE = re.compile(r"^af_[A-Z0-9]+_\d+_\d+_(\d+)\.(\d+)\.(\d+)(?:\.\d+)?$")
CATH_DOMAIN_RE = re.compile(r"^[0-9][a-z0-9]{3}[A-Za-z0-9]\d{2}$")
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
CATH_URL = "https://download.cathdb.info/cath/releases/all-releases/{ver}/cath-classification-data/cath-domain-list-{ver}.txt"

STATUS_DONE = "DONE"
STATUS_PENDING = "PENDING"
STATUS_CHAIN_NOT_FOUND = "CHAIN_NOT_FOUND"
STATUS_DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
STATUS_EXTRACT_FAILED = "EXTRACT_FAILED"
STATUS_FOLDSEEK_FAILED = "FOLDSEEK_FAILED"


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(f"[{now_iso()}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV containing chain IDs in column --csv-col")
    p.add_argument("--csv-col", default="pdb", help="CSV column with chain IDs (default: pdb)")
    p.add_argument("--out-dir", default="foldseek_cath_batch")
    p.add_argument("--mode", default="tmalign", choices=["3diaa", "tmalign", "3di", "lolalign"])
    p.add_argument("--iterative-search", type=int, default=0, dest="iterative_search")
    p.add_argument("--top-k", type=int, default=10, dest="top_k")
    p.add_argument("--tm-threshold", type=float, default=0.5, dest="tm_threshold")
    p.add_argument("--cath-version", default="v4_3_0", dest="cath_version")
    p.add_argument("--chains", default="", help="Optional comma-separated subset of chain IDs")
    p.add_argument("--retry-failed", action="store_true", dest="retry_failed",
                   help="Re-run anything not DONE (default: re-run only PENDING)")
    p.add_argument("--script", default=str(pathlib.Path.home() / "proteina/scripts/foldseek_server_search.py"))
    return p.parse_args()


def read_chain_ids(csv_path, col, subset):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if col not in rows[0]:
        sys.exit(f"Column {col!r} not in CSV columns {list(rows[0])}")
    chain_ids = [r[col].strip() for r in rows if r[col].strip()]
    if subset:
        wanted = {c.strip() for c in subset.split(",") if c.strip()}
        chain_ids = [c for c in chain_ids if c in wanted]
        missing = wanted - set(chain_ids)
        if missing:
            log(f"WARNING: --chains specified IDs not in CSV: {sorted(missing)}")
    return chain_ids


def ensure_cath_lookup(out_dir, ver):
    cache = out_dir / f"cath_lookup_{ver}.txt"
    if not cache.exists():
        url = CATH_URL.format(ver=ver)
        log(f"Downloading {url}")
        urllib.request.urlretrieve(url, cache)
        log(f"Saved {cache} ({cache.stat().st_size} bytes)")
    lookup = {}
    with open(cache) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            domain, c, a, t = parts[0], parts[1], parts[2], parts[3]
            lookup[domain] = f"{c}.{a}.{t}"
    log(f"CATH lookup loaded: {len(lookup)} domain IDs")
    return lookup


def split_chain_id(chain_id):
    m = CHAIN_ID_RE.match(chain_id)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2)


def download_cif(pdb_id, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = RCSB_CIF_URL.format(pdb_id=pdb_id)
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "foldseek_cath_batch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    except urllib.error.HTTPError as e:
        log(f"  HTTP {e.code} for {pdb_id}")
        return False
    return True


class _ChainSelect(Select):
    def __init__(self, chain):
        self._chain = chain
    def accept_chain(self, c):
        return c.id == self._chain
    def accept_residue(self, r):
        # Keep only standard amino-acid residues (hetflag empty), drop waters/ligands.
        return r.id[0] == " "


def extract_chain(cif_path, chain_letter, out_pdb):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    model = next(structure.get_models())
    if chain_letter not in {c.id for c in model.get_chains()}:
        return False
    io = PDBIO()
    # First model only: saving the whole structure emits every NMR model, which
    # foldseek then indexes as separate entries. Model 1 matches the single-model
    # native used for the TM-based y-axis.
    io.set_structure(model)
    io.save(str(out_pdb), _ChainSelect(chain_letter))
    with open(out_pdb) as f:
        has_atoms = any(line.startswith("ATOM") for line in f)
    return has_atoms


def target_to_cat(target, cath_lookup):
    m = AF_TARGET_RE.match(target)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    # PDB-domain style, e.g. 4fprA00; also accept 4fprA01 etc.
    head = target.split()[0]
    if CATH_DOMAIN_RE.match(head) and head in cath_lookup:
        return cath_lookup[head]
    return "UNK"


def analyse_cath50(chain_id, chain_dir, cath_lookup, top_k, tm_threshold):
    # In Foldseek `tmalign` mode the `prob` column carries the TM-score (0..1);
    # the `score` column is a bits-like int. We verified this empirically on a
    # 7AD5_A run by comparing column 11 values to TM-align scaling.
    tsv = chain_dir / "hits_compiled.tsv"
    cath = []
    with open(tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["database"] != "CATH50":
                continue
            try:
                row["_tm"] = float(row.get("prob", "") or "nan")
            except ValueError:
                row["_tm"] = float("nan")
            cath.append(row)
    cath.sort(key=lambda r: (r["_tm"] if r["_tm"] == r["_tm"] else -1.0), reverse=True)
    n_total = len(cath)
    n_sig = sum(1 for r in cath if r["_tm"] == r["_tm"] and r["_tm"] > tm_threshold)
    score_min = min((r["_tm"] for r in cath if r["_tm"] == r["_tm"]), default=float("nan"))
    score_max = max((r["_tm"] for r in cath if r["_tm"] == r["_tm"]), default=float("nan"))
    top = cath[:top_k]
    top_cats = [target_to_cat(r["target"], cath_lookup) for r in top]
    dist = dict(collections.Counter(top_cats))
    top1 = top[0] if top else None
    return {
        "chain_id": chain_id,
        "n_cath50_hits": n_total,
        "n_significant_tm_gt_{}".format(tm_threshold): n_sig,
        "tm_threshold": tm_threshold,
        "top_k": top_k,
        "score_min": score_min,
        "score_max": score_max,
        "top1_target": top1["target"] if top1 else "",
        "top1_cat": top_cats[0] if top_cats else "",
        "top1_tm": top1["_tm"] if top1 else float("nan"),
        "top10_distinct_cats": len(set(top_cats)),
        "top10_cat_distribution": dist,
    }


def chain_status(chain_dir):
    if (chain_dir / "done.flag").exists():
        return STATUS_DONE
    for status_flag in (STATUS_CHAIN_NOT_FOUND, STATUS_DOWNLOAD_FAILED, STATUS_EXTRACT_FAILED, STATUS_FOLDSEEK_FAILED):
        if (chain_dir / f"status_{status_flag}.flag").exists():
            return status_flag
    return STATUS_PENDING


def write_status(chain_dir, status, error=""):
    for f in chain_dir.glob("status_*.flag"):
        f.unlink()
    if status != STATUS_DONE and status != STATUS_PENDING:
        (chain_dir / f"status_{status}.flag").write_text(error or "")


def process_chain(chain_id, args, root, cath_lookup):
    chain_dir = root / "chains" / chain_id
    chain_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"{chain_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing = chain_status(chain_dir)
    if existing == STATUS_DONE and not args.retry_failed:
        log(f"{chain_id}: SKIP (DONE)")
        return STATUS_DONE
    if existing not in (STATUS_PENDING, STATUS_DONE) and not args.retry_failed:
        log(f"{chain_id}: SKIP (status={existing}; pass --retry-failed to redo)")
        return existing

    pdb_id, chain_letter = split_chain_id(chain_id)
    if pdb_id is None:
        log(f"{chain_id}: bad chain ID format")
        write_status(chain_dir, STATUS_CHAIN_NOT_FOUND, "bad chain ID format")
        return STATUS_CHAIN_NOT_FOUND

    cif_path = chain_dir / "full.cif"
    if not download_cif(pdb_id, cif_path):
        write_status(chain_dir, STATUS_DOWNLOAD_FAILED, f"RCSB fetch failed for {pdb_id}")
        return STATUS_DOWNLOAD_FAILED

    query_pdb = chain_dir / "query.pdb"
    if not extract_chain(cif_path, chain_letter, query_pdb):
        write_status(chain_dir, STATUS_CHAIN_NOT_FOUND, f"chain {chain_letter} not found in {pdb_id}")
        return STATUS_CHAIN_NOT_FOUND

    cmd = [sys.executable, args.script,
           "--file", str(query_pdb),
           "--mode", args.mode,
           "--iterative-search", str(args.iterative_search),
           "--out-dir", str(chain_dir)]
    log(f"{chain_id}: launching foldseek subprocess")
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log(f"{chain_id}: foldseek subprocess exit {proc.returncode}; see {log_path}")
        write_status(chain_dir, STATUS_FOLDSEEK_FAILED, f"exit code {proc.returncode}")
        return STATUS_FOLDSEEK_FAILED

    analysis = analyse_cath50(chain_id, chain_dir, cath_lookup, args.top_k, args.tm_threshold)
    (chain_dir / "cath50_analysis.json").write_text(json.dumps(analysis, indent=2))
    (chain_dir / "done.flag").write_text("")
    write_status(chain_dir, STATUS_DONE)
    log(f"{chain_id}: DONE  cath50_hits={analysis['n_cath50_hits']}  "
        f"sig_TM>{args.tm_threshold}={analysis[f'n_significant_tm_gt_{args.tm_threshold}']}  "
        f"top1_cat={analysis['top1_cat']}  top1_tm={analysis['top1_tm']:.3f}")
    return STATUS_DONE


def rebuild_manifest(root, chain_ids, tm_threshold):
    manifest_path = root / "manifest.tsv"
    summary_path = root / "cath50_summary.tsv"
    sig_col = f"n_significant_tm_gt_{tm_threshold}"
    manifest_cols = ["chain_id", "status", "n_cath50_hits", sig_col,
                     "top1_cat", "top1_tm", "top1_target", "top10_distinct_cats"]
    summary_cols = manifest_cols + ["top10_cat_distribution_json", "score_min", "score_max"]
    status_counts = collections.Counter()
    manifest_rows = []
    summary_rows = []
    for cid in chain_ids:
        cdir = root / "chains" / cid
        status = chain_status(cdir) if cdir.exists() else STATUS_PENDING
        status_counts[status] += 1
        if status == STATUS_DONE and (cdir / "cath50_analysis.json").exists():
            a = json.loads((cdir / "cath50_analysis.json").read_text())
            manifest_rows.append({
                "chain_id": cid, "status": status,
                "n_cath50_hits": a["n_cath50_hits"],
                sig_col: a[sig_col],
                "top1_cat": a["top1_cat"], "top1_tm": f"{a['top1_tm']:.4f}",
                "top1_target": a["top1_target"],
                "top10_distinct_cats": a["top10_distinct_cats"],
            })
            summary_rows.append({**manifest_rows[-1],
                "top10_cat_distribution_json": json.dumps(a["top10_cat_distribution"], sort_keys=True),
                "score_min": f"{a['score_min']:.4f}" if a["score_min"] == a["score_min"] else "",
                "score_max": f"{a['score_max']:.4f}" if a["score_max"] == a["score_max"] else "",
            })
        else:
            empty = {c: "" for c in manifest_cols}
            empty.update({"chain_id": cid, "status": status})
            manifest_rows.append(empty)
            summary_rows.append({**empty, "top10_cat_distribution_json": "", "score_min": "", "score_max": ""})
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_cols, delimiter="\t")
        w.writeheader(); w.writerows(manifest_rows)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols, delimiter="\t")
        w.writeheader(); w.writerows(summary_rows)
    return status_counts, manifest_rows


def main():
    args = parse_args()
    root = pathlib.Path(args.out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "chains").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    chain_ids = read_chain_ids(args.csv, args.csv_col, args.chains)
    log(f"Processing {len(chain_ids)} chain(s) from {args.csv}")

    cath_lookup = ensure_cath_lookup(root, args.cath_version)

    for cid in chain_ids:
        # Top-level try/except is required by the stop-and-resume contract: a single
        # bad chain must not stop the run. (Explicit user requirement.)
        try:
            process_chain(cid, args, root, cath_lookup)
        except Exception as e:
            log(f"{cid}: UNHANDLED {type(e).__name__}: {e}")
            cdir = root / "chains" / cid
            cdir.mkdir(parents=True, exist_ok=True)
            write_status(cdir, STATUS_FOLDSEEK_FAILED, f"{type(e).__name__}: {e}")

    status_counts, _ = rebuild_manifest(root, chain_ids, args.tm_threshold)
    sig_col = f"n_significant_tm_gt_{args.tm_threshold}"

    # Final stdout report.
    print()
    print("| metric | value |")
    print("|---|---|")
    print(f"| chains attempted | {len(chain_ids)} |")
    for st in [STATUS_DONE, STATUS_FOLDSEEK_FAILED, STATUS_CHAIN_NOT_FOUND,
               STATUS_DOWNLOAD_FAILED, STATUS_EXTRACT_FAILED, STATUS_PENDING]:
        if status_counts.get(st, 0):
            print(f"| {st} | {status_counts[st]} |")
    n_any = n_sig = n_zero = 0
    novel_candidates = []
    for cid in chain_ids:
        a_path = root / "chains" / cid / "cath50_analysis.json"
        if a_path.exists():
            a = json.loads(a_path.read_text())
            if a["n_cath50_hits"] > 0:
                n_any += 1
            if a[sig_col] > 0:
                n_sig += 1
            if a["n_cath50_hits"] == 0:
                novel_candidates.append(cid)
            elif a[sig_col] == 0:
                novel_candidates.append(cid)
    print(f"| chains with >=1 CATH50 hit | {n_any} |")
    print(f"| chains with >=1 significant CATH50 hit (TM>{args.tm_threshold}) | {n_sig} |")
    print(f"| chains with no significant CATH50 hit (novelty candidates) | {len(novel_candidates)} |")
    if novel_candidates:
        print()
        print("Novelty candidates (no TM>{:.1f} CATH50 hit):".format(args.tm_threshold))
        for cid in novel_candidates:
            a = json.loads((root / "chains" / cid / "cath50_analysis.json").read_text())
            print(f"  {cid}  total_cath50={a['n_cath50_hits']}  top1_tm={a['top1_tm']:.3f}  top1_cat={a['top1_cat']}")
    log(f"Wrote {root / 'manifest.tsv'} and {root / 'cath50_summary.tsv'}")


if __name__ == "__main__":
    main()
