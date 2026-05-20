#!/usr/bin/env python3
"""Submit a structure to the public Foldseek search server and compile hits.

Replicates the https://search.foldseek.com web UI flow:
  * --accession PDBID  : fetch PDBID.cif from RCSB, then submit (LOAD ACCESSION path)
  * --file PATH        : submit a local PDB/mmCIF file

Searches every non-FoldDisco database the server exposes and writes:
  <out-dir>/query.cif
  <out-dir>/submission.json
  <out-dir>/hits_compiled.tsv
  <out-dir>/raw/result_entry<N>.json
  <out-dir>/raw/m8/<db>.m8
"""

import argparse
import csv
import hashlib
import io
import json
import pathlib
import re
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone

import requests


DEFAULT_SERVER = "https://search.foldseek.com"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")
VALID_MODES = ("3diaa", "tmalign", "3di", "lolalign")
HITS_COLUMNS = [
    "database", "db_version", "query", "target",
    "seqId", "alnLength", "mismatches", "gapsopened",
    "qStart", "qEnd", "dbStart", "dbEnd",
    "prob", "evalue", "score", "qLen", "dbLen",
    "taxId", "taxName",
    "qAln", "dbAln", "q3di", "t3di", "tCa", "tSeq",
]
# Map (output column) -> (key in alignment JSON). Server uses camelCase + a typo.
ALN_FIELD_MAP = {
    "query": "query", "target": "target",
    "seqId": "seqId", "alnLength": "alnLength",
    "mismatches": "missmatches", "gapsopened": "gapsopened",
    "qStart": "qStartPos", "qEnd": "qEndPos",
    "dbStart": "dbStartPos", "dbEnd": "dbEndPos",
    "prob": "prob", "evalue": "eval", "score": "score",
    "qLen": "qLen", "dbLen": "dbLen",
    "taxId": "taxId", "taxName": "taxName",
    "qAln": "qAln", "dbAln": "dbAln",
    "q3di": "q3di", "t3di": "t3di",
    "tCa": "tCa", "tSeq": "tSeq",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(f"[{now_iso()}] {msg}", flush=True)


def fetch_databases(server):
    r = requests.get(f"{server}/api/databases/all", timeout=30)
    r.raise_for_status()
    dbs = r.json()["databases"]
    # Drop FoldDisco motif-search variants; sort for reproducibility.
    dbs = [d for d in dbs if not d["path"].endswith("_folddisco")]
    dbs.sort(key=lambda d: (d.get("order", 99), d["name"]))
    return dbs


def download_accession(pdb_id, out_path):
    if not PDB_ID_RE.match(pdb_id):
        sys.exit(f"--accession must be a 4-char PDB ID; got {pdb_id!r}")
    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    log(f"Downloading {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    log(f"Saved {len(r.content)} bytes to {out_path}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def submit(server, query_path, mode, n_iter, taxfilter, db_paths):
    with open(query_path, "rb") as f:
        files = {"q": f}
        data = [("mode", mode), ("iterativesearch", str(n_iter))]
        data += [("database[]", d) for d in db_paths]
        if taxfilter:
            data.append(("taxfilter", taxfilter))
        log(f"POST {server}/api/ticket  mode={mode} iter={n_iter} dbs={len(db_paths)}")
        r = requests.post(f"{server}/api/ticket", files=files, data=data, timeout=180)
    r.raise_for_status()
    body = r.json()
    log(f"Submitted: ticket={body['id']} status={body.get('status')}")
    return body["id"]


def poll(server, ticket, max_minutes):
    deadline = time.time() + max_minutes * 60
    last_status = None
    delay = 5
    while time.time() < deadline:
        r = requests.get(f"{server}/api/ticket/{ticket}", timeout=30)
        r.raise_for_status()
        status = r.json().get("status", "UNKNOWN")
        if status != last_status:
            log(f"Ticket {ticket}: {status}")
            last_status = status
        if status == "COMPLETE":
            return status
        if status == "ERROR":
            raise RuntimeError(f"Ticket {ticket} returned ERROR")
        time.sleep(delay)
        # 5s for first ~minute, then 15s.
        if delay < 15 and time.time() - (deadline - max_minutes * 60) > 60:
            delay = 15
    raise TimeoutError(f"Ticket {ticket} did not complete within {max_minutes} min")


def list_query_entries(server, ticket):
    r = requests.get(f"{server}/api/result/queries/{ticket}/100/0", timeout=30)
    r.raise_for_status()
    return r.json().get("lookup", [])


def fetch_entry(server, ticket, entry_id):
    r = requests.get(f"{server}/api/result/{ticket}/{entry_id}", timeout=120)
    r.raise_for_status()
    return r.json()


def download_archive(server, ticket, out_tar):
    log(f"Downloading result archive for ticket {ticket}")
    with requests.get(f"{server}/api/result/download/{ticket}", stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_tar, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    log(f"Saved archive to {out_tar} ({out_tar.stat().st_size} bytes)")


def extract_m8(tar_path, m8_dir):
    m8_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = pathlib.Path(member.name).name
            if not name.endswith(".m8"):
                continue
            with tar.extractfile(member) as src, open(m8_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return sorted(p.name for p in m8_dir.glob("*.m8"))


def compile_hits(entry_jsons, db_meta_by_path, out_tsv):
    # JSON shape (verified against a live response):
    #   entry["queries"]            = list[{"header","sequence"}]
    #   entry["results"][i]["db"]   = database path string
    #   entry["results"][i]["alignments"] = list[list[dict]]
    #     outer index aligns 1:1 with entry["queries"]; inner list = hits per query.
    rows = []
    counts = {}
    for entry in entry_jsons:
        queries = entry.get("queries", []) or []
        for per_db in entry.get("results", []):
            db_path = per_db.get("db", "")
            meta = db_meta_by_path.get(db_path, {})
            db_name = meta.get("name", db_path)
            db_version = meta.get("version", "")
            n = 0
            per_query_alns = per_db.get("alignments", []) or []
            for q_idx, hits in enumerate(per_query_alns):
                q_header = queries[q_idx]["header"] if q_idx < len(queries) else ""
                for aln in hits or []:
                    row = {"database": db_name, "db_version": db_version,
                           "query": aln.get("query", q_header)}
                    for col, src in ALN_FIELD_MAP.items():
                        if col == "query":
                            continue
                        val = aln.get(src, "")
                        row[col] = "" if val is None else val
                    rows.append(row)
                    n += 1
            counts[db_name] = counts.get(db_name, 0) + n
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HITS_COLUMNS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows, counts


def print_summary(counts, rows):
    log(f"Wrote {len(rows)} total hits across {len(counts)} databases")
    print()
    print("| database | n_hits | best_evalue | best_target |")
    print("|---|---|---|---|")
    for db, n in sorted(counts.items()):
        db_rows = [r for r in rows if r["database"] == db]
        if db_rows:
            best = min(db_rows, key=lambda r: float(r.get("evalue", 1e9) or 1e9))
            print(f"| {db} | {n} | {best.get('evalue','')} | {best.get('target','')} |")
        else:
            print(f"| {db} | 0 |  |  |")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--accession", help="4-char PDB ID; script downloads {ID}.cif from RCSB")
    g.add_argument("--file", help="Path to a local PDB or mmCIF file")
    p.add_argument("--mode", choices=VALID_MODES, default="3diaa")
    p.add_argument("--iterative-search", type=int, default=0, dest="iterative_search",
                   help="Number of iterations (server default 0)")
    p.add_argument("--taxfilter", default="", help="Optional NCBI taxonomy filter")
    p.add_argument("--out-dir", help="Output directory (default: ./results/<stem>_<mode>_iter<N>)")
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--poll-max-min", type=int, default=60, dest="poll_max_min")
    p.add_argument("--keep-tar", action="store_true",
                   help="Keep the raw download.tar.gz after extracting M8 files")
    return p.parse_args()


def main():
    args = parse_args()

    if args.accession:
        stem = args.accession.upper()
    else:
        stem = pathlib.Path(args.file).stem
    out_dir = pathlib.Path(args.out_dir or f"results/{stem}_{args.mode}_iter{args.iterative_search}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    if args.accession:
        query_path = out_dir / "query.cif"
        download_accession(args.accession, query_path)
        query_source = f"rcsb:{args.accession.upper()}"
    else:
        src = pathlib.Path(args.file).expanduser().resolve()
        if not src.is_file():
            sys.exit(f"--file not found: {src}")
        ext = src.suffix.lower() or ".pdb"
        query_path = out_dir / f"query{ext}"
        shutil.copyfile(src, query_path)
        query_source = f"file:{src}"

    head = query_path.read_bytes()[:200].decode("utf-8", errors="ignore")
    if not any(tok in head for tok in ("HEADER", "ATOM", "data_", "loop_")):
        sys.exit(f"Query does not look like a PDB/mmCIF file (first 200 chars: {head!r})")

    query_hash = sha256(query_path)
    log(f"Query sha256: {query_hash}")

    dbs = fetch_databases(args.server)
    db_paths = [d["path"] for d in dbs]
    db_meta_by_path = {d["path"]: d for d in dbs}
    log("Selected databases (path | name | version):")
    for d in dbs:
        log(f"  {d['path']:<24} {d['name']:<22} {d['version']}")

    started_at = now_iso()
    ticket = submit(args.server, query_path, args.mode, args.iterative_search, args.taxfilter, db_paths)
    poll(args.server, ticket, args.poll_max_min)
    completed_at = now_iso()

    entries = list_query_entries(args.server, ticket)
    log(f"Found {len(entries)} query entries")
    entry_jsons = []
    for e in entries:
        entry_json = fetch_entry(args.server, ticket, e["id"])
        out_json = raw_dir / f"result_entry{e['id']}.json"
        out_json.write_text(json.dumps(entry_json, indent=2))
        log(f"Wrote {out_json}")
        entry_jsons.append(entry_json)

    tar_path = raw_dir / "download.tar.gz"
    download_archive(args.server, ticket, tar_path)
    m8_files = extract_m8(tar_path, raw_dir / "m8")
    log(f"Extracted {len(m8_files)} M8 files: {m8_files}")
    if not args.keep_tar:
        tar_path.unlink()

    rows, counts = compile_hits(entry_jsons, db_meta_by_path, out_dir / "hits_compiled.tsv")

    manifest = {
        "ticket": ticket,
        "server": args.server,
        "mode": args.mode,
        "iterativesearch": args.iterative_search,
        "taxfilter": args.taxfilter,
        "databases": [{"path": d["path"], "name": d["name"], "version": d["version"]} for d in dbs],
        "query_source": query_source,
        "query_sha256": query_hash,
        "started_at": started_at,
        "completed_at": completed_at,
        "hit_counts_per_db": counts,
        "n_total_hits": len(rows),
    }
    (out_dir / "submission.json").write_text(json.dumps(manifest, indent=2))
    log(f"Wrote {out_dir / 'submission.json'}")

    print_summary(counts, rows)


if __name__ == "__main__":
    main()
