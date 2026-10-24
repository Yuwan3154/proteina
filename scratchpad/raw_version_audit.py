"""Audit flat vs bucketed raw mmCIFs for every PDB id of a pack index (user 2026-10-24: "Yes, audit").

_resolve_raw_sharded checks the flat raw/<id>.cif(.gz) before the bucketed raw/<bucket>/<id>.cif(.gz), so a flat file
(e.g. the 2026-05-27 re-download) shadows the bucketed one a stored pack may have been built from. Per id (spelling as
reprocess_seqres_chains.raw_path resolves it): both paths, existence, decompressed md5, identical flag, and when both
exist and differ the last _pdbx_audit_revision_history.revision_date of each. Read-only on the data tree; writes one TSV.
Usage: python scratchpad/raw_version_audit.py --pack PACK --data-dir DATA_PATH/pdb_train --out OUT.tsv [--workers N]
"""

import argparse
import collections
import gzip
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from proteinfoundation.utils.reprocess_seqres_chains import decompressed_md5, raw_versions

COLUMNS = ("pdb", "flat_path", "flat_exists", "bucket_path", "bucket_exists", "flat_md5", "bucket_md5", "identical",
           "flat_last_revision", "bucket_last_revision")
G = {}


def init(data_dir):
    G["raw"] = pathlib.Path(data_dir) / "raw"
    G["man"] = json.load(open(pathlib.Path(data_dir) / "shard_manifest.json"))


def _line_tokens(line):
    """(token, bare) of one CIF line: a quote closes only before whitespace / EOL; '#' at a token start = comment."""
    if "'" not in line and '"' not in line and "#" not in line:
        for t in line.split():
            yield t, True
        return
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif c == "#":
            return
        elif c in "'\"":
            j = i + 1
            while j < n and not (line[j] == c and (j + 1 == n or line[j + 1].isspace())):
                j += 1
            yield line[i + 1 : j], False
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            yield line[i:j], True
            i = j


def _cif_tokens(f):
    """(token, bare) over a CIF stream; a column-0 ';' text field is one non-bare token."""
    text = None
    for line in f:
        if line.startswith(";"):
            if text is None:
                text = [line[1:].rstrip("\n")]
                continue
            yield "\n".join(text), False
            text = None
            yield from _line_tokens(line[1:])
        elif text is not None:
            text.append(line.rstrip("\n"))
        else:
            yield from _line_tokens(line)


def last_revision(path) -> str:
    """Latest non-null _pdbx_audit_revision_history.revision_date, streamed (single-row and loop_ forms)."""
    cat = "_pdbx_audit_revision_history."
    values, cols, vals, single, pending_loop, seen = [], None, [], None, False, False
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        for tok, bare in _cif_tokens(f):
            low = tok.lower()
            is_tag = bare and tok.startswith("_")
            is_kw = bare and (low == "loop_" or low.startswith(("data_", "save_", "global_", "stop_")))
            if cols is not None:
                if is_tag and tok.startswith(cat) and not vals:
                    cols.append(tok[len(cat):])
                    continue
                if is_tag or is_kw:
                    break
                vals.append((tok, bare))
            elif single is not None:
                if single == "revision_date":
                    values.append((tok, bare))
                single = None
            elif is_kw and low == "loop_":
                pending_loop = True
            elif is_tag and tok.startswith(cat):
                if pending_loop:
                    cols, pending_loop = [tok[len(cat):]], False
                else:
                    seen, single = True, tok[len(cat):]
            elif is_tag and seen:
                break
            else:
                pending_loop = False
    if cols is not None and "revision_date" in cols:
        k, w = cols.index("revision_date"), len(cols)
        values = [vals[r * w + k] for r in range(len(vals) // w)]
    dates = [t for t, bare in values if not (bare and t in ("?", "."))]
    return max(dates) if dates else "absent"


def audit(pdb) -> dict:
    flat, buck = raw_versions(G["raw"], pdb, G["man"])
    r = dict.fromkeys(COLUMNS, "")
    r.update(pdb=pdb, flat_path=flat or "", flat_exists=flat is not None, bucket_path=buck or "",
             bucket_exists=buck is not None)
    if flat is not None:
        r["flat_md5"] = decompressed_md5(flat)
    if buck is not None:
        r["bucket_md5"] = decompressed_md5(buck)
    if flat is not None and buck is not None:
        r["identical"] = r["flat_md5"] == r["bucket_md5"]
        if not r["identical"]:
            r.update(flat_last_revision=last_revision(flat), bucket_last_revision=last_revision(buck))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    data_dir = pathlib.Path(a.data_dir).resolve()
    assert data_dir not in pathlib.Path(a.out).resolve().parents, "read-only on the data tree: write OUT elsewhere"
    idx = np.load(a.pack + ".idx.npz", allow_pickle=True)
    pdbs = sorted({str(s).split("_")[0] for s in idx["stems"]})
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(str(data_dir),)) as ex:
        rows = list(ex.map(audit, pdbs, chunksize=64))
    assert len(rows) == len(pdbs), (len(rows), len(pdbs))
    with open(a.out + ".tmp", "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    os.replace(a.out + ".tmp", a.out)
    c = collections.Counter((r["flat_exists"], r["bucket_exists"], r["identical"]) for r in rows)
    print(f"{len(pdbs)} PDB ids from {len(idx['stems'])} stems -> {a.out}")
    for (fe, be, same), n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  flat {fe!s:5} bucket {be!s:5} identical {same!s:5}: {n} of {len(pdbs)}")


if __name__ == "__main__":
    main()
