#!/usr/bin/env python
"""Structural distance of each validation-target native to the training set.

For every validation target's NATIVE structure, find its single closest structural
match in the model's EXACT PDB training split (foldseek TM-align) and record the
TM-score and the paired RMSD to that closest training chain. This is the
"novelty / memorization" x-axis: high closeness = the fold was likely memorized.

Output ``train_closeness.csv`` is keyed by ``protein_id`` (``{pdb}_{chain}``) so it
merges directly into ``cross_protein_summary_data.csv``.

Stages (each guarded by a sentinel under ``--work_dir`` so the run is resumable):
  1. train_ids.txt   exact train split via the Hydra datamodule (honors the
                     deposition cutoff + cluster-aware exclusions, which the cached
                     cath_codes CSV does NOT encode).
  2. train_pdb/      per-chain backbone PDBs written from the processed .pt files.
  3. db/trainDB      foldseek structural DB (built once; reusable across val sets).
  4. query_pdb/      native chain PDBs for this validation set.
  5. hits.m8         foldseek easy-search --alignment-type 1 (TM-align).
  6. train_closeness.csv

Run in the ``proteina`` conda env (needs torch + hydra + biopython + pandas).
foldseek comes from a dedicated env or ``--foldseek_bin``.

NOTE on disk: a full train DB (~150k chains) needs roughly 7-11 GB of scratch for
the per-chain PDBs plus the DB. Point ``--work_dir`` at a filesystem with room
(NEVER a charged shared mount). Use ``--smoke_db_size`` for a quick end-to-end test.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

_THIS = Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]  # proteinfoundation/prediction_pipeline/ -> repo root
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

# Reused, proven helpers (see plan reuse map).
from scripts.analysis.mark_in_train import load_train_ids, normalize_pdb_chain  # noqa: E402
from scripts.analysis.audit_broken_chains import _write_example_pdb  # noqa: E402
from scripts.foldseek_cath_batch import extract_chain  # noqa: E402

log = logging.getLogger("train_closeness")

_M8_COLS = [
    "query", "target", "alntmscore", "qtmscore", "ttmscore",
    "rmsd", "lddt", "alnlen", "qlen", "tlen",
]


# --------------------------------------------------------------------------- #
# Stage 1: exact train split
# --------------------------------------------------------------------------- #
def materialize_train_ids(work_dir: Path, config_name: str, config_subdir: str,
                          data_path: str) -> list[str]:
    cache = work_dir / "train_ids.txt"
    if cache.exists() and cache.stat().st_size > 0:
        ids = [ln.strip() for ln in cache.read_text().splitlines() if ln.strip()]
        log.info("train_ids: %d (cached %s)", len(ids), cache)
        return ids
    # DATA_PATH must be set before load_train_ids triggers Hydra/oc.env resolution.
    os.environ["DATA_PATH"] = data_path
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    log.info("Materializing train split (config=%s, DATA_PATH=%s) ...", config_name, data_path)
    ids = sorted(load_train_ids(config_name, config_subdir))
    cache.write_text("\n".join(ids) + "\n")
    log.info("train_ids: %d (wrote %s)", len(ids), cache)
    return ids


# --------------------------------------------------------------------------- #
# Stage 2: per-chain train PDBs (.pt -> backbone PDB)
# --------------------------------------------------------------------------- #
def _pdb_worker(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    train_id, pt_path, out_pdb = args
    try:
        if not os.path.exists(pt_path):
            return train_id, False, "missing_pt"
        _write_example_pdb(Path(pt_path), Path(out_pdb))
        ok = os.path.exists(out_pdb) and os.path.getsize(out_pdb) > 0
        return train_id, ok, "" if ok else "empty_pdb"
    except Exception as e:  # noqa: BLE001 - a single bad .pt must not kill the sweep
        return train_id, False, repr(e)[:200]


def _build_chunk_pdbs(chunk_ids: list[str], processed_dir: Path, out_dir: Path, n_workers: int) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(tid, str(processed_dir / f"{tid}.pt"), str(out_dir / f"{tid}.pdb")) for tid in chunk_ids]
    n_ok = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for _tid, ok, _err in ex.map(_pdb_worker, jobs):
            if ok:
                n_ok += 1
    return n_ok


# --------------------------------------------------------------------------- #
# Stage 3: foldseek DB — chunked build to keep peak disk small (the per-chain
# PDBs are the disk hog; build a chunk, fold it into a persistent DB via
# concatdbs, delete the chunk's PDBs, repeat). Peak ~= one chunk of PDBs + the
# (compact) growing DB, instead of all ~160k PDBs at once.
# --------------------------------------------------------------------------- #
_FS_COMPONENTS = ["", "_h", "_ss", "_ca"]  # the sub-DBs a foldseek structure DB is made of


def _rm_prefix(prefix: str) -> None:
    import glob as _glob
    for f in _glob.glob(prefix + "*"):
        try:
            os.remove(f)
        except OSError:
            pass


def _mv_prefix(src: str, dst: str) -> None:
    import glob as _glob
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    for f in _glob.glob(src + "*"):
        shutil.move(f, dst + f[len(src):])


def build_db_chunked(foldseek: str, train_ids: list[str], processed_dir: Path, db_path: Path,
                     tmp_dir: Path, chunk_size: int, n_workers: int, rebuild: bool,
                     min_free_gb: float = 3.0) -> None:
    sentinel = db_path.parent / "db.done"
    if sentinel.exists() and not rebuild:
        log.info("foldseek DB present (%s); skipping build", db_path)
        return
    work = db_path.parent
    work.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    chunk_pdb = work / "_chunk_pdb"
    chunk_db = str(work / "_chunk_db" / "cdb")
    acc = str(db_path)
    _rm_prefix(acc)  # fresh build
    n_chunks = (len(train_ids) + chunk_size - 1) // chunk_size
    log.info("Chunked DB build: %d chains, %d chunks of %d", len(train_ids), n_chunks, chunk_size)
    have_acc = False
    for ci in range(n_chunks):
        free_gb = shutil.disk_usage(work).free / 1e9
        if free_gb < min_free_gb:
            raise SystemExit(f"Aborting: only {free_gb:.1f} GB free (< {min_free_gb}); DB build would risk filling the disk.")
        chunk = train_ids[ci * chunk_size:(ci + 1) * chunk_size]
        n_ok = _build_chunk_pdbs(chunk, processed_dir, chunk_pdb, n_workers)
        _rm_prefix(chunk_db)
        Path(chunk_db).parent.mkdir(parents=True, exist_ok=True)
        _run([foldseek, "createdb", str(chunk_pdb), chunk_db], tmp_dir)
        if not have_acc:
            _mv_prefix(chunk_db, acc)
            have_acc = True
        else:
            new = str(work / "_acc_new" / "cdb")
            _rm_prefix(new)
            for comp in _FS_COMPONENTS:
                _run([foldseek, "concatdbs", acc + comp, chunk_db + comp, new + comp], tmp_dir)
            _rm_prefix(acc)
            _mv_prefix(new, acc)
        _rm_prefix(chunk_db)
        shutil.rmtree(chunk_pdb, ignore_errors=True)
        log.info("  chunk %d/%d: +%d chains (free=%.1f GB)", ci + 1, n_chunks, n_ok,
                 shutil.disk_usage(work).free / 1e9)
    for d in ["_chunk_pdb", "_chunk_db", "_acc_new"]:
        shutil.rmtree(work / d, ignore_errors=True)
    sentinel.write_text("ok\n")
    log.info("foldseek DB built (chunked): %s", db_path)


# --------------------------------------------------------------------------- #
# Stage 4: query native PDBs
# --------------------------------------------------------------------------- #
def _find_reference_cif(protein_id: str, cif_dir: Path) -> Optional[Path]:
    """Resolve {cif_dir}/{pdb}.cif, directly or one level down (sharded layout).

    Mirrors proteina_analysis._find_reference_cif so natives resolve identically.
    """
    pdb_id = protein_id.split("_")[0]
    direct = cif_dir / f"{pdb_id}.cif"
    if direct.exists():
        return direct
    for child in sorted(p for p in cif_dir.iterdir() if p.is_dir()):
        cand = child / f"{pdb_id}.cif"
        if cand.exists():
            return cand
    return None


def _chain_of(protein_id: str) -> str:
    return protein_id.split("_", 1)[1] if "_" in protein_id else "A"


def build_query_pdbs(ids: list[str], cif_dir: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    n_missing_cif, n_no_chain = 0, 0
    for pid in ids:
        out_pdb = out_dir / f"{pid}.pdb"
        if out_pdb.exists() and out_pdb.stat().st_size > 0:
            built.append(pid)
            continue
        cif = _find_reference_cif(pid, cif_dir)
        if cif is None:
            n_missing_cif += 1
            continue
        ok = extract_chain(str(cif), _chain_of(pid), str(out_pdb))
        if ok:
            built.append(pid)
        else:
            n_no_chain += 1
            if out_pdb.exists():
                out_pdb.unlink()
    log.info("query_pdb: %d built (%d missing CIF, %d chain-extract failed) -> %s",
             len(built), n_missing_cif, n_no_chain, out_dir)
    return built


# --------------------------------------------------------------------------- #
# Stage 5/6: search + reduce
# --------------------------------------------------------------------------- #
def run_easy_search(foldseek: str, query_dir: Path, db_path: Path, out_m8: Path,
                    tmp_dir: Path, threads: int, max_seqs: int, evalue: float) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        foldseek, "easy-search", str(query_dir), str(db_path), str(out_m8), str(tmp_dir),
        "--alignment-type", "1",            # TM-align: real TM-score + RMSD
        "--tmscore-threshold", "0.0",
        "-e", str(evalue),
        "--max-seqs", str(max_seqs),
        "--format-output", ",".join(_M8_COLS),
        "--threads", str(threads),
    ]
    log.info("foldseek easy-search %s vs %s -> %s", query_dir, db_path, out_m8)
    _run(cmd, tmp_dir)


def _strip_pdb(name: str) -> str:
    n = str(name)
    if n.endswith(".pdb"):
        n = n[:-4]
    return n


def reduce_best_hits(out_m8: Path, query_ids: list[str], train_ids: set[str],
                     tm_norm: str, keep_self: bool) -> pd.DataFrame:
    if not out_m8.exists() or out_m8.stat().st_size == 0:
        hits = pd.DataFrame(columns=_M8_COLS)
    else:
        hits = pd.read_csv(out_m8, sep="\t", names=_M8_COLS)
    rows = []
    if len(hits):
        hits = hits.copy()
        hits["query_id"] = hits["query"].map(_strip_pdb)
        hits["target_id"] = hits["target"].map(_strip_pdb)
        for c in ["qtmscore", "ttmscore", "alntmscore", "rmsd"]:
            hits[c] = pd.to_numeric(hits[c], errors="coerce")
        if not keep_self:
            same = hits["query_id"].map(normalize_pdb_chain) == hits["target_id"].map(normalize_pdb_chain)
            hits = hits[~same]
        if tm_norm == "qtm":
            hits["_tm"] = hits["qtmscore"]
        elif tm_norm == "ttm":
            hits["_tm"] = hits["ttmscore"]
        else:  # max
            hits["_tm"] = hits[["qtmscore", "ttmscore"]].max(axis=1)
        hits = hits.dropna(subset=["_tm"])
        if len(hits):
            best_idx = hits.groupby("query_id")["_tm"].idxmax()
            best = hits.loc[best_idx].set_index("query_id")
            for qid in best.index:
                r = best.loc[qid]
                rows.append({
                    "protein_id": qid,
                    "train_closest_tm": float(r["_tm"]),
                    "train_closest_rmsd": float(r["rmsd"]),
                    "closest_train_id": str(r["target_id"]),
                    "train_closest_qtm": float(r["qtmscore"]),
                    "train_closest_ttm": float(r["ttmscore"]),
                    "is_self_in_train": normalize_pdb_chain(qid) in train_ids,
                })
    found = {row["protein_id"] for row in rows}
    # Queries with no surviving hit = maximally novel (far from training). tm=0.
    for qid in query_ids:
        if qid not in found:
            rows.append({
                "protein_id": qid,
                "train_closest_tm": 0.0,
                "train_closest_rmsd": float("nan"),
                "closest_train_id": "",
                "train_closest_qtm": 0.0,
                "train_closest_ttm": 0.0,
                "is_self_in_train": normalize_pdb_chain(qid) in train_ids,
            })
    return pd.DataFrame(rows).sort_values("protein_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def _run(cmd: list[str], tmp_dir: Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)  # keep foldseek scratch off the (full) root fs
    subprocess.run(cmd, check=True, env=env)


def _resolve_foldseek(arg: Optional[str]) -> str:
    if arg:
        return arg
    found = shutil.which("foldseek")
    if found:
        return found
    env_bin = Path.home() / "miniforge3" / "envs" / "foldseek" / "bin" / "foldseek"
    if env_bin.exists():
        return str(env_bin)
    raise SystemExit(
        "foldseek not found. Pass --foldseek_bin, or create a dedicated env:\n"
        "  conda create -y -n foldseek -c conda-forge -c bioconda foldseek"
    )


def _disk_check(work_dir: Path, need_gb: float) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(work_dir).free / 1e9
    log.info("work_dir=%s  free=%.1f GB", work_dir, free_gb)
    if free_gb < need_gb:
        log.warning("Only %.1f GB free at %s (recommend >= %.0f GB for the full DB). "
                    "Consider --smoke_db_size or a roomier --work_dir / another host.",
                    free_gb, work_dir, need_gb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids_csv", required=True, help="Validation manifest CSV.")
    ap.add_argument("--id_col", default="natives_rcsb", help="ID column (default: natives_rcsb).")
    ap.add_argument("--cif_dir", required=True, help="Directory of native ground-truth CIFs.")
    ap.add_argument("--work_dir", required=True,
                    help="Scratch for train_ids/train_pdb/db/query_pdb/hits (NOT a charged shared mount).")
    ap.add_argument("--data_path", default=os.environ.get("DATA_PATH") or str(Path.home() / "proteina/data"),
                    help="DATA_PATH for the datamodule (default: $DATA_PATH or ~/proteina/data).")
    ap.add_argument("--processed_dir", default=None,
                    help="Train processed .pt dir (default: <data_path>/pdb_train/processed).")
    ap.add_argument("--config_name", default="pdb_train_S25_max256_purge-test_cutoff-190828")
    ap.add_argument("--config_subdir", default="pdb")
    ap.add_argument("--foldseek_bin", default=None)
    ap.add_argument("--tm_norm", choices=["qtm", "ttm", "max"], default="qtm",
                    help="TM normalization for 'closest' (default qtm = normalized by the native query).")
    ap.add_argument("--keep_self_hits", action="store_true",
                    help="Keep self-matches (in-train targets match themselves at TM~1).")
    ap.add_argument("--rebuild_db", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=8000,
                    help="Train chains per chunk for the low-disk chunked DB build (peak ~= one chunk of PDBs).")
    ap.add_argument("--smoke_db_size", type=int, default=0,
                    help="If >0, build the DB over only the first N train chains (quick e2e test).")
    ap.add_argument("--n_workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--max_seqs", type=int, default=3000)
    ap.add_argument("--evalue", type=float, default=100.0)
    ap.add_argument("--out_csv", default=None, help="Output CSV (default: <work_dir>/train_closeness.csv).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s", stream=sys.stderr)

    work_dir = Path(args.work_dir).expanduser()
    processed_dir = Path(args.processed_dir).expanduser() if args.processed_dir \
        else Path(args.data_path).expanduser() / "pdb_train" / "processed"
    foldseek = _resolve_foldseek(args.foldseek_bin)
    _disk_check(work_dir, need_gb=12.0 if not args.smoke_db_size else 1.0)

    # Stage 1: train ids
    train_ids = materialize_train_ids(work_dir, args.config_name, args.config_subdir, args.data_path)
    train_ids_set = set(train_ids)
    if args.smoke_db_size and args.smoke_db_size > 0:
        train_ids = train_ids[: args.smoke_db_size]
        log.info("SMOKE: restricting DB to first %d train chains", len(train_ids))

    # Stage 2+3: chunked foldseek DB build (low peak disk; persistent + reusable)
    db_path = work_dir / ("db_smoke" if args.smoke_db_size else "db") / "trainDB"
    tmp_dir = work_dir / "fs_tmp"
    build_db_chunked(foldseek, train_ids, processed_dir, db_path, tmp_dir,
                     args.chunk_size, args.n_workers, args.rebuild_db)

    # Stage 4: query native PDBs
    ids = pd.read_csv(args.ids_csv, dtype={args.id_col: str}, keep_default_na=False)[args.id_col].astype(str).tolist()
    ids = [i for i in dict.fromkeys(ids) if i]  # de-dupe, keep order
    query_dir = work_dir / "query_pdb"
    built_ids = build_query_pdbs(ids, Path(args.cif_dir).expanduser(), query_dir)
    if not built_ids:
        raise SystemExit("No query PDBs were built — check --cif_dir / --id_col.")

    # Stage 5: search
    out_m8 = work_dir / "hits.m8"
    run_easy_search(foldseek, query_dir, db_path, out_m8, tmp_dir,
                    args.threads, args.max_seqs, args.evalue)

    # Stage 6: reduce
    df = reduce_best_hits(out_m8, built_ids, train_ids_set, args.tm_norm, args.keep_self_hits)
    out_csv = Path(args.out_csv).expanduser() if args.out_csv else work_dir / "train_closeness.csv"
    df.to_csv(out_csv, index=False)
    n_hit = int((df["train_closest_tm"] > 0).sum())
    log.info("Wrote %s (%d rows; %d with a hit; %d self-in-train)",
             out_csv, len(df), n_hit, int(df["is_self_in_train"].sum()))


if __name__ == "__main__":
    main()
