"""Stamp CSV rows with an ``in_train`` boolean against a Hydra-configured train split.

Reproduces (and supersedes) the ad-hoc cells in ``proteina.ipynb`` that did::

    train_id = set(pdb_datamodule._get_dataset("train").file_names)
    df["in_train"] = df[id_col].apply(lower_pdb).isin(train_id)

The training set is materialised exactly once via the dataset config (default
``pdb_train_S25_max256_purge-test_cutoff-190828``), so a single invocation can
update many CSVs.

Usage::

    # one CSV
    python scripts/analysis/mark_in_train.py \\
        --input /home/ubuntu/data/af2rank_single/af2rank_single_set_combined_tms.csv \\
        --id-col natives_rcsb

    # every *.csv in a directory (rewrites in place via atomic rename)
    python scripts/analysis/mark_in_train.py \\
        --input-dir /home/ubuntu/data/af2rank_single/ \\
        --id-col natives_rcsb
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import List, Optional

import pandas as pd

_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

log = logging.getLogger("mark_in_train")


def normalize_pdb_chain(s: str) -> str:
    """Lowercase the PDB code before the first underscore; keep the rest verbatim.

    Matches the dataset's ``file_names`` entries (e.g. ``4k08_A``, ``6sg9_Ci``).
    Inputs without an underscore are lowercased whole — best-effort fallback.
    """
    s = str(s)
    if "_" not in s:
        return s.lower()
    pdb, rest = s.split("_", 1)
    return f"{pdb.lower()}_{rest}"


def load_train_ids(config_name: str, config_subdir: str = "pdb") -> set[str]:
    """Hydra-compose the dataset config and pull ``file_names`` for the train split."""
    import hydra  # noqa: E402  (delay heavy imports until after env setup)
    from hydra.core.global_hydra import GlobalHydra

    config_dir = _PROTEINA_ROOT / "configs" / "datasets_config" / config_subdir
    if not config_dir.exists():
        raise SystemExit(f"config_dir does not exist: {config_dir}")

    log.info("config_dir=%s config_name=%s", config_dir, config_name)

    # Re-entrancy: clear any prior global Hydra state so the context manager works.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with hydra.initialize_config_dir(
        config_dir=str(config_dir), version_base=hydra.__version__
    ):
        cfg = hydra.compose(config_name=config_name)
    dm = hydra.utils.instantiate(cfg.datamodule)

    log.info("dm.prepare_data() ...")
    dm.prepare_data()
    log.info("dm.setup('fit') ...")
    dm.setup("fit")

    train_ds = dm._get_dataset("train")
    file_names = train_ds.file_names or []
    log.info("|train| = %d entries", len(file_names))
    return set(file_names)


def discover_csvs(input_dir: pathlib.Path, glob: str) -> List[pathlib.Path]:
    paths = sorted(p for p in input_dir.glob(glob) if p.is_file())
    return paths


def annotate_csv(
    path: pathlib.Path,
    id_col: str,
    train_ids: set[str],
    output: Optional[pathlib.Path] = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Add ``in_train`` to ``path``, returning (n_in_train, n_rows)."""
    df = pd.read_csv(path, dtype={id_col: str}, keep_default_na=False)
    if id_col not in df.columns:
        raise SystemExit(f"{path}: column {id_col!r} not in {list(df.columns)}")

    normalized = df[id_col].astype(str).map(normalize_pdb_chain)
    df["in_train"] = normalized.isin(train_ids)
    n_hit = int(df["in_train"].sum())
    n_rows = len(df)

    if dry_run:
        log.info("[dry-run] %s -> %d/%d in_train", path.name, n_hit, n_rows)
        return n_hit, n_rows

    dest = output if output is not None else path
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    log.info("%s -> %d/%d in_train (wrote %s)", path.name, n_hit, n_rows, dest)
    return n_hit, n_rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stamp `in_train` boolean onto CSVs using a Hydra-configured train split."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=pathlib.Path, help="Single CSV path to annotate.")
    src.add_argument(
        "--input-dir", type=pathlib.Path, help="Directory of CSVs to annotate (in place)."
    )
    ap.add_argument(
        "--id-col",
        default="natives_rcsb",
        help="Column with PDB_CHAIN-style IDs (default: natives_rcsb).",
    )
    ap.add_argument(
        "--config-name",
        default="pdb_train_S25_max256_purge-test_cutoff-190828",
        help="Hydra config basename (no .yaml).",
    )
    ap.add_argument(
        "--config-subdir",
        default="pdb",
        help="Subdir under configs/datasets_config/ (default: pdb).",
    )
    ap.add_argument(
        "--glob",
        default="*.csv",
        help="Glob for --input-dir mode (default: *.csv).",
    )
    ap.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output CSV path (only with --input). Defaults to overwriting --input.",
    )
    ap.add_argument(
        "--data-path",
        default=None,
        help=(
            "DATA_PATH for Hydra ${oc.env:DATA_PATH} (default: $DATA_PATH or "
            f"{pathlib.Path.home() / 'proteina/data'})."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Don't write CSVs, just report counts.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--quiet", action="store_true")
    grp.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="[%(name)s] %(message)s", stream=sys.stderr)

    if args.output is not None and args.input is None:
        ap.error("--output requires --input")

    # Set Hydra env vars BEFORE importing hydra/proteinfoundation.
    data_path = args.data_path or os.environ.get("DATA_PATH") or str(pathlib.Path.home() / "proteina/data")
    os.environ["DATA_PATH"] = data_path
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    log.info("DATA_PATH=%s", data_path)

    if args.input is not None:
        paths = [args.input]
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"--input-dir not a directory: {args.input_dir}")
        paths = discover_csvs(args.input_dir, args.glob)
        if not paths:
            raise SystemExit(f"No CSVs matching {args.glob!r} in {args.input_dir}")
        log.info("Found %d CSV(s) in %s", len(paths), args.input_dir)

    train_ids = load_train_ids(args.config_name, args.config_subdir)

    summary_rows: list[tuple[str, int, int]] = []
    for p in paths:
        out = args.output if args.output is not None else None
        n_hit, n_rows = annotate_csv(p, args.id_col, train_ids, output=out, dry_run=args.dry_run)
        summary_rows.append((str(p), n_hit, n_rows))

    # TSV summary to stdout for easy grepping / piping.
    print("path\tin_train\ttotal", file=sys.stdout)
    for path_str, n_hit, n_rows in summary_rows:
        print(f"{path_str}\t{n_hit}\t{n_rows}", file=sys.stdout)


if __name__ == "__main__":
    main()
