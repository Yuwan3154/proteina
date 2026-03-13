#!/usr/bin/env python3
"""
Process training data from a dataset config file.

Instantiates the datamodule specified in the config, then calls prepare_data()
which handles downloading raw CIF/PDB files and converting them to .pt graph
files using CPU multiprocessing.

Usage
-----
    # PDB dataset
    DATA_PATH=/data/pdb_train python proteinfoundation/scripts/process_training_data.py \\
        --config configs/datasets_config/pdb/pdb_train_S25_max256_purge-test_cutoff-190828.yaml \\
        --num_workers 32

    # AFDB dataset
    DATA_PATH=/data python proteinfoundation/scripts/process_training_data.py \\
        --config configs/datasets_config/afdb/d_21M.yaml \\
        --num_workers 64

Notes
-----
- DATA_PATH must be set (or exported) so that ${oc.env:DATA_PATH} in the YAML resolves.
- --num_workers controls both download parallelism and structure-processing parallelism.
- Use --overwrite to force reprocessing of already-existing .pt files.
- Use --regenerate_missing to only reprocess files that are absent (default behaviour
  in most configs but can be made explicit here).
"""

import argparse
import os
import sys
import time
from pathlib import Path


def _resolve_oc_env(cfg_str: str) -> str:
    """Expand ${oc.env:VAR} and ${oc.env:VAR,default} placeholders in a YAML string."""
    import re

    def _replacer(m):
        content = m.group(1)
        if "," in content:
            var, default = content.split(",", 1)
        else:
            var, default = content, None
        val = os.environ.get(var.strip())
        if val is None:
            if default is not None:
                return default.strip()
            raise EnvironmentError(
                f"Environment variable '{var.strip()}' is required by the dataset config "
                f"but is not set. Export it before running this script, e.g.:\n"
                f"  export {var.strip()}=/path/to/data"
            )
        return val

    return re.sub(r"\$\{oc\.env:([^}]+)\}", _replacer, cfg_str)


def _load_config(config_path: str):
    """Load a dataset YAML config, resolving ${oc.env:...} variables."""
    from omegaconf import OmegaConf

    raw = Path(config_path).read_text()
    resolved = _resolve_oc_env(raw)
    cfg = OmegaConf.create(resolved)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process training data from a dataset config YAML"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to dataset config YAML (e.g. configs/datasets_config/pdb/pdb_train_S25_max256_purge-test_cutoff-190828.yaml)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of CPU workers for multiprocessing. "
             "Defaults to min(32, os.cpu_count()). Overrides the value in the config.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Force reprocessing of existing .pt files (sets datamodule.overwrite=True).",
    )
    parser.add_argument(
        "--regenerate_missing", action=argparse.BooleanOptionalAction, default=None,
        help="Regenerate only missing .pt files. "
             "If not specified, uses the value from the config (usually True).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    num_workers = args.num_workers if args.num_workers is not None else min(32, os.cpu_count() or 1)
    print(f"Using {num_workers} CPU workers for multiprocessing")

    # ── Load config ──────────────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    cfg = _load_config(args.config)

    if "datamodule" not in cfg:
        print("ERROR: config does not contain a 'datamodule' key.", file=sys.stderr)
        print(f"  Top-level keys: {list(cfg.keys())}", file=sys.stderr)
        sys.exit(1)

    dm_cfg = cfg.datamodule

    # ── Apply CLI overrides ──────────────────────────────────────────────────
    from omegaconf import OmegaConf

    overrides = {"num_workers": num_workers, "use_multiprocessing": True}
    if args.overwrite:
        overrides["overwrite"] = True
    if args.regenerate_missing is not None:
        overrides["regenerate_missing"] = args.regenerate_missing

    dm_cfg = OmegaConf.merge(dm_cfg, OmegaConf.create(overrides))

    # ── Instantiate datamodule ───────────────────────────────────────────────
    print(f"Instantiating datamodule: {dm_cfg.get('_target_', '(unknown)')}")
    import hydra
    datamodule = hydra.utils.instantiate(dm_cfg)

    # Ensure multiprocessing is on (in case not in the datamodule __init__ signature)
    if hasattr(datamodule, "use_multiprocessing"):
        datamodule.use_multiprocessing = True
    if hasattr(datamodule, "num_workers"):
        datamodule.num_workers = num_workers

    # ── Run prepare_data ─────────────────────────────────────────────────────
    print("Running prepare_data() ...")
    t0 = time.time()
    datamodule.prepare_data()
    elapsed = time.time() - t0
    print(f"prepare_data() completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
