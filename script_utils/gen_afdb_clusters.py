#!/usr/bin/env python3
"""
Generate AFDB (d_FS) sequence-similarity cluster TSV for CAT-balanced sampling.

Instantiates the d_FS datamodule with the splitter switched to
sequence_similarity at the requested min-seq-id, then calls prepare_data() +
setup("fit") so PDBDataSplitter runs mmseqs and writes
cluster_seqid_<seqid>_d_FS.tsv next to the data. Reuses all existing dataset
logic. Does NOT reprocess structures (regenerate_missing=False, overwrite=False)
and does NOT set min/max_length (so _filter_processed_by_length never unlinks).

Requires DATA_PATH to be exported (d_FS.yaml interpolates ${oc.env:DATA_PATH}).

Usage:
  python script_utils/gen_afdb_clusters.py --seqid 0.25                    # full run
  python script_utils/gen_afdb_clusters.py --seqid 0.25 --max-samples 500  # smoke test
"""

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proteinfoundation.utils.cluster_utils import read_cluster_tsv


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/datasets_config/afdb/d_FS.yaml")
    parser.add_argument("--seqid", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dm_cfg = cfg.datamodule
    OmegaConf.set_struct(dm_cfg, False)

    # Read-only over structures: skip reprocessing, never unlink, no RAM load.
    dm_cfg.overwrite = False
    dm_cfg.regenerate_missing = False
    dm_cfg.in_memory = False
    dm_cfg.sampling_mode = "random"
    if args.max_samples is not None:
        dm_cfg.max_samples = args.max_samples

    # Switch the splitter to mmseqs sequence-similarity clustering at --seqid.
    dm_cfg.datasplitter.split_type = "sequence_similarity"
    dm_cfg.datasplitter.split_sequence_similarity = args.seqid
    dm_cfg.datasplitter.overwrite_sequence_clusters = True

    print(
        f"[gen_afdb_clusters] data_dir={dm_cfg.data_dir} seqid={args.seqid} "
        f"max_samples={args.max_samples}",
        flush=True,
    )
    dm = hydra.utils.instantiate(dm_cfg)
    dm.prepare_data()
    dm.setup("fit")

    data_dir = Path(dm.data_dir)
    tsvs = sorted(data_dir.glob("cluster_seqid_*_d_FS.tsv"))
    assert len(tsvs) >= 1, f"no cluster TSV produced under {data_dir}"
    for t in tsvs:
        m = read_cluster_tsv(t)
        n_clusters = len(m)
        n_chains = sum(len(v) for v in m.values())
        print(
            f"[gen_afdb_clusters] {t.name}: {n_clusters} clusters, {n_chains} chains",
            flush=True,
        )


if __name__ == "__main__":
    main()
