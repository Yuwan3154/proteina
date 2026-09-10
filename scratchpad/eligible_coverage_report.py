"""How much of the TRAIN split survives the eligible-ids restriction, per split, with the exact same
filter the datamodule applies. This is the number the "launch now or wait for the backfill" decision
turns on: clusters kept, members kept, and which clusters would be dropped.

usage: python eligible_coverage_report.py --dataset <yaml stem> --eligible <ids.txt> [--repo .]
"""

import argparse
import os
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-subdir", default="pdb")
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--repo", default=os.environ.get("PROTEINA_REPO", "."))
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(a.repo, "configs/datasets_config", a.dataset_subdir, a.dataset + ".yaml"))
    OmegaConf.resolve(cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup()
    with open(a.eligible) as fh:
        eligible = {line.strip() for line in fh if line.strip()}
    print(f"eligible ids: {len(eligible)}")
    print(f"\n{'split':>6} {'clusters':>9} {'kept':>7} {'kept%':>7} {'members':>9} {'kept':>9} {'kept%':>7}")
    for split in ("train", "val", "test"):
        mapping = dm.datasplitter.clusterid_to_seqid_mappings.get(split)
        if not mapping:
            continue
        n_cl = len(mapping)
        n_mem = sum(len(v) for v in mapping.values())
        kept = {cl: [s for s in seqs if s in eligible] for cl, seqs in mapping.items()}
        kept = {cl: v for cl, v in kept.items() if v}
        k_cl = len(kept)
        k_mem = sum(len(v) for v in kept.values())
        print(f"{split:>6} {n_cl:9d} {k_cl:7d} {100*k_cl/max(n_cl,1):6.1f}% {n_mem:9d} {k_mem:9d} {100*k_mem/max(n_mem,1):6.1f}%")
        if split == "train":
            dropped = [cl for cl in mapping if cl not in kept]
            sizes = sorted(len(mapping[cl]) for cl in dropped)
            if sizes:
                print(f"       dropped clusters: {len(dropped)}; their sizes min {sizes[0]} "
                      f"median {sizes[len(sizes)//2]} max {sizes[-1]}; singletons among them "
                      f"{sum(1 for x in sizes if x == 1)}")
                print(f"       chains in dropped clusters: {sum(sizes)}")
            # one epoch = one draw per cluster, so this is the per-epoch sample count
            print(f"       ⇒ per-epoch samples: {n_cl} unrestricted -> {k_cl} restricted")
    print("COVERAGE_REPORT_DONE")


if __name__ == "__main__":
    main()
