"""Print the first N validation chain IDs in loader order.

The generation probe labels chains gen00..gen15 by position; the leakage search keys on PDB IDs.
Without this map the two cannot be joined, and the question that matters -- do the MIRRORED chains
have closer or more distant training homologs than the correct ones? -- is unanswerable.

Both consumers walk the same val_dataloader with the same config, and both logs show the same first
cluster sample, so position is a valid key.
"""

import argparse
import os
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    it = iter(dm.val_dataloader())
    print(f"{'idx':>5} {'label':>7} {'chain_id':>12} {'L':>5}")
    seen = 0
    while seen < args.n:
        b = next(it)
        ids = b.get("protein_id") or b.get("id") or []
        m = b["mask_dict"]["coords"][..., 0, 0].bool()
        for j in range(m.shape[0]):
            if seen >= args.n:
                break
            cid = str(ids[j]) if j < len(ids) else "?"
            print(f"{seen:>5} {'gen%02d' % seen:>7} {cid:>12} {int(m[j].sum()):>5}")
            seen += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
