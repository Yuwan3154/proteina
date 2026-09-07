"""Write CA-only PDBs for the ENTIRE validation split, sharded.

12/12 on the probed chains is a striking number but a tiny sample -- it is consistent with "the
split is systematically contaminated" and with "those twelve were unlucky". Running every
validation chain against the training database settles which.

⛔ Sharded at 1000/dir: the val split is ~4158 chains, well past the 1024-per-directory limit.
"""

import argparse
import os
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CA_LINE = "ATOM  {:>5d}  CA  ALA A{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}  1.00  0.00           C\n"


def write_ca(path, ca):
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(ca):
            fh.write(CA_LINE.format(i + 1, i + 1, float(x), float(y), float(z)))
        fh.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_val", type=int, default=100000)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    qdir = os.path.join(args.out, "query")
    os.makedirs(qdir, exist_ok=True)

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    it = iter(dm.val_dataloader())
    written, short = 0, 0
    while written < args.n_val:
        try:
            b = next(it)
        except StopIteration:
            break
        ids = b.get("protein_id") or b.get("id") or []
        m = b["mask_dict"]["coords"][..., 0, 0].bool()
        coords = b["coords"].float()
        for j in range(coords.shape[0]):
            if written >= args.n_val:
                break
            stem = str(ids[j]) if j < len(ids) else f"val{written:06d}"
            ca = coords[j][m[j]][:, 1, :]
            if ca.shape[0] < 30:
                short += 1
                continue
            sub = os.path.join(qdir, f"s{written // 1000:03d}")
            os.makedirs(sub, exist_ok=True)
            write_ca(os.path.join(sub, f"{stem}.pdb"), ca.numpy())
            written += 1
    print(f"[val] wrote {written} CA-only PDBs to {qdir}/s*/  (skipped {short} shorter than 30 res)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
