"""Write CA-only PDBs for the probed validation chains and a training sample, for foldseek.

The 12 chains the generation probe actually sampled are the first N of the val dataloader in order,
so they are reproduced here by walking the same loader rather than by hard-coding ids -- a
hard-coded list would silently drift if the loader order or split changed.
"""

import argparse
import os
import sys

import hydra
import torch
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
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    qdir, tdir = os.path.join(args.out, "query"), os.path.join(args.out, "train")
    os.makedirs(qdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    def dump(loader, out, n, tag):
        it, written = iter(loader), 0
        while written < n:
            try:
                b = next(it)
            except StopIteration:
                break
            ids = b.get("protein_id") or b.get("id") or []
            m = b["mask_dict"]["coords"][..., 0, 0].bool()
            coords = b["coords"].float()
            for j in range(coords.shape[0]):
                if written >= n:
                    break
                stem = str(ids[j]) if j < len(ids) else f"{tag}{written:06d}"
                ca = coords[j][m[j]][:, 1, :]
                if ca.shape[0] < 30:      # foldseek is unreliable on very short chains
                    continue
                write_ca(os.path.join(out, f"{stem}.pdb"), ca.numpy())
                written += 1
        print(f"[{tag}] wrote {written} CA-only PDBs to {out}", flush=True)
        return written

    # ⭐ The SAME first-N-in-loader-order the generation probe used, so the leakage question is asked
    # about the exact chains whose TM scores we reported.
    dump(dm.val_dataloader(), qdir, args.n_val, "val")
    dump(dm.train_dataloader(), tdir, args.n_train, "train")
    print("\n⚠️ The training sample is the first n_train chains in loader order, not the full split.")
    print("   A NEGATIVE result (no fold match) is therefore weaker than a positive one: a neighbour")
    print("   could exist in the part of training that was not searched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
