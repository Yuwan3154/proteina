"""Is the TRAINING DATA uniformly right-handed?

The user's objection is decisive: a denoiser trained only on right-handed structures should emit
right-handed structures when uncertain. A 50/50 output therefore points at the DATA before it points
at the loss. If some fraction of training targets are stored mirrored -- a permutation, an axis
flip, a bad atom-order remap -- the model would be learning exactly the coin flip we observe.

Measures the coordinates that ACTUALLY reach the loss, i.e. batch["coords"] after the full
dataset -> transform -> collate -> _prepare path, not a re-read of the source files.

Three independent sign tests per chain:
  1. CA stereocentre  : sign of (N-CA) x (C-CA) . (CB-CA). Positive = L. Per residue.
  2. helix handedness : sign of the CA(i..i+3) pseudo-dihedral in helical range.
  3. atom14 target    : the same test on the atom14 tensor the diffusion loss consumes.

⛔ Tests the atom14 tensor SEPARATELY from atom37, because _atom37_to_atom14 is a gather over an
index table and a wrong table would silently permute atoms -- which is exactly the kind of bug that
would flip chirality without touching a single coordinate value.
"""

import argparse
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

MODEL_CFG = dict(
    c_s=64, c_z=32, c_token=64, c_atom=32, c_atompair=8, n_blocks=1, n_heads=2,
    n_tri_blocks=1, tri_hidden=16, transition_n=1, atom_blocks=1, atom_heads=2,
)


def ca_dihedrals(ca):
    b0, b1, b2 = ca[1:-2] - ca[:-3], ca[2:-1] - ca[1:-2], ca[3:] - ca[2:-1]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    return np.degrees(np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="train", choices=["train", "val"])
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
    loader = dm.train_dataloader() if args.split == "train" else dm.val_dataloader()

    mod = ContactToCoordTrainer(model_cfg=MODEL_CFG)

    rows = []
    it = iter(loader)
    while len(rows) < args.n:
        try:
            raw = next(it)
        except StopIteration:
            break
        ids = raw.get("protein_id") or raw.get("id") or []
        b = mod._prepare(raw, train=False)
        L = b["mask"].shape[1]
        a14 = b["atom_pos"].reshape(-1, L, 14, 3)
        m = b["mask"].bool()
        for j in range(a14.shape[0]):
            if len(rows) >= args.n:
                break
            keep = m[j]
            if int(keep.sum()) < 40:
                continue
            x = a14[j][keep].float().numpy()          # [L, 14, 3] -- N,CA,C,O,CB,...
            N, CA, C, CB = x[:, 0], x[:, 1], x[:, 2], x[:, 4]
            has_cb = np.linalg.norm(CB, axis=-1) > 1e-3
            v = np.einsum("ij,ij->i", np.cross(N - CA, C - CA), CB - CA)[has_cb]
            d = ca_dihedrals(CA)
            sel = d[(np.abs(d) > 30) & (np.abs(d) < 90)]
            rows.append((
                str(ids[j]) if j < len(ids) else f"chain{len(rows)}",
                int(keep.sum()),
                float((v > 0).mean()) if len(v) else float("nan"),
                float(np.median(v)) if len(v) else float("nan"),
                float((sel > 0).mean()) if len(sel) >= 5 else float("nan"),
                len(sel),
            ))

    print(f"\nchains examined: {len(rows)}  (split={args.split})")
    print(f"{'chain':>10} {'L':>5} {'CA stereo +ve':>14} {'median vol':>11} {'helix +ve':>10} {'n':>5}")
    for r in rows[:25]:
        print(f"{r[0]:>10} {r[1]:>5} {r[2]:>14.3f} {r[3]:>11.2f} {r[4]:>10.3f} {r[5]:>5}")
    if len(rows) > 25:
        print(f"  ... {len(rows)-25} more")

    st = np.array([r[2] for r in rows if not np.isnan(r[2])])
    hx = np.array([r[4] for r in rows if not np.isnan(r[4])])
    print(f"\nCA stereocentre, fraction L per chain : mean {st.mean():.4f}  min {st.min():.4f}")
    print(f"helix dihedral, fraction positive     : mean {hx.mean():.4f}  min {hx.min():.4f}  "
          f"max {hx.max():.4f}")
    n_bad = int((st < 0.5).sum())
    print(f"\nchains whose residues are predominantly D (stereo < 0.5): {n_bad}/{len(st)}")
    print(f"chains whose helices are predominantly the MINORITY sense: "
          f"{int((hx > 0.5).sum())}/{len(hx)}")
    print("\n⭐ If both are ~0 the data is clean and the model has no excuse -- look elsewhere.")
    print("⭐ If either is ~half, the training targets themselves are mixed-hand and THAT is the bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
