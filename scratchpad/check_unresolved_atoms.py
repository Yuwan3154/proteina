"""How many diffusion targets are PHANTOM atoms, and how many get random chirality supervision?

`_prepare` builds the atom mask from the residue-TYPE table only (atom14_features(aatype, mask)) and
never reads `coord_mask`, which the dataset computes (pdb_data.py:1837) and carries in
batch["mask_dict"]. So an atom that is absent from the deposited structure -- a disordered surface
side chain, say -- is supervised as a real target sitting at the fill value 1e-5, i.e. the origin.

⛔ This also invalidates my earlier "training data is 100% right-handed" check: that filtered on
`norm(CB) > 1e-3`, which silently EXCLUDES exactly these atoms. Re-measured here without the filter.

The chirality consequence is specific: a residue whose CB is unresolved has its ground-truth
tetrahedron (N-CA)x(C-CA).(CB-CA) computed against a CB at the origin, tens of Angstrom away (the
structures are not centred). Its signed volume then has an essentially arbitrary sign -- random
handedness supervision, injected by the data pipeline, at exactly those residues.
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
# atom37 -> the four atoms of the chirality tetrahedron
A37 = {"N": 0, "CA": 1, "C": 2, "CB": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
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
    mod = ContactToCoordTrainer(model_cfg=MODEL_CFG)

    tot_sup = tot_phantom = 0
    tot_res = tot_cb_missing = 0
    signs_all, signs_resolved = [], []
    seen = 0
    it = iter(dm.train_dataloader())
    while seen < args.n:
        try:
            raw = next(it)
        except StopIteration:
            break
        # ⛔ NOT raw["mask_dict"]["coords"] -- that is the PADDING mask from _dense_padded_collate.
        # Crystallographic resolvedness is `graph.coords != fill_value_coords` (pdb_data.py:1837),
        # i.e. an atom is unresolved exactly when its coordinates sit at the 1e-5 fill value.
        # Measuring against the padding mask (my first attempt) proves nothing about resolvedness.
        FILL = 1e-5
        cm = (raw["coords"].float().abs() > (FILL * 10)).any(-1)   # [B, L, 37] resolved
        b = mod._prepare(raw, train=False)
        L = b["mask"].shape[1]
        amask = b["atom_mask"].reshape(-1, L, 14).bool()        # what the loss actually supervises
        coords37 = raw["coords"].float()
        for j in range(amask.shape[0]):
            if seen >= args.n:
                break
            res_ok = b["mask"][j].bool()
            if int(res_ok.sum()) < 40:
                continue
            seen += 1
            # atom14 slot k <-> atom37 index: only the tetrahedron matters for chirality
            sup = amask[j][res_ok]
            tot_sup += int(sup.sum())
            resolved37 = cm[j][res_ok]
            # N,CA,C,O,CB occupy atom14 slots 0,1,2,3,4 and atom37 slots 0,1,2,4,3
            slot37 = [0, 1, 2, 4, 3]
            for k, a37 in enumerate(slot37):
                supervised = sup[:, k]
                unresolved = ~resolved37[:, a37]
                tot_phantom += int((supervised & unresolved).sum())
            x = coords37[j][res_ok]
            N, CA, C, CB = (x[:, A37["N"]], x[:, A37["CA"]], x[:, A37["C"]], x[:, A37["CB"]])
            has_slot = sup[:, 4]                                # CB supervised (non-Gly)
            cb_res = resolved37[:, A37["CB"]]
            tot_res += int(has_slot.sum())
            tot_cb_missing += int((has_slot & ~cb_res).sum())
            v = torch.einsum("ni,ni->n", torch.cross(N - CA, C - CA, dim=-1), CB - CA).numpy()
            sel = has_slot.numpy()
            signs_all.extend(v[sel].tolist())
            signs_resolved.extend(v[sel & cb_res.numpy()].tolist())

    sa, sr = np.array(signs_all), np.array(signs_resolved)
    print(f"\nchains: {seen}")
    print(f"\nsupervised atom targets            : {tot_sup}")
    print(f"  of which UNRESOLVED (phantom)    : {tot_phantom}  "
          f"({100.0*tot_phantom/max(tot_sup,1):.2f}%)")
    print(f"\nresidues with a supervised CB      : {tot_res}")
    print(f"  of which CB is UNRESOLVED        : {tot_cb_missing}  "
          f"({100.0*tot_cb_missing/max(tot_res,1):.2f}%)")
    print(f"\nCA signed volume, ALL supervised CB residues (what the loss actually sees):")
    print(f"  fraction positive (L)            : {(sa > 0).mean():.4f}   n={len(sa)}")
    print(f"CA signed volume, RESOLVED CB only (what my earlier filtered check measured):")
    print(f"  fraction positive (L)            : {(sr > 0).mean():.4f}   n={len(sr)}")
    bad = int((sa <= 0).sum())
    print(f"\n⭐ residues given NON-L (wrong-sign) chirality supervision: {bad}/{len(sa)} "
          f"({100.0*bad/max(len(sa),1):.2f}%)")
    print("   Any nonzero count is random handedness supervision injected by the data pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
