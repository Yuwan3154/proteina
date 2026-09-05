"""Is the diffusion TARGET clean, or does it contain unresolved atoms parked at the origin?

`_prepare` builds the atom mask from `rc.restype_atom14_mask[aatype]` -- which encodes which atoms
a residue TYPE possesses, not which atoms the experiment actually resolved. Crystal structures are
routinely missing side-chain atoms; those slots come through the atom37->atom14 gather as (0,0,0).

If the dataset's own per-atom mask disagrees with the residue-type mask, then every unresolved atom
is a training target sitting at the coordinate origin. The model would be explicitly taught to pull
those atoms to a single point -- which is exactly the observed end state (Rg 0.20x native, 5.6% of
CA pairs under 3.0 A).

This does not guess: it compares the dataset mask against the residue-type mask and counts how many
target atoms are exactly at the origin while being marked valid.
"""

import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import atom14_features
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer

DS = "pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828"


def main():
    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=DS)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    it = iter(dm.val_dataloader())
    tot_valid = tot_origin = tot_datamask_off = 0
    n_batches = 12
    for k in range(n_batches):
        raw = next(it)
        md = raw["mask_dict"]["coords"]
        if k == 0:
            print(f"mask_dict['coords'] shape = {tuple(md.shape)}  dtype={md.dtype}")
            print(f"coords shape              = {tuple(raw['coords'].shape)}")
            # Per-atom resolution present?  [B, L, 37, 3] would mean yes.
            print(f"  -> per-atom mask available: {md.dim() >= 3 and md.shape[2] > 1}")

        aatype = raw["residue_type"].long()
        res_mask = md[..., 0, 0].float()
        _, _, _, amask = atom14_features(aatype, res_mask)          # residue-TYPE mask
        pos14 = ContactToCoordTrainer._atom37_to_atom14(raw["coords"].float(), aatype)
        B, L, _, _ = pos14.shape
        amask = amask.reshape(B, L, 14)

        at_origin = (pos14.abs().sum(-1) == 0)                      # exactly (0,0,0)
        valid = amask > 0.5
        tot_valid += int(valid.sum())
        tot_origin += int((valid & at_origin).sum())

        # What the dataset itself says, gathered to atom14 the same way.
        if md.dim() >= 3 and md.shape[2] >= 37:
            dm14 = ContactToCoordTrainer._atom37_to_atom14(md[..., :3].float(), aatype)
            data_valid = dm14[..., 0] > 0.5
            tot_datamask_off += int((valid & ~data_valid).sum())

    print(f"\nover {n_batches} batches:")
    print(f"  atom slots marked valid by residue-type mask : {tot_valid}")
    print(f"  ...of those, target sits exactly at (0,0,0)  : {tot_origin}"
          f"  ({100.0*tot_origin/max(tot_valid,1):.2f}%)")
    print(f"  ...of those, dataset's own mask says INVALID  : {tot_datamask_off}"
          f"  ({100.0*tot_datamask_off/max(tot_valid,1):.2f}%)")
    print("\nAny non-trivial percentage here means the diffusion target contains phantom atoms at")
    print("the origin, and the model is being trained to place them there.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
