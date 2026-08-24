"""Compute the noisy-input FLOOR directly, with NO model, and compare batch_size 1 vs 4.

The floor is precision@L of ranking pairs by the NOISED contact map against ground truth.
It depends only on (data, t, mask) -- never on the model. The two arms report floors that
differ 2.3x at thigh (0.862 local_attn vs 0.383 tri) and tri's violates monotonicity in t,
which is impossible for a sound measurement. This reproduces the quantity from first
principles on the SAME chains at the SAME t, under BOTH batch sizes, to find which is wrong.

Everything is taken from the training path: extract_clean_contact_map -> fm.sample_reference
-> fm.interpolate -> _compute_contact_map_metrics, so any discrepancy is real, not a
re-implementation artifact.
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier")

from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase

T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_CHAINS = 24
SEED = 0


def build_batches(batch_size):
    """Same dataset/transforms the arms use; only batch_size varies."""
    from proteinfoundation.datasets.pdb_data import PDBLightningDataModule
    cfg = OmegaConf.load(
        "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/configs/datasets_config/pdb/"
        "pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828.yaml"
    )
    OmegaConf.resolve(cfg)
    dm_cfg = cfg["datamodule"]
    dm_cfg["batch_size"] = batch_size
    dm_cfg["num_workers"] = 0
    from hydra.utils import instantiate
    dm = instantiate(dm_cfg)
    dm.setup("fit")
    dl = dm.val_dataloader()
    out, seen = [], 0
    for b in dl:
        d = b if isinstance(b, dict) else b.to_dict()
        d["mask"] = d["mask_dict"]["coords"][..., 0, 0]
        out.append(d)
        seen += int(d["mask"].shape[0])
        if seen >= N_CHAINS:
            break
    return out


def floors_for(batches, fm):
    """Per-chain floor at each t, keyed by t."""
    res = {t: [] for t in T_GRID}
    for d in batches:
        mask = d["mask"].bool()
        c_1 = ModelTrainerBase.extract_clean_contact_map(None, d, mask) \
            if not hasattr(ModelTrainerBase, "_unused") else None
        for tv in T_GRID:
            b = mask.shape[0]
            n = mask.shape[-1]
            torch.manual_seed(SEED)  # identical noise draw across batch sizes
            c_0 = fm.sample_reference(n=n, shape=(b,), device=mask.device,
                                      dtype=torch.float32, mask=mask, modality="contact_map")
            t = torch.full((b,), float(tv))
            c_t = fm.interpolate(c_0, c_1, t, mask=mask, modality="contact_map")
            for i in range(b):
                m = ModelTrainerBase._compute_contact_map_metrics(
                    c_t[i].float(), c_1[i].float(), mask[i])
                if m and "contact_precision_at_L" in m:
                    res[tv].append(m["contact_precision_at_L"])
    return res


print("This computes the FLOOR from first principles -- no model, no checkpoint.")
print(f"t grid: {T_GRID}\n")
