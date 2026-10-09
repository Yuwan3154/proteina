"""Real-data CPU integration check for CATemplateCompress1D: the tri's datamodule + Proteina's real training_step.

For each decompressor: pull real batches through the dataset transforms, run Proteina.training_step on CPU with TINY
test-only sizes, backprop, and record every logged loss term. Fails loudly if a loss is non-finite, a template/DSSP/
distogram term is missing, or a gradient is non-finite. Also compares the pack's stored DSSP labels with the
proline-donor-mask recomputation on raw (untransformed) training graphs.

  python scripts/sse1d_integration_check.py --n_batches 2
"""

import argparse
import json
import math
import os
import tempfile
from importlib.metadata import version

import hydra
import lightning as L
import torch

from proteinfoundation.datasets.transforms import DSSPTargetTransform
from proteinfoundation.proteinflow.proteina import Proteina

TEST_SIZES = [  # integration only: NOT run settings (those are still ??? for the user)
    "model.nn.token_dim=64", "model.nn.pair_repr_dim=32", "model.nn.dim_cond=64", "model.nn.nheads=4",
    "model.nn.n_pre=1", "model.nn.n_mid=1", "model.nn.n_post=1", "model.nn.mid_dim=32",
    "model.nn.spectral_heads=2", "model.nn.spectral_r=4", "model.nn.spectral_hidden=16",
]
REQUIRED = ("dssp_aux_loss", "distogram_loss", "align", "mlm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--n_dssp", type=int, default=200)
    ap.add_argument("--out", default="sse1d_integration.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    results = []
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        base = hydra.compose(config_name="training_ca_template_compress_v1",
                             overrides=TEST_SIZES + ["model.nn.decompress=basis_pool"])
    with hydra.initialize(f"../configs/datasets_config/{base['dataset_config_subdir']}", version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=base["dataset"])  # same path train.py uses
    cfg_data.datamodule.num_workers = 2  # the dataloader sets prefetch_factor, which needs workers
    cfg_data.datamodule.batch_size = 1
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    batches = []
    for i, b in enumerate(datamodule.train_dataloader()):
        batches.append(b)
        if len(batches) == a.n_batches:
            break
    keys = sorted(batches[0].keys())  # a PyG Batch; the trainer builds `mask` inside training_step
    shapes = {k: list(batches[0][k].shape) for k in keys if torch.is_tensor(batches[0][k])}
    print(json.dumps({"event": "data", "n_batches": len(batches), "keys": keys, "shapes": shapes}), flush=True)
    ds = datamodule.train_ds
    tf, ds.transform = ds.transform, None
    base, pro = DSSPTargetTransform(), DSSPTargetTransform(proline_donor_mask=True)
    n_res = n_diff = n_diff_near_pro = n_pro = n_stored = 0
    for i in range(a.n_dssp):
        g = ds[i]
        n_stored += int(getattr(g, "dssp_target", None) is not None)
        old = base(g.clone()).dssp_target  # stored label if present, else pydssp without the mask
        new = pro(g.clone()).dssp_target
        ok = (old >= 0) & (new >= 0)
        d = (old != new) & ok
        near = torch.zeros_like(d)
        for sh in range(-4, 5):  # a changed label within 4 residues of a proline
            near |= torch.roll(g.residue_type == DSSPTargetTransform.PRO_IDX, sh)
        n_res += int(ok.sum()); n_diff += int(d.sum()); n_diff_near_pro += int((d & near).sum())
        n_pro += int((g.residue_type == DSSPTargetTransform.PRO_IDX).sum())
    ds.transform = tf
    dssp_rec = {"event": "dssp_proline", "pydssp": version("pydssp"), "n_chains": a.n_dssp, "n_stored": n_stored,
                "n_res": n_res, "n_pro": n_pro,
                "changed": n_diff, "changed_frac": n_diff / max(n_res, 1), "changed_within4_of_pro": n_diff_near_pro}
    print(json.dumps(dssp_rec), flush=True)
    assert n_diff > 0, dssp_rec  # the mask must change something; the size of the change is reported, not gated
    for dec in ("pair_row_xattn", "pair_bias", "basis_pool", "spectral"):
        with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
            cfg = hydra.compose(config_name="training_ca_template_compress_v1",
                                overrides=TEST_SIZES + [f"model.nn.decompress={dec}"])
        model = Proteina(cfg, store_dir=tempfile.mkdtemp())
        # training_step reads trainer.world_size / global_step: attach a bare CPU trainer (no fit, no logger)
        model.trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                                  enable_progress_bar=False)
        logged = {}
        model.log = lambda name, value, *args, **kw: logged.__setitem__(name, float(value))
        for i, b in enumerate(batches):
            b = b.clone()
            loss = model.training_step(b, i)
            loss = loss["loss"] if isinstance(loss, dict) else loss
            model.zero_grad()
            loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            n_grad = sum(1 for p in model.parameters() if p.grad is not None)
            rec = {"decompress": dec, "batch": i, "loss": float(loss), "n_params_with_grad": n_grad,
                   "nonfinite_grads": bad, "logged": {k: v for k, v in logged.items() if k.startswith("train")}}
            results.append(rec)
            print(json.dumps(rec), flush=True)
            assert math.isfinite(float(loss)), rec
            assert not bad, rec
            for key in REQUIRED:
                assert any(key in k for k in logged), (key, sorted(logged))
    json.dump({"dssp": dssp_rec, "steps": results}, open(a.out, "w"), indent=1)
    print(json.dumps({"event": "PASS", "n_records": len(results)}), flush=True)


if __name__ == "__main__":
    main()
