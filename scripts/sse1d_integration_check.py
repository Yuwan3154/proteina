"""Real-data CPU integration check for CATemplateCompress1D: the tri's datamodule + Proteina's real training_step.

For each decompressor and both segmentation phases (true DSSP / own SSE head): pull real batches through the tri's
dataset transforms, run Proteina.training_step on CPU with TINY test-only sizes, backprop, and record every logged loss
term. Fails loudly if a loss is non-finite, a template/DSSP/distogram term is missing, or a gradient is non-finite.

  python scripts/sse1d_integration_check.py --n_batches 2
"""

import argparse
import json
import math
import os
import tempfile

import hydra
import torch
from omegaconf import OmegaConf

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
    ap.add_argument("--out", default="sse1d_integration.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    results = []
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        base = hydra.compose(config_name="training_ca_template_compress_v1",
                             overrides=TEST_SIZES + ["model.nn.decompress=basis_pool", "model.nn.true_seg_until_step=1"])
    with hydra.initialize(f"../configs/datasets_config/{base['dataset_config_subdir']}", version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=base["dataset"])  # same path train.py uses
    cfg_data.datamodule.num_workers = 0
    cfg_data.datamodule.batch_size = 1
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")
    batches = []
    for i, b in enumerate(datamodule.train_dataloader()):
        batches.append(b)
        if len(batches) == a.n_batches:
            break
    print(json.dumps({"event": "data", "n_batches": len(batches), "keys": sorted(batches[0].keys()),
                      "L": [int(b["mask"].shape[1]) for b in batches]}), flush=True)
    for dec in ("pair_row_xattn", "pair_bias", "basis_pool", "spectral"):
        for until, phase in ((1, "true_seg"), (0, "pred_seg")):
            with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
                cfg = hydra.compose(config_name="training_ca_template_compress_v1",
                                    overrides=TEST_SIZES + [f"model.nn.decompress={dec}", f"model.nn.true_seg_until_step={until}"])
            model = Proteina(cfg, store_dir=tempfile.mkdtemp())
            logged = {}
            model.log = lambda name, value, *args, **kw: logged.__setitem__(name, float(value))
            for i, b in enumerate(batches):
                b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
                loss = model.training_step(b, i)
                loss = loss["loss"] if isinstance(loss, dict) else loss
                model.zero_grad()
                loss.backward()
                bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                n_grad = sum(1 for p in model.parameters() if p.grad is not None)
                rec = {"decompress": dec, "phase": phase, "batch": i, "loss": float(loss), "n_params_with_grad": n_grad,
                       "nonfinite_grads": bad, "logged": {k: v for k, v in logged.items() if k.startswith("train")}}
                results.append(rec)
                print(json.dumps(rec), flush=True)
                assert math.isfinite(float(loss)), rec
                assert not bad, rec
                for key in REQUIRED:
                    assert any(key in k for k in logged), (key, sorted(logged))
    json.dump(results, open(a.out, "w"), indent=1)
    print(json.dumps({"event": "PASS", "n_records": len(results)}), flush=True)


if __name__ == "__main__":
    main()
