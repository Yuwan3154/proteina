"""List every file the CA template model's data path opens (Engaging), so the A6000 copy is exactly that set.

Runs prepare_data + setup('fit'), pulls train and val batches in-process (num_workers 0, so opens are seen), loads every
chain of validation_sampling.fixed_chain_list, and builds the model. A sys.addaudithook records each opened path and
each listed directory.

  python scripts/sse1d_trace_data_files.py --n_batches 4 --out files.json
"""

import argparse
import json
import os
import sys
import tempfile

import hydra
import torch

from proteinfoundation.datasets.pdb_data import PDBDataset
from proteinfoundation.proteinflow.proteina import Proteina

OPENED, LISTED = set(), set()


def _hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.add(os.path.abspath(os.fsdecode(args[0])))
    elif event in ("os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        LISTED.add(os.path.abspath(os.fsdecode(args[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--sizes", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.addaudithook(_hook)
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg = hydra.compose(config_name="training_ca_template_compress_v1", overrides=a.sizes)
    with hydra.initialize(f"../configs/datasets_config/{cfg['dataset_config_subdir']}", version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg["dataset"])
    cfg_data.datamodule.num_workers = 0
    cfg_data.datamodule.prefetch_factor = None
    cfg_data.datamodule.batch_size = 1
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.prepare_data()
    dm.setup("fit")
    n_train = sum(1 for _, _ in zip(range(a.n_batches), dm.train_dataloader()))
    n_val = sum(1 for _, _ in zip(range(a.n_batches), dm.val_dataloader()))
    stems = [ln.strip() for ln in open(cfg.validation_sampling.fixed_chain_list) if ln.strip()]
    ds = PDBDataset(pdb_codes=[st.split("_", 1)[0] for st in stems], chains=[st.split("_", 1)[1] for st in stems],
                    data_dir=str(dm.data_dir), transform=dm.transform, format=dm.format, in_memory=False,
                    file_names=stems, num_workers=0)
    n_fixed = sum(1 for i in range(len(ds)) if ds[i] is not None)
    if a.sizes:
        Proteina(cfg, store_dir=tempfile.mkdtemp())
    rec = {"n_train": n_train, "n_val": n_val, "n_fixed": n_fixed, "n_fixed_listed": len(stems),
           "opened": sorted(p for p in OPENED if os.path.isfile(p)), "listed": sorted(LISTED)}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k.startswith("n_")}), flush=True)


if __name__ == "__main__":
    main()
