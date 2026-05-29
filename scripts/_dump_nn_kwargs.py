"""One-shot helper: compose the H1d Hydra config, dump cfg.model.nn (with the
runtime-injected discrete dims) to JSON.

Mirrors proteinfoundation/proteinflow/proteina.py:140 which injects
contact_map_input_dim / dssp_input_dim / dssp_num_classes into cfg.model.nn
based on the discrete_diffusion + dssp_diffusion configs.

Usage: python scripts/_dump_nn_kwargs.py
Writes: scripts/_debug_nn_kwargs.json
"""
from __future__ import annotations

import json
import os

import hydra
from omegaconf import OmegaConf

CONFIG_DIR = "/home/ubuntu/proteina/configs/experiment_config"
CONFIG_NAME = "training_dssp_contact_20M_udlm_pb_v2_stage1"
OVERRIDES = [
    "+model.nn.use_torch_compile=false",
    "training.self_cond_use_copy=False",
]
OUT = "/home/ubuntu/proteina/scripts/_debug_nn_kwargs.json"

# Mirror _compute_discrete_input_dims(): for "udlm" type there is no mask
# token, so the input_dim is exactly vocab_size; for absorbing types it is
# vocab_size + 1.
def discrete_input_dim(dd_cfg):
    vs = int(dd_cfg["vocab_size"])
    return vs + 1 if dd_cfg.get("type", "").lower() in ("md4", "genmd4") else vs


def main():
    os.environ.setdefault("DATA_PATH", "/home/ubuntu/proteina/data")
    with hydra.initialize_config_dir(CONFIG_DIR, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=OVERRIDES)

    nn_cfg = OmegaConf.to_container(cfg.model.nn, resolve=True)

    # Inject runtime-computed dims (cf. proteina.py:140)
    dd = cfg.model.get("discrete_diffusion")
    if dd is not None and dd.get("enabled"):
        kind = dd.get("type", "udlm")
        nn_cfg["contact_map_input_dim"] = discrete_input_dim(dd.get(kind, dd))
    dsspd = cfg.model.get("dssp_diffusion")
    if dsspd is not None and dsspd.get("enabled"):
        nn_cfg["dssp_input_dim"] = discrete_input_dim(dsspd)
        nn_cfg["dssp_num_classes"] = int(dsspd["vocab_size"])

    with open(OUT, "w") as f:
        json.dump(nn_cfg, f, indent=2, sort_keys=True)
    print(f"Dumped {len(nn_cfg)} top-level keys to {OUT}")
    for k in ("contact_map_input_dim", "dssp_input_dim", "dssp_num_classes"):
        print(f"  injected {k} = {nn_cfg.get(k)}")


if __name__ == "__main__":
    main()
