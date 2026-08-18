# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""How many reverse-diffusion steps does a contact-map model actually need?

Validation sampling runs 200 steps (dt=0.005), inherited from coordinate diffusion. A contact map
is a binary L x L object, not a delicate 3D geometry, so the same schedule is very likely far
finer than it needs to be -- and at ~51 ms per step for a batch of 4 the difference is most of the
validation cost. This measures quality against step count on a FIXED chain set, so the numbers are
comparable across settings and across checkpoints.

Quality is top-L precision of the sampled contact map (the metric the best-checkpoint callback
monitors), not TM-score: this architecture emits no coordinates.

The step ladder is the experiment's independent variable and is passed in explicitly; nothing
here picks it.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="experiment config name the checkpoint came from")
    p.add_argument("--chains", required=True, help="one chain stem per line (the fixed val set)")
    p.add_argument("--processed-dir", required=True)
    p.add_argument("--shard-manifest", default=None)
    p.add_argument("--topology-index", required=True)
    p.add_argument("--steps", required=True,
                   help="comma-separated step counts to sweep, e.g. 200,100,50,25,10,5")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=512)
    return p.parse_args()


def main():
    import hydra
    from hydra import compose, initialize_config_dir

    from proteinfoundation.datasets.pdb_data import _processed_path_sharded
    from proteinfoundation.utils.dense_padding_data_loader import dense_padded_collate

    args = parse_args()
    ladder = [int(s) for s in args.steps.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_dir = os.path.join(os.environ["PROTEINA_REPO"], "configs", "experiment_config")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg_exp = compose(config_name=args.config)

    model = hydra.utils.instantiate(cfg_exp.model, cfg_exp=cfg_exp, _recursive_=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    print(f"loaded {args.checkpoint} (global_step={state.get('global_step')}) "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()

    manifest = json.load(open(args.shard_manifest)) if args.shard_manifest else None
    stems = [l.strip() for l in open(args.chains) if l.strip()]
    transform = TopologyReferenceTransform(
        index_path=args.topology_index,
        max_topology_len=int(cfg_exp.model.nn.get("max_topology_len", 128)),
        mutate_prob=0.0, sigma_frac=0.0, drop_prob=0.0,
    )

    graphs = []
    for stem in stems:
        path = str(_processed_path_sharded(args.processed_dir, stem, manifest))
        g = torch.load(path, map_location="cpu", weights_only=False)
        g.coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        g.coord_mask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        graphs.append(g)
    print(f"{len(graphs)} chains loaded", flush=True)

    rows = []
    for nsteps in ladder:
        dt = 1.0 / nsteps
        precs, t0 = [], time.time()
        for start in range(0, len(graphs), args.batch_size):
            chunk = graphs[start:start + args.batch_size]
            batch, _ = dense_padded_collate(type(chunk[0]), chunk)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.to_dict().items()}
            mask = batch["mask"].to(device)
            n = mask.shape[-1]
            refs = [
                transform.self_reference(s, int(mask[i].sum()))
                for i, s in enumerate(stems[start:start + len(chunk)])
            ]
            if any(r is None for r in refs):
                raise ValueError(
                    "a fixed-set chain is absent from the topology index; the sweep must "
                    "condition every chain identically or the step comparison is not like-for-like"
                )
            topology = {
                k: model._stack_topology([r[k] for r in refs]).to(device)
                for k in model.TOPOLOGY_KEYS
            }
            with torch.no_grad():
                result = model.generate(
                    nsamples=len(chunk), n=n, dt=dt,
                    self_cond=cfg_exp.training.self_cond,
                    residue_type=batch.get("residue_type_unmasked", batch.get("residue_type")),
                    dtype=torch.float32, mask=mask, topology=topology,
                )
            logits = result.get("contact_map_logits")
            gt = model.extract_clean_contact_map(batch, mask)
            for i in range(len(chunk)):
                m = model._compute_contact_map_metrics(logits[i], gt[i], mask[i])
                if m and "contact_precision_at_L" in m:
                    precs.append(m["contact_precision_at_L"])
        elapsed = time.time() - t0
        arr = np.array(precs)
        rows.append(dict(steps=nsteps, dt=dt, n=len(arr), mean=float(arr.mean()),
                         median=float(np.median(arr)), elapsed_s=elapsed))
        print(f"steps={nsteps:4d}  top-L precision mean={arr.mean():.4f} "
              f"median={np.median(arr):.4f}  ({elapsed:.1f}s for {len(arr)} chains)", flush=True)

    with open(args.out, "w") as fh:
        fh.write("steps\tdt\tn\tprec_mean\tprec_median\telapsed_s\n")
        for r in rows:
            fh.write(f"{r['steps']}\t{r['dt']:.6f}\t{r['n']}\t{r['mean']:.4f}\t"
                     f"{r['median']:.4f}\t{r['elapsed_s']:.1f}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
