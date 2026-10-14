"""drop_ref_len_range for T8 by the rule that set the old [5, 39] (D22, job 22502519): the p5-p95 of the element count T
(the model's topology axis, topology_he_tokens) over 1,500 references SERVED by the real TopologyReferenceTransform.
The transform is built from the T8 dataset yaml's own kwargs, except drop_prob 0 (only served, never dropped, references
count) and drop_ref_len_range, which is the value being measured (unused when nothing is dropped). Queries: a seeded
sample of TRAIN chains that have a template in range (the index's eligible list), real lengths from the native pack.
Usage (PYTHONPATH=repo): python scratchpad/t8_measure_drop_ref_len.py --index IDX --eligible TXT --train TXT --pack PACK
"""

import argparse
import io
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

YAML = "configs/datasets_config/pdb/pdb_train_contact-confind-synthtopo_S25_max512_t8.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--n", type=int, default=1500, help="the D22 sample size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(YAML)
    kw = next(t for t in OmegaConf.to_container(cfg.datamodule.transforms, resolve=False)
              if t["_target_"].endswith("TopologyReferenceTransform"))
    kw = {k: v for k, v in kw.items() if not k.startswith("_")}
    kw.update(index_path=args.index, drop_prob=0.0, drop_ref_len_range=[1, 1], seed=args.seed)
    print("transform kwargs:", kw, flush=True)
    tr = TopologyReferenceTransform(**kw)
    elig = set(l.strip() for l in open(args.eligible) if l.strip())
    train = [l.strip() for l in open(args.train) if l.strip() in elig]
    sample = random.Random(args.seed).sample(sorted(train), args.n)
    z = np.load(args.pack + ".idx.npz", allow_pickle=True)
    off = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    T, Ls, missing = [], [], 0
    with open(args.pack, "rb") as fh:
        for stem in sample:
            o, ln = off[stem]
            fh.seek(o)
            g0 = torch.load(io.BytesIO(fh.read(ln)), map_location="cpu", weights_only=False)
            g = tr(Data(coords=g0.coords, protein_id=stem))
            if int(g.topology_missing_ref.sum()):
                missing += 1
                continue
            T.append(int(g.topology_he_tokens.numel()))
            Ls.append(int(g0.coords.shape[0]))
    T = np.array(T)
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"served {len(T)} of {len(sample)} (missing reference {missing}); query lengths {min(Ls)}-{max(Ls)}")
    # the old D22 rule does not record its percentile convention: both shown
    print("element count T percentiles (linear) " + " ".join(f"p{q}={np.percentile(T, q):.1f}" for q in qs))
    print("element count T percentiles (nearest) " + " ".join(f"p{q}={int(np.percentile(T, q, method='nearest'))}" for q in qs)
          + f"; mean {T.mean():.1f} sd {T.std():.1f}; at the cap {kw['max_topology_he_len']}: {int((T >= kw['max_topology_he_len']).sum())}")


if __name__ == "__main__":
    main()
