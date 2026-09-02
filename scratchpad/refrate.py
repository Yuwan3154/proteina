"""How often does a sample get its OWN topology as the reference, on TRAIN as well as val?

The val loader measured 19.6% self-reference against a recorded 4.29%. If train shows the same
inflation, the conditioning signal both arms trained on is materially weaker than designed, which
is a training finding, not a probe finding.

Also separates the TWO routes to a self-reference, which the original `measure_template_pool.py`
did not:
  (a) no different-sequence cluster-mate  -> `_pick_template` returns `row`
  (b) the picked template has no usable DSSP runs -> `forward()` does `t_row = row`
Route (b) was never counted. This attributes each observed self-reference to one or the other by
re-running the picker's own logic against the index.
"""

import os
import sys
from collections import Counter

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = "training_contact_tri_full384_v1"
SPLIT = os.environ.get("SPLIT", "val")
N_BATCHES = int(os.environ.get("N_BATCHES", "4000"))


def main():
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=CONFIG)
    OmegaConf.set_struct(cfg_exp, False)
    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.setup("fit")
    dl = dm.train_dataloader() if SPLIT == "train" else dm.val_dataloader()
    print(f"[cfg] split={SPLIT} dataset={cfg_exp.dataset}", flush=True)

    # Reach the transform to replay the picker's logic on the observed self-references.
    def _find_topology_transform(obj, depth=0):
        """A Compose is not iterable -- its list lives on `.transforms`."""
        if obj is None or depth > 3:
            return None
        if type(obj).__name__ == "TopologyReferenceTransform":
            return obj
        inner = getattr(obj, "transforms", None)
        if isinstance(inner, (list, tuple)):
            for t in inner:
                found = _find_topology_transform(t, depth + 1)
                if found is not None:
                    return found
        return None

    # Read the index FILE; the live transform object is not reachable from dm or the dataset.
    data_dir = os.environ["DATA_PATH"] + "/pdb_train"
    idx = torch.load(os.path.join(data_dir, "topology_index.pt"), map_location="cpu",
                     weights_only=False, mmap=True)
    id_to_row = {str(v): i for i, v in enumerate(idx["ids"])}
    print(f"[index] {len(id_to_row)} chains", flush=True)

    def has_candidates(row):
        """Would _pick_template have had a different-sequence mate to choose from?"""
        cl = int(idx["cluster_of"][row])
        lo, hi = int(idx["members_offset"][cl]), int(idx["members_offset"][cl + 1])
        members = idx["members_flat"][lo:hi]
        if members.numel() <= 1:
            return False
        own = idx["seq_hash"][row]
        return int((idx["seq_hash"][members.long()] != own).sum()) > 0

    seen = present = empty = self_ref = 0
    routes = Counter()
    for i, batch in enumerate(dl):
        if i >= N_BATCHES:
            break
        refs = batch["topology_ref_id"]
        pids = batch["protein_id"]
        for r, p in zip(refs, pids):
            seen += 1
            if r == "":
                empty += 1
                continue
            present += 1
            if r != p:
                continue
            self_ref += 1
            row = id_to_row.get(str(p))
            if row is None:
                routes["query not in index"] += 1
            elif has_candidates(row):
                # The picker had real alternatives yet the emitted id equals the query: this
                # cannot come from _pick_template, which returns self only 1.7% of the time.
                routes["HAD candidates -- unexplained"] += 1
            else:
                routes["no different-seq mate (expected)"] += 1

    print(f"\nsplit={SPLIT}  batches={min(N_BATCHES, i + 1)}  samples={seen}")
    print(f"  reference present : {present}  ({present / max(seen, 1):.1%})")
    print(f"  empty (drop_prob) : {empty}  ({empty / max(seen, 1):.1%})")
    print(f"  SELF-reference    : {self_ref}  ({self_ref / max(present, 1):.1%} of present)")
    if routes:
        print("\n  route attribution for the self-references:")
        for k, v in routes.most_common():
            print(f"    {k:38s} {v:5d}  ({v / max(self_ref, 1):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
