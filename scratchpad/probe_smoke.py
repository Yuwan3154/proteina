"""Does `topology_ref_id` survive the real dataloader, and what fraction of samples are usable?

The attribute is new and non-tensor, so the only thing that settles whether it survives
torch.save -> dense collate -> PaddingTransform is running the REAL dataloader. A string that
silently vanishes in the collate would make the probe's ground truth unbuildable while looking
fine in the transform itself.
"""

import os
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = "training_contact_tri_full384_v1"
N_BATCHES = 25


def main():
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=CONFIG)
    OmegaConf.set_struct(cfg_exp, False)
    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.setup("fit")
    dl = dm.val_dataloader()
    print(f"[cfg] dataset={cfg_exp.dataset}", flush=True)

    seen = present = empty = self_ref = 0
    examples = []
    for i, batch in enumerate(dl):
        if i >= N_BATCHES:
            break
        keys = batch.keys() if hasattr(batch, "keys") else []
        if "topology_ref_id" not in keys:
            print("FAIL: topology_ref_id ABSENT from the collated batch", flush=True)
            print(f"      keys = {sorted(keys)}", flush=True)
            return 2
        refs = batch["topology_ref_id"]
        pids = batch["protein_id"]
        if not isinstance(refs, (list, tuple)):
            print(f"FAIL: expected a list after collate, got {type(refs)}", flush=True)
            return 3
        if len(refs) != len(pids):
            print(f"FAIL: len mismatch refs={len(refs)} protein_id={len(pids)}", flush=True)
            return 4
        for r, p in zip(refs, pids):
            seen += 1
            if r == "":
                empty += 1
            else:
                present += 1
                if r == p:
                    self_ref += 1
                if len(examples) < 8:
                    examples.append((str(p), str(r)))

    print(f"\nbatches={N_BATCHES} samples={seen}")
    print(f"  reference present : {present}  ({present / max(seen, 1):.1%})")
    print(f"  empty (drop_prob) : {empty}  ({empty / max(seen, 1):.1%})   <- expect ~25%")
    print(f"  SELF-reference    : {self_ref}  ({self_ref / max(present, 1):.1%} of present)  <- expect ~6.7%")
    print("\n  query -> reference examples:")
    for p, r in examples:
        print(f"    {p:14s} -> {r}")

    if present == 0:
        print("\nFAIL: no sample carried a reference id", flush=True)
        return 5
    print("\nPASS: topology_ref_id survives the real dataloader", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
