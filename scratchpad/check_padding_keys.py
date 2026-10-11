"""Read-only: what changes if PaddingTransform is dropped and the dense collate pads each batch to its longest chain.

For n train samples: the dataset's transforms run ONCE with a fixed seed except PaddingTransform; the result is then
(a) passed through PaddingTransform and collated alone, and (b) collated together with a longer sample (so the collate
pads it). For every key: is the valid region identical, and what fills the padded region in (a) vs (b)? Keys whose
fills differ are listed with their dtype, so each can be checked to be read only under a mask.
Usage (PYTHONPATH=repo): python scratchpad/check_padding_keys.py --config_name C [--n 50]
"""

import argparse
import random
from collections import defaultdict

import torch

from proteinfoundation.datasets.transforms import PaddingTransform
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingCollater
from scratchpad.test_plan_datamodule_integration import compose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    _, dm = compose(args.config_name, plan_on=False, num_workers=0)
    dm.setup("fit")
    ds = dm.train_ds
    tf = ds.transform
    steps = list(tf.transforms)
    pad_idx = [i for i, t in enumerate(steps) if isinstance(t, PaddingTransform)]
    assert len(pad_idx) == 1 and pad_idx[0] == len(steps) - 1, [type(t).__name__ for t in steps]
    pad = steps[-1]
    pre = steps[:-1]
    ds.transform = None
    collate = DensePaddingCollater(ds, None, None, cath_code_dir=dm.cath_code_dir, multilabel_mode=dm.multilabel_mode,
                                   cath_dedupe_codes=dm.cath_dedupe_codes).collate_fn
    rng = random.Random(0)
    idx = rng.sample(range(len(ds)), args.n)
    differ = defaultdict(lambda: [0, set(), None])
    valid_bad = defaultdict(int)
    checked, mask_ok = 0, 0
    for k, i in enumerate(idx):
        g = ds[i]
        torch.manual_seed(1000 + k)
        random.seed(1000 + k)
        for t in pre:
            g = t(g)
        L = int(g.coord_mask.shape[0])
        long_g = g.clone()
        a = collate([pad(g.clone())])
        # a longer partner forces collate padding of g up to pad.max_size
        partner = pad(long_g.clone())
        b = collate([g.clone(), partner])
        checked += 1
        # the residue mask the model uses: mask_dict["coords"][..., 0, 0] (proteina.py / model_trainer_base)
        ma, mb = a.mask_dict["coords"][0][..., 0, 0], b.mask_dict["coords"][0][..., 0, 0]
        mask_ok += bool(torch.equal(ma[:L], mb[:L]) and not ma[L:].any() and not mb[L:].any())
        for key in a.keys():
            va, vb = a[key], b[key]
            if not torch.is_tensor(va) or va.dim() < 2:
                continue
            va, vb = va[0], vb[0]
            if va.shape != vb.shape:
                differ[key][1].add(f"shape {tuple(va.shape)} vs {tuple(vb.shape)}")
                continue
            sl = tuple(slice(0, L) if s == pad.max_size else slice(None) for s in va.shape)
            if not torch.equal(va[sl], vb[sl]):
                valid_bad[key] += 1
            pad_a, pad_b = va.clone(), vb.clone()
            pad_a[sl], pad_b[sl] = 0, 0
            if not torch.equal(pad_a, pad_b):
                differ[key][0] += 1
                differ[key][2] = str(va.dtype)
                outside = torch.ones_like(va, dtype=torch.bool)
                outside[sl] = False
                differ[key][1].add(f"fills a={torch.unique(va[outside])[:3].tolist()} b={torch.unique(vb[outside])[:3].tolist()}")
    print(f"checked {checked} samples; model residue mask identical on the valid region and empty on the padding in "
          f"{mask_ok} of {checked}")
    print("keys whose VALID region differs (must be none):", dict(valid_bad) or "none")
    print("keys whose PADDED region differs (check each is read under a mask):")
    for key, (n, notes, dt) in sorted(differ.items()):
        print(f"  {key} dtype={dt} in {n} samples: {sorted(notes)[:3]}")


if __name__ == "__main__":
    main()
