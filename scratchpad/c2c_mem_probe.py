"""Worst-case GPU memory for the c2c model: a full forward+backward at the DATASET'S MAX LENGTH,
for a grid of (n_diff, diff_chunk). The smoke run only walks a handful of real batches, so the
longest chain it happens to draw is not the cap -- this pins the number that decides whether AF3's
48 diffusion samples fit on the target card.

usage: python c2c_mem_probe.py --lengths 384 --n_diff 48 --chunks 0 8 16 24 [--batch 1]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord

# the production model config (scratchpad/train_c2c.py MODEL_CFG)
CFG = dict(c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
           n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
           atom_blocks=3, atom_heads=4)


def make_batch(B, L, device, dtype):
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(B, L)
    aatype = torch.randint(0, 20, (B, L), generator=g)
    c = (torch.rand(B, L, L, generator=g) < 0.076).float()      # CB-8 density
    c = torch.triu(c, 1)
    c = c + c.transpose(1, 2) + torch.eye(L)[None]              # ContactEBM keeps the diagonal
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aatype, mask)
    b = {"contacts": c, "aatype": aatype, "mask": mask, "ref_feats": ref_feats, "ref_pos": ref_pos,
         "atom_to_token": a2t, "atom_mask": amask, "ref_space_uid": ruid,
         "atom_pos": torch.randn(B, L * 14, 3, generator=g) * 12.0 * amask[..., None]}
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[384])
    ap.add_argument("--n_diff", type=int, default=48)
    ap.add_argument("--chunks", type=int, nargs="+", default=[0, 8, 16, 24])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    args = ap.parse_args()
    dev = "cuda"
    print(f"[gpu] {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    print(f"{'L':>5} {'n_diff':>7} {'chunk':>6} {'peak_alloc_GiB':>15} {'peak_resv_GiB':>14} {'status':>10}")
    for L in args.lengths:
        for chunk in args.chunks:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = ContactToCoord(**CFG, n_diffusion_samples=args.n_diff, diff_chunk=chunk).to(dev)
            batch = make_batch(args.batch, L, dev, torch.float32)
            status = "ok"
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision.startswith("bf16")):
                    out = model(batch)
                dl, _ = diffusion_loss(out["x_denoised"].float(), out["x_gt_rep"].float(), out["sigma"],
                                       out["atom_mask_rep"], use_smooth_lddt=False)
                dl.mean().backward()
            except torch.cuda.OutOfMemoryError:
                status = "OOM"
            peak_a = torch.cuda.max_memory_allocated() / 2 ** 30
            peak_r = torch.cuda.max_memory_reserved() / 2 ** 30
            print(f"{L:5d} {args.n_diff:7d} {chunk:6d} {peak_a:15.1f} {peak_r:14.1f} {status:>10}", flush=True)
            del model, batch
            torch.cuda.empty_cache()
    print("MEMPROBE_DONE")


if __name__ == "__main__":
    main()
