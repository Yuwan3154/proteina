"""Speed/memory of the contact-to-coordinate model at the training shape.

Reports fwd+bwd seconds per sample and peak VRAM at L=384 (the dataset's max) for a range of batch
sizes, under the same bf16 autocast the trainer uses. The batch-size sweep is the point: the
dataset config pins batch_size=1, but that number was MEASURED for the tri model, not this one.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def make_batch(B, L, dev):
    aatype = torch.randint(0, 20, (B, L), device=dev)
    mask = torch.ones(B, L, device=dev)
    contacts = (torch.rand(B, L, L, device=dev) < 0.05).float()
    contacts = ((contacts + contacts.transpose(1, 2)) > 0).float()
    ref_feats, ref_pos, a2t, amask = atom14_features(aatype, mask)
    atom_pos = torch.randn(B, L * 14, 3, device=dev) * 16.0 * amask[..., None]
    return {"contacts": contacts, "aatype": aatype, "mask": mask, "ref_feats": ref_feats,
            "ref_pos": ref_pos, "atom_to_token": a2t, "atom_mask": amask, "atom_pos": atom_pos}


def run(model, B, L, dev, iters=6):
    batch = make_batch(B, L, dev)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        loss = diffusion_loss(out["x_denoised"], batch["atom_pos"], out["sigma"],
                              batch["atom_mask"])[0].mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
        if i == 1:                      # drop warmup, then restart the peak counter
            torch.cuda.reset_peak_memory_stats()
    steady = sorted(ts[2:])[len(ts[2:]) // 2]
    return steady, torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[384])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--ndiff", type=int, nargs="+", default=[1, 4, 8, 16, 32, 48])
    args = ap.parse_args()

    dev = "cuda"
    print(torch.cuda.get_device_name(0), flush=True)
    print(f"{'L':>5} {'B':>3} {'ndiff':>6} {'s/iter':>9} {'peak GiB':>9}")
    for L in args.lengths:
        for B in args.batches:
            for nd in args.ndiff:
                torch.cuda.empty_cache()
                cfg = dict(MODEL_CFG, n_diffusion_samples=nd)
                model = ContactToCoord(**cfg, n_ref_feats=N_REF_FEATS).to(dev)
                try:
                    t, mem = run(model, B, L, dev)
                    print(f"{L:>5} {B:>3} {nd:>6} {t:>9.3f} {mem:>9.2f}", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{L:>5} {B:>3} {nd:>6} {'OOM':>9} {'-':>9}", flush=True)
                del model


if __name__ == "__main__":
    main()
