"""Measured speed and VRAM for the structure head, against the predicted table.

The plan predicted ~10.5 GB/sample at L=384 head-only and an L40S holding batch 3. That was
arithmetic from parameter counts and an activation coefficient calibrated off one measurement. This
measures it. Reports three regimes so the claim "freezing buys memory, not time" is checked rather
than asserted:

  trained  -- trunk trains, structure head trains       (the expensive baseline)
  detached -- trunk trains on its own losses, head detached from it
  frozen   -- trunk requires_grad_(False), only the head trains

⛔ Peak VRAM is read with torch.cuda.max_memory_allocated after a reset, and every timing is taken
after warm-up with an explicit synchronize -- CUDA is asynchronous and an unsynchronised timer
measures queue submission, not compute.
"""

import argparse
import json
import sys
import time

import torch

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

BASE = dict(
    pair_dim=320, tri_hidden=320, n_blocks=12, transition_n=4, dim_cond=128, max_rel_pos=64,
    topology_cond=True, max_topology_he_len=64, topology_vocab_size=65, n_residue_types=22,
    pair_ref_features="both", contact_map_mode=True, contact_map_input_dim=1, non_contact_value=0,
)
SH = dict(enabled=True, mode="diffusion", c_s=384, c_z=128,
          diffusion=dict(c_token=768, n_blocks=24, n_heads=16))


def make_batch(B, L, T, device, dtype):
    return {
        "contact_map_t": torch.rand(B, L, L, device=device, dtype=dtype),
        "contact_map_sc": torch.rand(B, L, L, device=device, dtype=dtype),
        "residue_type": torch.randint(0, 21, (B, L), device=device),
        "mask": torch.ones(B, L, device=device, dtype=dtype),
        "t": torch.rand(B, device=device, dtype=dtype),
        "topology_he_tokens": torch.randint(1, 65, (B, T), device=device),
        "topology_he_pos_raw": torch.arange(T, device=device).float()[None].repeat(B, 1),
        "topology_he_feat": torch.rand(B, T, T, N_PAIR_FEATURES, device=device, dtype=dtype),
        "x_1_ca": torch.randn(B, L, 3, device=device, dtype=dtype) * 5.0,
    }


def build(regime, device):
    sh = dict(SH)
    if regime == "frozen":
        sh["freeze_trunk"] = True
    kw = dict(BASE)
    if regime == "trained":
        m = ContactMapTriSiT(**kw)          # no head at all: the pure contact baseline
    else:
        m = ContactMapTriSiT(**kw, structure_head=sh)
    return m.to(device).train()


def run(regime, L, B, T, device, steps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build(regime, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch = make_batch(B, L, T, device, torch.float32)

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(dict(batch))
            if "x_denoised" in out:
                loss, _ = diffusion_loss(out["x_denoised"].float(), batch["x_1_ca"],
                                         out["sigma"], batch["mask"])
                loss = loss.mean()
            else:
                loss = out["contact_map_logits"].float().pow(2).mean()
        loss.backward()
        opt.step()

    one()                                    # warm-up: allocator + autotune out of the timing
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del model, opt, batch
    return {"regime": regime, "L": L, "B": B, "T": T, "s_per_step": round(dt, 4),
            "peak_gb": round(peak, 2), "trainable_M": round(n_train / 1e6, 2),
            "total_M": round(n_total / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 384])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--topology", type=int, default=64)
    args = ap.parse_args()

    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"total: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    rows = []
    for L in args.lengths:
        for regime in ("frozen", "detached", "trained"):
            try:
                rec = run(regime, L, args.batch, args.topology, dev)
            except torch.cuda.OutOfMemoryError:
                rec = {"regime": regime, "L": L, "B": args.batch, "T": args.topology,
                       "s_per_step": None, "peak_gb": "OOM", "trainable_M": None, "total_M": None}
                torch.cuda.empty_cache()
            rows.append(rec)
            print(f"  L={rec['L']:4d} B={rec['B']} {rec['regime']:9s} "
                  f"peak={str(rec['peak_gb']):>7s} GB  step={str(rec['s_per_step']):>7s} s  "
                  f"trainable={rec['trainable_M']} M / {rec['total_M']} M")
    print()
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
