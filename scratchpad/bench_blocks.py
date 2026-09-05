"""Budget report: AF3 diffusion transformer at 24 vs 8 vs 6 blocks.

Isolates the diffusion head itself (not the trunk), because that is the only thing the block count
changes. Reports parameters, peak VRAM and s/step for a full train step (forward + diffusion loss +
backward + optimizer), which is the number that decides what fits and how long an epoch takes.

Run under bf16 autocast, which is what training actually uses.
"""

import argparse
import json
import sys
import time

import torch

from proteinfoundation.nn.af3_diffusion import AF3DiffusionHead, diffusion_loss, sample_noise_level

# AF3 widths throughout; only depth varies. Every replica surveyed keeps these fixed.
C_S, C_Z, C_TOKEN, N_HEADS = 384, 128, 768, 16
VARIANTS = [(24, "AF3 / Protenix base / Boltz / OF3 / IntelliFold v1+v2"),
            (8, "Protenix mini + tiny"),
            (6, "IntelliFold v2-flash (their default)")]


def run(n_blocks, L, B, device, steps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    head = AF3DiffusionHead(c_s=C_S, c_z=C_Z, c_token=C_TOKEN,
                            n_blocks=n_blocks, n_heads=N_HEADS).to(device).train()
    n_par = sum(p.numel() for p in head.parameters())
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)

    s = torch.randn(B, L, C_S, device=device)
    z = torch.randn(B, L, L, C_Z, device=device)
    mask = torch.ones(B, L, device=device)
    x_gt = torch.randn(B, L, 3, device=device) * 5.0

    def one():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sigma = sample_noise_level((B,), device)
            x_noisy = x_gt + torch.randn_like(x_gt) * sigma[:, None, None]
            x_den = head.denoise(x_noisy, sigma, s, z, mask)
            loss, _ = diffusion_loss(x_den.float(), x_gt, sigma, mask)
        loss.mean().backward()
        opt.step()

    one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    del head, opt
    return {"blocks": n_blocks, "L": L, "B": B, "params_M": round(n_par / 1e6, 2),
            "peak_gb": round(peak, 2), "s_per_step": round(dt, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 384])
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB   "
          f"bf16 autocast, AF3 widths c_s={C_S} c_z={C_Z} c_token={C_TOKEN} heads={N_HEADS}\n")
    rows = []
    for nb, who in VARIANTS:
        print(f"--- {nb} blocks   [{who}]")
        for L in args.lengths:
            try:
                rec = run(nb, L, args.batch, dev)
            except torch.cuda.OutOfMemoryError:
                rec = {"blocks": nb, "L": L, "B": args.batch, "params_M": None,
                       "peak_gb": "OOM", "s_per_step": None}
                torch.cuda.empty_cache()
            rows.append(rec)
            print(f"    L={rec['L']:4d}  params={str(rec['params_M']):>7s} M   "
                  f"peak={str(rec['peak_gb']):>6s} GB   step={str(rec['s_per_step']):>7s} s")
    print()
    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
