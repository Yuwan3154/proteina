"""WHY is 44% of generation mirrored, and WHEN does handedness get decided?

The generation probe established WHAT the failure is: correct local stereochemistry (chirality_agree
0.998) assembled into a globally reflected fold (mirror_rmsd 1.2-2.3 A). This script asks the two
follow-up questions that decide whether raising the diffusion multiplicity can help at all.

  Q1  Is handedness a coin flip driven by the initial NOISE, or a property of the CHAIN?
      Same chain, K different seeds. Two very different worlds:
        per-chain rate ~50% for most chains  -> spontaneous symmetry breaking. The conditioning
            genuinely does not determine handedness, and the fix must be a training signal that
            does (fix B strengthens exactly that signal; fix C adds a new one).
        chains cleanly split into always-right / always-wrong -> NOT symmetry breaking. Something
            about those chains (fold class, length, contact density) predicts it, and more noise
            draws per step would not help.

  Q2  WHERE in the reverse trajectory does handedness commit?
      Score the partially-denoised structure at every step. If handedness is settled while sigma is
      still large, any fix acting only at low noise is too late -- and it tells us which end of the
      noise schedule the loss weighting needs to emphasise.

⛔ Reports per-chain counts, never a pooled mean. A pooled 44% is consistent with BOTH worlds above,
which is exactly why the earlier number could not settle the question.
"""

import argparse
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import noise_schedule
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import chirality_agreement

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def kabsch_rmsd(a, b, allow_reflection=False):
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ rot) - b) ** 2).sum(-1).mean()))


def handedness(ca_gen, ca_gt):
    """(proper_rmsd, improper_rmsd, is_mirror). Mirror = the reflection fits distinctly better."""
    p = kabsch_rmsd(ca_gen, ca_gt, allow_reflection=False)
    m = kabsch_rmsd(ca_gen, ca_gt, allow_reflection=True)
    return p, m, (p > 2.0 * m and p - m > 1.0)


@torch.no_grad()
def rollout_traced(model, b, n_steps, seed, ca_gt, keep, L):
    """One reverse-diffusion trajectory, scoring handedness at every step.

    A copy of Contact2Coord.rollout with the per-step readout added. Kept literal rather than
    refactoring the trainer: a monitoring hook in the real sampler would be a behaviour change on
    the path the training run uses.
    """
    dev = next(model.parameters()).device
    s, z, _ = model.encode(b["contacts"], b["aatype"], b["mask"])
    mask, amask = b["mask"], b["atom_mask"]
    B, A = amask.shape
    m = amask[..., None]
    nreal = amask.sum(dim=1, keepdim=True).clamp_min(1.0)[..., None]
    sig = noise_schedule(torch.linspace(0.0, 1.0, n_steps + 1, device=dev))

    g = torch.Generator(device=dev).manual_seed(seed)
    x = sig[0] * torch.randn(B, A, 3, device=dev, generator=g) * m
    trace = []
    for i in range(n_steps):
        s_prev, s_cur = sig[i], sig[i + 1]
        x = (x - (x * m).sum(dim=1, keepdim=True) / nreal) * m
        gamma = 0.8 if s_cur > 1.0 else 0.0
        t_hat = s_prev * (gamma + 1.0)
        x_noisy = x + 1.003 * torch.sqrt((t_hat ** 2 - s_prev ** 2).clamp_min(0)) \
            * torch.randn(x.shape, device=dev, generator=g) * m
        d = model.denoise(x_noisy, t_hat.expand(B), s, z, mask, b["ref_feats"], b["ref_pos"],
                          b["atom_to_token"], amask, b["ref_space_uid"]) * m
        x = (x_noisy + 1.5 * (s_cur - t_hat) * (x_noisy - d) / t_hat) * m
        # Score the DENOISER'S current estimate of the clean structure, not the noisy iterate:
        # at high sigma the iterate is mostly noise and its handedness is meaningless.
        ca = d.reshape(B, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
        p, mm, is_mir = handedness(ca, ca_gt)
        trace.append((float(s_prev), p, mm, is_mir))
    return x, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_chains", type=int, default=6)
    ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--trace_chains", type=int, default=3)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.model.load_state_dict(ck["ema"]["params"], strict=False)
    # ⛔ A PRE-fix-A checkpoint loads here with fix A's three modules left RANDOM, which is not the
    # old model and not the new one. Zeroing them would not recover the old model either, because
    # the offset term is now gated by ref_space_uid regardless. So refuse outright rather than
    # report a number for a model that never existed.
    assert not missing, (
        f"checkpoint predates the reference-offset block ({sorted(set(k.rsplit('.', 1)[0] for k in missing))}). "
        "Run this against a fix-A checkpoint; a pre-A one cannot give a valid baseline here.")
    assert not unexpected, f"checkpoint has params the model lacks: {list(unexpected)[:8]}"
    print(f"[load] EMA @ step {ck.get('global_step')}", flush=True)
    model = model.to(dev).eval()

    batches = []
    it = iter(dm.val_dataloader())
    for _ in range(args.n_chains):
        b = model._prepare(next(it), train=False)
        batches.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})

    print(f"\n=== Q1: same chain, {args.n_seeds} seeds -- is handedness a coin flip? ===")
    print(f"{'chain':>7} {'L':>5} {'mirrored':>10} {'rate':>7}   per-seed (M=mirror, .=correct)")
    rates = []
    for ci, b in enumerate(batches):
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        ca_gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
        flags = []
        for sd in range(args.n_seeds):
            x, _ = rollout_traced(model.model, b, args.steps, 1000 + sd, ca_gt, keep, L)
            ca = x.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            flags.append(handedness(ca, ca_gt)[2])
        r = sum(flags) / len(flags)
        rates.append(r)
        print(f"{'ch%02d' % ci:>7} {int(keep.sum()):>5} {sum(flags):>4}/{len(flags):<5} {r:>6.2f}   "
              + "".join("M" if f else "." for f in flags), flush=True)

    mixed = sum(1 for r in rates if 0.0 < r < 1.0)
    print(f"\nchains with a MIXED outcome across seeds: {mixed}/{len(rates)}")
    print("  many mixed  -> spontaneous symmetry breaking; the training signal is what must change.")
    print("  none mixed  -> handedness is a property of the CHAIN, and more noise draws won't help.")

    print(f"\n=== Q2: where in the trajectory does handedness commit? ===")
    for ci, b in enumerate(batches[:args.trace_chains]):
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        ca_gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
        _, trace = rollout_traced(model.model, b, args.steps, 1000, ca_gt, keep, L)
        final = trace[-1][3]
        # First step from which the verdict never changes again.
        commit = len(trace) - 1
        while commit > 0 and trace[commit - 1][3] == final:
            commit -= 1
        print(f"\n  ch{ci:02d} (L={int(keep.sum())}) final={'MIRROR' if final else 'correct'}  "
              f"commits at step {commit}/{len(trace)} (sigma={trace[commit][0]:.2f})")
        print(f"    {'step':>5} {'sigma':>9} {'proper':>8} {'improper':>9}  verdict")
        for i in list(range(0, len(trace), max(1, len(trace) // 10))) + [len(trace) - 1]:
            sg, p, mm, isf = trace[i]
            print(f"    {i:>5} {sg:>9.3f} {p:>8.2f} {mm:>9.2f}  {'MIRROR' if isf else 'correct'}")
    print("\n⚠️ 'commits' = the first step whose verdict survives to the end. If that sits at high")
    print("   sigma, a fix acting only at low noise arrives too late to change the outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
