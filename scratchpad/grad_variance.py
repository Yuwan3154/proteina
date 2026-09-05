"""Measure how much the diffusion mini-batch (n) actually reduces gradient variance.

The question: our optimizer step averages B_struct structures x n noise copies. Linear LR scaling
assumes the average is over i.i.d. samples, so Var ~ 1/(B*n). But the n copies share a structure,
a contact map, a trunk pass and a ground truth -- they differ only in sigma and the noise vector.
By the law of total variance,

    Var(g) = Var_struct( E_sigma[g|s] ) / B_struct  +  E_struct[ Var_sigma(g|s) ] / (B_struct * n)
             \_________ between: INVARIANT to n ________/   \______ within: shrinks with n ______/

so the honest denominator is  B_eff = B*n / (1 + (n-1)*rho).  rho=0 -> linear in pairs;
rho=1 -> only B_struct counts.

⛔ Estimator note: pairwise COSINE SIMILARITY is the intuitive framing and the wrong statistic --
gradients from different structures are also strongly correlated through the shared task (the
common mean mu), so within-structure cosine looks high even when the n-axis is doing real work.
This uses a one-way ANOVA decomposition of the gradient vectors instead, which separates mu,
between-structure and within-structure components properly.

Memory: never holds B*n full gradients. Only B per-structure sums plus scalars, which is what the
sums-of-squares identity needs.
"""

import argparse
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.proteinflow.contact2coord_trainer import (
    ALPHA_DIFFUSION,
    ALPHA_DISTOGRAM,
    ContactToCoordTrainer,
)

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=2, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def flat_grad(model):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
        for p in model.parameters()
    ])


def one_copy_grad(mod, b, include_distogram: bool):
    """Gradient of the loss from ONE noise copy of one structure."""
    mod.zero_grad(set_to_none=True)
    out = mod.model(b)
    dl, _ = diffusion_loss(out["x_denoised"], out["x_gt_rep"], out["sigma"], out["atom_mask_rep"])
    loss = ALPHA_DIFFUSION * dl.mean()
    if include_distogram:
        dg = mod._distogram_loss(out["pair_logits"], b["atom_pos"], b["aatype"], b["mask"])
        loss = loss + ALPHA_DISTOGRAM * dg.mean()
    loss.backward()
    return flat_grad(mod.model)


def decompose(mod, batches, n, include_distogram, dev):
    """One-way ANOVA over per-copy gradients. Returns (between, within, mu2, B, n)."""
    B = len(batches)
    ssq_total = 0.0                     # sum_ij ||g_ij||^2
    struct_sums = []                    # per structure: sum_j g_ij  (kept on CPU)
    for b in batches:
        acc = None
        for _ in range(n):
            g = one_copy_grad(mod, b, include_distogram)
            ssq_total += float(g.pow(2).sum())
            g = g.cpu()
            acc = g if acc is None else acc + g
        struct_sums.append(acc)
    grand = torch.stack(struct_sums).sum(0)

    ss_struct = sum(float(s.pow(2).sum()) / n for s in struct_sums)   # sum_i ||G_i||^2 / n
    ss_grand = float(grand.pow(2).sum()) / (B * n)

    # Classic sums-of-squares identity.
    ss_within = ssq_total - ss_struct                 # df = B*(n-1)
    ss_between = ss_struct - ss_grand                 # df = B-1
    within = ss_within / max(B * (n - 1), 1)          # = E_struct[Var_sigma], unbiased
    ms_between = ss_between / max(B - 1, 1)
    between = max((ms_between - within) / n, 0.0)     # unbiased Var_struct
    mu2 = ss_grand / (B * n)                          # ||mean gradient||^2 scale
    return between, within, mu2


def report(tag, between, within, B, n):
    tot = between + within
    rho = between / tot if tot > 0 else float("nan")
    var_cfg = between / B + within / (B * n)
    b_eff = tot / var_cfg if var_cfg > 0 else float("nan")
    print(f"\n--- {tag} ---")
    print(f"  between-structure variance : {between:.6g}")
    print(f"  within-structure  variance : {within:.6g}")
    print(f"  rho (between / total)      : {rho:.4f}")
    print(f"  Var at B={B}, n={n}          : {var_cfg:.6g}")
    print(f"  B_eff (vs one pair)        : {b_eff:.1f}   [pairs={B*n}, structures={B}]")
    print(f"  n-axis credit              : {b_eff/B:.2f}x  (1.00 = n buys nothing, "
          f"{n:.2f} = n is as good as independent examples)")
    return b_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--structures", type=int, default=4)
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--dataset", default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    dev = "cuda"

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    mod = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=1))
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}\n  missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    mod = mod.to(dev).eval()
    # ⛔ n=1 per forward: each call draws its OWN sigma and noise, which is exactly one independent
    # copy. Batching them would share nothing extra but would hide the per-copy gradient.
    mod.model.n_diffusion_samples = 1

    it = iter(dm.val_dataloader())
    batches = []
    for _ in range(args.structures):
        raw = next(it)
        batches.append(mod._prepare(
            {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
            if isinstance(raw, dict) else raw, train=False))
    print(f"{len(batches)} structures x {args.copies} noise copies "
          f"= {len(batches)*args.copies} backward passes", flush=True)

    B, n = len(batches), args.copies
    for tag, incl in [("FULL loss (diffusion + distogram, as trained)", True),
                      ("DIFFUSION term only", False)]:
        torch.manual_seed(0)
        bet, wit, _ = decompose(mod, batches, n, incl, dev)
        report(tag, bet, wit, B, n)

    print("\n⚠️  Measured at ONE checkpoint. rho drifts over training -- early on the gradient is")
    print("    dominated by a global signal shared across copies (high rho); it falls as the model")
    print("    starts fitting structure-specific detail. This bounds the scaling rule, it does not")
    print("    certify any particular learning rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
