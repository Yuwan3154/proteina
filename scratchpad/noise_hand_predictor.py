"""Can the OUTPUT hand be predicted from the INPUT NOISE, before paying for a rollout?

Motivation, and why this is not a fishing expedition. The sampler was measured exactly
reflection-equivariant: rollout(Me) = M rollout(e), so hand(rollout(Me)) = 1 - hand(rollout(e)).
That means the output hand IS a deterministic function of the seed. The open question is whether it
is a CHEAP function. If some pseudoscalar of the raw noise predicts it, we can reject bad seeds for
free and turn rejection resampling from 2.10 rollouts per accepted sample into ~1.0 -- a 2x sampling
speedup on top of the fix that already works.

⚠️ PRIOR EXPECTATION, RECORDED BEFORE RUNNING: probably WEAK. Equivariance guarantees the map
e -> hand exists, but guarantees nothing about it being low-order; 50 steps of a nonlinear denoiser
can scramble it arbitrarily. A null result is still worth having: it says the hand is decided by
something the model computes, not by a surface statistic of the noise, and it closes off an obvious
optimisation so nobody re-tries it.

Two candidate predictors, both computed on the raw starting noise interpreted as a CA trace:
  1. helix_pos_frac of the noise -- the SAME statistic used on outputs, so no new definition.
  2. mean signed volume over consecutive CA quadruples -- a different pseudoscalar, in case (1) is
     too tied to the helical-range window.
Both are reflection-odd, so under e -> Me each flips, matching the known output behaviour. That is
the minimum a candidate predictor must satisfy.
"""

import argparse
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.nn.af3_diffusion import noise_schedule
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import _ca_dihedrals

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def mean_signed_volume(ca):
    """Mean of (b-a) x (c-b) . (d-c) over consecutive CA quadruples. Reflection-odd."""
    if len(ca) < 4:
        return float("nan")
    a, b, c, d = ca[:-3], ca[1:-2], ca[2:-1], ca[3:]
    return float(np.mean(np.sum(np.cross(b - a, c - b) * (d - c), axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
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
    assert "ema" in ck, "refusing to score the unaveraged model"
    model.model.load_state_dict(ck["ema"]["params"], strict=True)
    print(f"[load] EMA @ step {ck.get('global_step')}", flush=True)
    model = model.to(dev).eval()

    sig0 = noise_schedule(torch.linspace(0.0, 1.0, args.steps + 1, device=dev))[0]
    rows = []
    it = iter(dm.val_dataloader())
    for ci in range(args.chains):
        raw = next(it)
        b = model._prepare(raw, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        A = b["atom_mask"].shape[1]
        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
            for si in range(args.seeds):
                g = torch.Generator(device=dev).manual_seed(90_000 + 100 * ci + si)
                e = sig0 * torch.randn(1, A, 3, device=dev, generator=g)
                # The noise, read as a CA trace, in the same atom14 slot layout as the output.
                ca_noise = e.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
                p1 = helix_pos_frac(ca_noise)
                p2 = mean_signed_volume(ca_noise)
                out = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                          b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                          n_steps=args.steps, x_init=e)
                ca = out.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
                h = helix_pos_frac(ca)
                if not (np.isnan(p1) or np.isnan(p2) or np.isnan(h)):
                    rows.append((p1, p2, h))
        print(f"  chain {ci}: {len(rows)} samples so far", flush=True)

    a = np.asarray(rows)
    y = (a[:, 2] > 0.5).astype(float)
    print(f"\n=== n={len(a)} samples, mirrored {100*y.mean():.1f}% ===")
    for j, nm in ((0, "helix_pos_frac(noise)"), (1, "mean_signed_volume(noise)")):
        x = a[:, j]
        r = float(np.corrcoef(x, y)[0, 1])
        # Accuracy of the best single split on this predictor -- an upper bound on its usefulness.
        order = np.argsort(x)
        ys = y[order]
        best = max(max((ys[:k] == 0).sum() + (ys[k:] == 1).sum(),
                       (ys[:k] == 1).sum() + (ys[k:] == 0).sum()) for k in range(len(ys) + 1))
        print(f"  {nm:28s} r={r:+.4f}   best-split accuracy {100*best/len(ys):5.1f}%  "
              f"(chance {100*max(y.mean(), 1-y.mean()):.1f}%)")
    print("\n  A predictor is only useful if best-split accuracy clearly beats chance. If it does")
    print("  not, the hand is decided by something the network computes, not by a surface")
    print("  statistic of the noise -- and seed pre-screening cannot work.")


if __name__ == "__main__":
    main()
