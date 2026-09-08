"""Does starting the sampler inside the TRAINED sigma regime change the mirror rate?

The sampler starts at sigma = 16*S_MAX = 2560, a noise level a training draw reaches once in 69,657
samples (training draws sigma = 16*exp(-1.2 + 1.5*N(0,1)), median 4.82). Only 0.77% of training
draws even reach SNR < 0.1. So the model is asked to denoise from a regime it has essentially never
seen, and the regime where it WOULD have to invent a hand is the one it never trains in.

⛔ THE SWEPT VALUES ARE MEASURED, NOT INVENTED. Each S_MAX corresponds to a quantile of the model's
OWN training sigma distribution (p50/p90/p99/p99.9/p99.99), plus the shipped default. This is a
sweep over measured quantities rather than a guessed point value.

⚠️ PRIOR EXPECTATION, RECORDED BEFORE RUNNING: this will probably NOT fix the mirror rate, because
the sampler was measured to be reflection-equivariant along the WHOLE trajectory, not merely at its
top (job 22250760: on 21 chains where the mirrored rollout tracked, hand_B = 1 - hand_A to within
0.024). A null result is still worth having -- it separates "wrong sigma regime" from "the learned
map is equivariant" as the operative cause, and only one of those has a cheap fix.
"""

import argparse
import math
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.nn.af3_diffusion import P_MEAN, P_STD, S_MAX, SIGMA_DATA
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import _ca_dihedrals

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)
# Training-sigma quantiles -> the S_MAX that makes the sampler START there. sigma = SIGMA_DATA*S_MAX,
# so S_MAX = quantile/SIGMA_DATA. z-scores are the standard normal quantiles.
QUANTILES = [("p50", 0.0), ("p90", 1.2816), ("p99", 2.3263), ("p99.9", 3.0902), ("p99.99", 3.7190)]


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    arms = [("default", None, SIGMA_DATA * S_MAX)]
    for name, z in QUANTILES:
        sig = SIGMA_DATA * math.exp(P_MEAN + P_STD * z)
        arms.append((f"train_{name}", sig / SIGMA_DATA, sig))
    print("ARMS (sampler starting sigma):")
    for n, sm, sig in arms:
        print(f"  {n:14s} S_MAX={sm if sm is None else f'{sm:.3f}':>10}  sigma_start={sig:9.2f}")

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
    print(f"\n[load] EMA @ step {ck.get('global_step')}", flush=True)
    model = model.to(dev).eval()

    # ⛔ Same chains AND same seed for every arm: only the starting sigma differs, so any change is
    # attributable. Re-drawing chains per arm would confound the comparison with chain difficulty.
    batches = []
    it = iter(dm.val_dataloader())
    for _ in range(args.n):
        raw = next(it)
        b = model._prepare(raw, train=False)
        batches.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})

    print(f"\n{'arm':>14} {'sigma_start':>12} {'mirrored':>10} {'rmsd_mean':>10} {'rmsd_med':>9}")
    for name, sm, sig_start in arms:
        hands, rmsds = [], []
        for i, b in enumerate(batches):
            L = b["mask"].shape[1]
            keep = b["mask"][0].bool()
            gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                torch.manual_seed(4242 + i)
                c = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                        b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                        n_steps=args.steps, s_max=sm)
            ca = c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            h = helix_pos_frac(ca)
            if not np.isnan(h):
                hands.append(h)
            rmsds.append(kabsch_rmsd(ca, gt))
        hn, rn = np.array(hands), np.array(rmsds)
        print(f"{name:>14} {sig_start:12.2f} {100*float((hn > 0.5).mean()):9.1f}% "
              f"{rn.mean():10.3f} {np.median(rn):9.3f}", flush=True)


if __name__ == "__main__":
    main()
