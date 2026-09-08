"""Is the SAMPLER reflection-equivariant? Roll out from a noise seed and from its MIRROR.

⭐ THIS TEST NEEDS NO THRESHOLD AND NO CONTROL ARM, which is exactly why it replaces the failed
`test_reflection_equivariance.py`. That one compared a reflection residual against a proper-rotation
residual, and the control was invalid by construction: `weighted_rigid_align` aligns ground truth
ONTO the prediction, so the loss never constrains the output's global orientation and rotation
equivariance is never taught.

The logic here is exact. The sampler is deterministic given its starting noise. So if the denoiser f
commutes with a reflection M, the ENTIRE rollout commutes with it, and
        rollout(M e)  ==  M rollout(e)
identically -- not approximately. So:
    A = rollout(e1)
    B = rollout(M e1)     d_equiv = RMSD(B, M A)   -> ~0 IFF the sampler is reflection-equivariant
    C = rollout(e2)       d_scale = RMSD(C, M A)   -> the scale of two UNRELATED structures
    ratio = d_equiv / d_scale
ratio ~ 0  => the mirror is ARCHITECTURALLY FREE: output hand is slaved to the input noise's hand,
              and no amount of right-handed training data can teach a prior, because the model never
              learns a prior -- it learns a MAP.
ratio ~ 1  => the sampler genuinely breaks reflection symmetry, and the 50/50 comes from elsewhere.

⛔ RMSD is det=+1 Kabsch. An improper fit would call a mirror a match and invert the whole result.
"""

import argparse
import os
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
    atom_blocks=3, atom_heads=4, n_diffusion_samples=8,
)


def kabsch_rmsd(a, b):
    """CA-RMSD under a PROPER (det=+1) superposition."""
    a = a - a.mean(0)
    b = b - b.mean(0)
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
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--steps", type=int, default=50)
    # ⛔ Taken verbatim from gen_c2c_structures.py:95, the script that actually produced the 254-chain
    # sweep. Do not guess this name -- an invented one fails with a hydra MissingConfigException.
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

    model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert "ema" in ck, "refusing to score the unaveraged model"
    model.model.load_state_dict(ck["ema"]["params"], strict=True)
    print(f"[load] EMA @ step {ck.get('global_step')} from {args.ckpt}", flush=True)
    model = model.to(dev).eval()

    M = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=dev))
    assert torch.linalg.det(M) < 0

    sig0 = noise_schedule(torch.linspace(0.0, 1.0, args.steps + 1, device=dev))[0]
    print(f"[sched] sigma_0 = {sig0.item():.2f}")
    print(f"\n{'chain':>6} {'L':>5} {'hand_A':>7} {'hand_B':>7} {'d_equiv':>8} {'d_scale':>8} {'ratio':>7}")

    it = iter(dm.val_dataloader())
    rows = []
    for i in range(args.n):
        raw = next(it)
        b = model._prepare(raw, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        A_atoms = b["atom_mask"].shape[1]

        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])

            def roll(x0, churn):
                # ⛔ augment_steps stays False: it re-randomises the frame every step, which would
                # swamp the residual we are trying to measure.
                return model.model.rollout(
                    s, z, b["mask"], b["ref_feats"], b["ref_pos"], b["atom_to_token"],
                    b["atom_mask"], b["ref_space_uid"], n_steps=args.steps, x_init=x0,
                    augment_steps=False, churn_noise=churn)

            g = torch.Generator(device=dev).manual_seed(1000 + i)
            e1 = sig0 * torch.randn(1, A_atoms, 3, device=dev, generator=g)
            e2 = sig0 * torch.randn(1, A_atoms, 3, device=dev, generator=g)
            # ⛔⛔ The sampler injects FRESH per-step EDM churn noise. Two rollouts drawing different
            # churn diverge for reasons that have nothing to do with chirality, which would make the
            # whole test uninformative. Pin the sequence, and REFLECT it for the mirrored arm so the
            # trajectory genuinely commutes with M.
            churn = torch.randn(args.steps, 1, A_atoms, 3, device=dev, generator=g)
            churn_M = churn @ M.T
            churn2 = torch.randn(args.steps, 1, A_atoms, 3, device=dev, generator=g)
            cA = roll(e1, churn)
            cB = roll(e1 @ M.T, churn_M)
            cC = roll(e2, churn2)

        def ca(c):
            return c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()

        caA, caB, caC = ca(cA), ca(cB), ca(cC)
        mA = caA @ np.diag([1.0, 1.0, -1.0])          # M applied to structure A
        d_equiv = kabsch_rmsd(caB, mA)
        d_scale = kabsch_rmsd(caC, mA)
        ratio = d_equiv / max(d_scale, 1e-9)
        hA, hB = helix_pos_frac(caA), helix_pos_frac(caB)
        rows.append((d_equiv, d_scale, ratio, hA, hB))
        print(f"{i:6d} {int(keep.sum()):5d} {hA:7.3f} {hB:7.3f} {d_equiv:8.3f} {d_scale:8.3f} {ratio:7.3f}",
              flush=True)

    a = np.array(rows, dtype=float)
    print("\n=== SUMMARY ===")
    print(f"  d_equiv  RMSD(rollout(Me), M rollout(e)) : mean {a[:,0].mean():.3f} A  median {np.median(a[:,0]):.3f}")
    print(f"  d_scale  RMSD(unrelated seed, M A)       : mean {a[:,1].mean():.3f} A  median {np.median(a[:,1]):.3f}")
    print(f"  ratio                                    : mean {a[:,2].mean():.3f}  median {np.median(a[:,2]):.3f}")
    print(f"  hand_A vs hand_B (should MATCH if the sampler mirrors the noise):")
    print(f"     mean |hand_A - hand_B| = {np.nanmean(np.abs(a[:,3] - a[:,4])):.3f}")
    print("\n  ratio ~ 0 => sampler is REFLECTION-EQUIVARIANT: the mirror is architecturally free.")
    print("  ratio ~ 1 => sampler breaks reflection symmetry; the 50/50 is elsewhere.")


if __name__ == "__main__":
    main()
