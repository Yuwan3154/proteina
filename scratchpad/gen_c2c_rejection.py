"""Rejection resampling: redraw a sample whose CA-dihedral hand says it is mirrored.

Ranked #1 by the fix-viability investigation, and the only route that is FREE -- it introduces no
hyperparameter that a reference does not already ship.

WHY REJECTION AND NOT REPAIR. Reflecting a mirrored structure fixes the fold but turns every residue
into its D enantiomer, and the frame-conjugation dodge (reflect, then conjugate frames back) was
measured to leave N/CA/C untouched -- so the usual C(i)-N(i+1) geometry check is a NULL TEST that
cannot fail -- while moving carbonyl O by 1.18 A mean / 2.13 max, sidechains 3.65 A mean / 13.78 max,
driving the peptide improper CA-C-O-N from 1.3 to 54.3 degrees off-planar, and creating 300 heavy-atom
clashes at 0.08 A minimum contact. Rejection returns an UNMODIFIED model sample instead, with no
geometry damage and no corruption of the already-correct chains a detector false-positives.
This is what GeoMol and ET-Flow ship in production for exactly this failure.

⭐ MEASURED PREREQUISITE, established before writing this: the failure is a PURE GLOBAL sign flip.
Generated chains are as internally uniform in hand as natives (purity 0.8843 vs 0.8790, within-chain
window disagreement median 0.0000 for both). There is no mixed-handedness mode, so "reject and
redraw" cannot be defeated by a chain that is half right-handed.

⛔ THE 0.5 THRESHOLD IS NOT INVENTED. It is the midpoint of two MEASURED, well-separated modes:
an ideal right-handed helix gives helix_pos_frac 0.000 and its mirror 1.000
(scratchpad/test_dihedral_convention.py); 254 natives measure median 0.0815 and 254 generations
median 0.5526 with 51.8% above 0.5. MAX_ATTEMPTS is an operational cap, not a model parameter -- it
bounds cost and affects no measurement.
"""

import argparse
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import _ca_dihedrals, write_atom14_pdb

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def kabsch_rmsd(a, b, allow_reflection=False):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max_attempts", type=int, default=6)
    ap.add_argument("--out", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen_reject")
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
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

    print(f"\n{'chain':>6} {'L':>5} {'tries':>6} {'hand':>7} {'rmsd_A':>8} {'baseline_rmsd':>14} {'status':>9}")
    rows = []
    it = iter(dm.val_dataloader())
    for i in range(args.n):
        raw = next(it)
        b = model._prepare(raw, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        gt14 = b["atom_pos"].reshape(-1, L, 14, 3)[0]
        ca_t = gt14[keep][:, 1, :].float().cpu().numpy()

        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
            first_hand = first_rmsd = None
            accepted = None
            for attempt in range(1, args.max_attempts + 1):
                torch.manual_seed(10_000 * i + attempt)
                coords = model.model.rollout(
                    s, z, b["mask"], b["ref_feats"], b["ref_pos"], b["atom_to_token"],
                    b["atom_mask"], b["ref_space_uid"], n_steps=args.steps)
                gen14 = coords.reshape(-1, L, 14, 3)[0]
                ca_g = gen14[keep][:, 1, :].float().cpu().numpy()
                h = helix_pos_frac(ca_g)
                r = kabsch_rmsd(ca_g, ca_t)
                if first_hand is None:
                    first_hand, first_rmsd = h, r
                if not np.isnan(h) and h < 0.5:
                    accepted = (attempt, h, r, gen14)
                    break

        if accepted is None:
            # Report it rather than silently emitting a mirrored structure.
            print(f"{i:6d} {int(keep.sum()):5d} {args.max_attempts:6d} {first_hand:7.3f} "
                  f"{first_rmsd:8.3f} {first_rmsd:14.3f} {'EXHAUSTED':>9}", flush=True)
            rows.append((args.max_attempts, first_hand, first_rmsd, first_rmsd, 0))
            continue
        attempt, h, r, gen14 = accepted
        write_atom14_pdb(os.path.join(args.out, f"gen{i:03d}_gen.pdb"), gen14,
                         b["aatype"][0], b["mask"][0])
        print(f"{i:6d} {int(keep.sum()):5d} {attempt:6d} {h:7.3f} {r:8.3f} {first_rmsd:14.3f} "
              f"{'ok':>9}", flush=True)
        rows.append((attempt, h, r, first_rmsd, 1))

    a = np.array(rows, dtype=float)
    ok = a[:, 4] == 1
    print("\n=== SUMMARY ===")
    print(f"  chains                     : {len(a)}")
    print(f"  accepted                   : {int(ok.sum())}/{len(a)} = {100*ok.mean():.1f}%")
    print(f"  rollouts per accepted sample: {a[:,0].sum()/max(ok.sum(),1):.2f}")
    print(f"  mean attempts              : {a[:,0].mean():.2f}")
    if ok.any():
        print(f"  CA-RMSD accepted           : mean {a[ok,2].mean():.3f}  median {np.median(a[ok,2]):.3f}")
    print(f"  CA-RMSD first draw (baseline): mean {a[:,3].mean():.3f}  median {np.median(a[:,3]):.3f}")
    print(f"  first-draw mirrored          : {float((a[:,1] > 0.5).mean())*100:.1f}%")


if __name__ == "__main__":
    main()
