"""Independent generation: sample structures from the c2c model and score them honestly.

⛔ WHY THIS IS NOT val/rmsd. `val/rmsd` is a DENOISING error: the model receives the ground truth
plus noise at a sampled sigma and recovers it, and EDM preconditioning
(x_denoised = c_skip*x_noisy + c_out*F, c_skip -> 1 as sigma -> 0) means low-noise draws score well
almost for free. This script instead runs the FULL reverse diffusion from pure noise, conditioned
only on the contact map and sequence, and reports:

  ca_rmsd_A    CA-RMSD after optimal superposition (Kabsch). The number people mean by "RMSD".
  tm           USalign TM-score, sequence-dependent (-TMscore 5): generated vs native are the SAME
               chain, same length, index correspondence -- this is the same-sequence case, where
               -TMscore 0 would be wrong.
  dist_mae_A   mean |d_gen - d_gt| over CA pairs. Alignment-free, so no superposition can flatter it.
  rg_ratio     radius of gyration vs native: catches "right fold, wrong size" and collapse.

Reports every sampled chain, not a mean: the set is small and a mean over a bimodal set misleads.
"""

import argparse
import glob
import os
import subprocess
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import FULL_INFERENCE_STEPS
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import write_atom14_pdb

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def kabsch_rmsd(a, b, allow_reflection=False):
    """CA-RMSD after optimal superposition. a, b: [L, 3].

    allow_reflection=True permits an IMPROPER rotation (det = -1), i.e. it superimposes the mirror
    image. A structure with correct pairwise distances but the wrong HANDEDNESS has a large proper
    RMSD and a small improper one -- reflection preserves every distance exactly, so a contact map
    (or any distance-only readout) cannot distinguish the two. Comparing the pair is the decisive
    test for a mirrored fold.
    """
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = 1.0 if allow_reflection else np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ rot) - b) ** 2).sum(-1).mean()))


def usalign_tm(gen_pdb, gt_pdb, usalign):
    """Sequence-DEPENDENT TM (-TMscore 5): same chain, same length, index correspondence."""
    try:
        out = subprocess.run([usalign, gen_pdb, gt_pdb, "-TMscore", "5"],
                             capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # ⛔ Read the value normalised by the TRUE NATIVE (Structure_2 = the gt file), never max() of
    # the two -- max() inflates ~1.5-2x and has burned this project three times. The earlier
    # startswith("TM-score=") parse returned n/a for every chain: USalign indents these lines.
    import re
    for line in out.splitlines():
        if "TM-score=" in line and "Structure_2" in line:
            m = re.search(r"TM-score=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=FULL_INFERENCE_STEPS)
    ap.add_argument("--out", default="/orcd/scratch/orcd/011/chenxiou/c2c_gen")
    ap.add_argument("--usalign", default="USalign")
    ap.add_argument("--dataset",
                    default="pdb_train_contact-confind-topology_S25_max384_purge-test_cutoff-190828")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")

    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # ⛔ Prefer the EMA weights: state_dict is the raw unaveraged model, which nothing was measured on.
    if "ema" in ck:
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
        src = f"EMA (decay {ck['ema'].get('decay')})"
    else:
        model.load_state_dict(ck["state_dict"], strict=True)
        src = "state_dict (no EMA present)"
    print(f"[load] {src} from {args.ckpt} @ step {ck.get('global_step')}", flush=True)
    model = model.to(dev).eval()

    print(f"\n{'chain':>10} {'L':>5} {'ca_rmsd_A':>10} {'mirror_rmsd':>11} {'tm':>7} "
          f"{'dist_mae_A':>11} {'rg_ratio':>9} {'verdict':>7}")
    rows = []
    it = iter(dm.val_dataloader())
    for i in range(args.n):
        b = model._prepare(next(it), train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
            coords = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                         b["atom_to_token"], b["atom_mask"], n_steps=args.steps)
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()
        gen14 = coords.reshape(-1, L, 14, 3)[0]
        gt14 = b["atom_pos"].reshape(-1, L, 14, 3)[0]
        name = f"gen{i:02d}"
        gp, tp = os.path.join(args.out, f"{name}_gen.pdb"), os.path.join(args.out, f"{name}_gt.pdb")
        write_atom14_pdb(gp, gen14, b["aatype"][0], b["mask"][0])
        write_atom14_pdb(tp, gt14, b["aatype"][0], b["mask"][0])

        ca_g = gen14[keep][:, 1, :].float().cpu().numpy()
        ca_t = gt14[keep][:, 1, :].float().cpu().numpy()
        rmsd = kabsch_rmsd(ca_g, ca_t)
        rmsd_mir = kabsch_rmsd(ca_g, ca_t, allow_reflection=True)
        dg = np.linalg.norm(ca_g[:, None] - ca_g[None], axis=-1)
        dt = np.linalg.norm(ca_t[:, None] - ca_t[None], axis=-1)
        iu = np.triu_indices(len(ca_g), 1)
        mae = float(np.abs(dg[iu] - dt[iu]).mean())
        rg = float(np.sqrt((dg ** 2).sum() / (2 * len(ca_g) ** 2)) /
                   np.sqrt((dt ** 2).sum() / (2 * len(ca_t) ** 2)))
        tm = usalign_tm(gp, tp, args.usalign)
        mirrored = rmsd_mir < 0.5 * rmsd and rmsd > 4.0
        rows.append((name, int(keep.sum()), rmsd, rmsd_mir, tm, mae, rg, mirrored))
        print(f"{name:>10} {int(keep.sum()):>5} {rmsd:>10.2f} {rmsd_mir:>11.2f} "
              f"{('%.3f' % tm) if tm is not None else '  n/a':>7} {mae:>11.2f} {rg:>9.2f} "
              f"{'MIRROR' if mirrored else '':>7}", flush=True)

    print(f"\nwrote {2*len(rows)} PDBs to {args.out}")
    n_mir = sum(1 for r in rows if r[7])
    print(f"⚠️ GENERATION numbers (full reverse diffusion from noise), not the denoising val/rmsd.")
    print(f"   TM>0.5 is the usual 'same fold' threshold; random pairs sit near 0.17-0.3.")
    print(f"⭐ MIRRORED (correct distances, wrong handedness): {n_mir}/{len(rows)}.")
    print("   A contact map is chirality-blind -- reflection preserves every pairwise distance --")
    print("   so distance-only metrics (dist_mae, distogram, smooth_lddt) CANNOT see this failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
