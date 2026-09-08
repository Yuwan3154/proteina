"""Does the mirror rate improve as training progresses? Decides whether to just train longer.

If the rate is flat across checkpoints, the coin flip will NOT train itself away, and a structural
change (mirror augmentation + hand label) or the post-hoc detector is mandatory. If it is falling,
patience is a legitimate option and the whole question changes.

⛔ SAME chains and SAME seed for every checkpoint, so only the weights differ and any change is
attributable. Uses the repo's own `_ca_dihedrals`; the 0.5 boundary is the midpoint of two measured
modes (ideal right-handed helix 0.000, its mirror 1.000).

⛔ No try/except on the load, per project rules: the checkpoint's key set is compared against the
model's explicitly and a mismatched checkpoint is REPORTED and skipped rather than silently
half-loaded.
"""

import argparse
import glob
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/orcd/scratch/orcd/011/chenxiou/proteina_sh")

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import _ca_dihedrals

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return float(np.sqrt(((a @ (u @ np.diag([1.0, 1.0, d]) @ vt) - b) ** 2).sum(-1).mean()))


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_v1")
    ap.add_argument("--n", type=int, default=24)
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

    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8)).to(dev).eval()
    want = set(model.model.state_dict().keys())

    batches = []
    it = iter(dm.val_dataloader())
    for _ in range(args.n):
        raw = next(it)
        b = model._prepare(raw, train=False)
        batches.append({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()})

    ckpts = sorted(glob.glob(os.path.join(args.dir, "*.ckpt")), key=os.path.getmtime)
    print(f"\n{'checkpoint':>34} {'step':>7} {'mirrored':>10} {'rmsd_med':>9} {'hand_med':>9}")
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "ema" not in ck:
            print(f"{os.path.basename(path):>34} {'-':>7} {'NO EMA -- skipped':>10}")
            continue
        got = set(ck["ema"]["params"].keys())
        if got != want:
            print(f"{os.path.basename(path):>34} {ck.get('global_step'):>7} "
                  f"ARCH MISMATCH (+{len(got-want)}/-{len(want-got)}) -- skipped")
            continue
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
        hands, rmsds = [], []
        for i, b in enumerate(batches):
            L = b["mask"].shape[1]
            keep = b["mask"][0].bool()
            gt = b["atom_pos"].reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                torch.manual_seed(555 + i)
                c = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                        b["atom_to_token"], b["atom_mask"], b["ref_space_uid"],
                                        n_steps=args.steps)
            ca = c.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            h = helix_pos_frac(ca)
            if not np.isnan(h):
                hands.append(h)
            rmsds.append(kabsch_rmsd(ca, gt))
        hn, rn = np.array(hands), np.array(rmsds)
        print(f"{os.path.basename(path):>34} {ck.get('global_step'):>7} "
              f"{100*float((hn > 0.5).mean()):9.1f}% {np.median(rn):9.3f} {np.median(hn):9.3f}",
              flush=True)


if __name__ == "__main__":
    main()
