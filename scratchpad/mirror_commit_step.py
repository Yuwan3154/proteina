"""WHEN during the reverse diffusion does the sampler commit to a hand?

Two payoffs, one diagnostic and one immediately practical.

DIAGNOSTIC: if the hand is fixed at the very top of the schedule (high sigma, where the input is
nearly pure noise), the decision cannot be coming from the conditioning -- the contact map is achiral
-- so it is being read out of the noise, which is the signature of a learned reflection-equivariant
map. If instead it wanders and settles late, the model carries a chiral prior but applies it
inconsistently, which is a capacity/coverage problem with a different fix.

PRACTICAL: rejection resampling currently pays ~1.92 FULL rollouts per accepted sample. If the hand
is already decided at step k << n_steps, we can detect and abort at k, paying 1 full rollout plus
0.92 partial ones. That is a pure efficiency win with no new hyperparameter -- the commit step is
MEASURED here, not chosen.

Reports the full per-step trace. No threshold is applied; "commit step" is defined operationally as
the earliest step after which the sign of (helix_pos_frac - 0.5) never again disagrees with the final
sample's sign, which is a property of the trace rather than a tunable.
"""

import argparse
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


def helix_pos_frac(ca):
    d = _ca_dihedrals(ca)
    sel = d[(np.abs(d) > 30.0) & (np.abs(d) < 90.0)]
    return float((sel > 0).mean()) if len(sel) >= 5 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=16)
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

    commits, finals, traces = [], [], []
    it = iter(dm.val_dataloader())
    for i in range(args.n):
        raw = next(it)
        b = model._prepare(raw, train=False)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        L = b["mask"].shape[1]
        keep = b["mask"][0].bool()

        rec = []
        with torch.no_grad():
            s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
            torch.manual_seed(7000 + i)
            model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"], b["atom_to_token"],
                                b["atom_mask"], b["ref_space_uid"], n_steps=args.steps, record=rec)

        hs = []
        for sig, d in rec:
            ca = d.reshape(-1, L, 14, 3)[0][keep][:, 1, :].float().cpu().numpy()
            hs.append(helix_pos_frac(ca))
        hs = np.array(hs)
        final = hs[-1]
        finals.append(final)
        fsign = final > 0.5
        # earliest k after which the sign never disagrees again
        disagree = np.where((hs > 0.5) != fsign)[0]
        commit = int(disagree[-1] + 1) if len(disagree) else 0
        commits.append(commit)
        traces.append(hs)
        sigs = [f"{s:.0f}" for s, _ in rec]
        print(f"chain {i:3d} L={int(keep.sum()):4d} final_hand={final:.3f} "
              f"commit_step={commit}/{args.steps} (sigma~{sigs[commit] if commit < len(sigs) else 'NA'})",
              flush=True)

    c = np.array(commits, dtype=float)
    f = np.array(finals)
    print("\n=== SUMMARY ===")
    print(f"  chains                : {len(c)}")
    print(f"  final mirrored (>0.5) : {int((f > 0.5).sum())}/{len(f)} = {100*(f > 0.5).mean():.1f}%")
    print(f"  commit step           : mean {c.mean():.1f}  median {np.median(c):.1f}  "
          f"max {c.max():.0f}  of {args.steps}")
    print(f"  committed in first 20%: {100*float((c <= 0.2*args.steps).mean()):.1f}% of chains")
    print("\n  mean |helix_pos_frac - 0.5| by step decile (0 = undecided, 0.5 = fully committed):")
    T = np.array([t for t in traces])
    for q in range(10):
        lo, hi = int(q * args.steps / 10), int((q + 1) * args.steps / 10)
        seg = np.abs(T[:, lo:hi] - 0.5)
        print(f"    steps {lo:3d}-{hi:3d}: {np.nanmean(seg):.4f}")


if __name__ == "__main__":
    main()
