"""Mirror rate for SEVERAL checkpoints on ONE FIXED set of chains, paired per chain.

⛔⛔ WHY THIS EXISTS. Every arm-vs-control mirror claim so far has come from the runs' own
validation dumps, and validation DRAWS DIFFERENT CHAINS EVERY ROUND. That makes a per-round rate not
a curve, forces the control to be read off a noisy descending band by interpolation, and has already
produced one retracted z=+2.23. Here the chain set is IDENTICAL across checkpoints and the starting
noise is seeded per chain, so every checkpoint answers the same 16 questions and the comparison is
PAIRED. No interpolation, no binning, no matched-dist_mae proxy.

⛔⛔ KNOWN-GOOD CONTROL (this is not optional). A null is worthless from a gate that has never been
shown to fire. Two calibration checkpoints bracket the effect: tbeta@8,076 is from the run's
MIRRORED era, tbeta@9,750 is where refl-sign had already fallen to 0.000. If those two do not
separate on this chain set, the assay is broken and nothing else in the table may be read.

⛔ refl-sign (proper > reflected) is the statistic, NOT is_mirrored -- is_mirrored needs
proper > 2x reflected and so cannot fire when both superpositions land similarly, which is exactly
the regime a half-fit structure is in. dist_mae is printed beside it because a worse fit drags
refl-sign toward 0.5, and that confound must be visible rather than assumed away.

⭐ Rows are appended to JSONL after EVERY checkpoint, so a wall-clock kill still leaves the
checkpoints already measured.
"""

import argparse
import hashlib
import json
import os
import sys
from math import comb

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hydra
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import handedness_metrics

MODEL_CFG = dict(
    c_s=384, c_z=128, c_token=768, c_atom=128, c_atompair=16,
    n_blocks=24, n_heads=16, n_tri_blocks=4, tri_hidden=128, transition_n=2,
    atom_blocks=3, atom_heads=4,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", required=True, help="';'-separated list of label=path")
ap.add_argument("--dataset", default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--n_chains", type=int, default=16)
ap.add_argument("--skip_chains", type=int, default=0,
                help="discard this many validation batches BEFORE collecting, giving a DISJOINT "
                     "chain set for an independent replication. Protocol is otherwise identical.")
ap.add_argument("--n_seeds", type=int, default=1,
                help="rollouts per chain from DIFFERENT starting noise. >1 turns the fixed-seed "
                     "design into a per-chain RATE, which is what the per-target-vs-coin-flip "
                     "question needs; seed_idx 0 reproduces the single-seed runs exactly.")
ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
ap.add_argument("--out", required=True)
args = ap.parse_args()

# ⛔ ';' not ',' -- sbatch --export splits on commas and would silently deliver only the first entry.
SPECS = []
for item in args.ckpts.split(";"):
    if not item.strip():
        continue
    label, path = item.split("=", 1)
    SPECS.append((label.strip(), path.strip()))
assert len(SPECS) >= 2, "need at least the two calibration checkpoints"
for label, path in SPECS:
    assert os.path.exists(path), f"missing checkpoint for {label}: {path}"

CFG_DIR = os.path.join(REPO, "configs", "datasets_config", "pdb")
assert os.path.isdir(CFG_DIR), f"config dir missing: {CFG_DIR}"
with hydra.initialize_config_dir(CFG_DIR, version_base=hydra.__version__):
    cfg_data = hydra.compose(config_name=args.dataset)
OmegaConf.set_struct(cfg_data, False)
cfg_data.datamodule.num_workers = 0
cfg_data.datamodule.prefetch_factor = None
dm = hydra.utils.instantiate(cfg_data.datamodule)
dm.setup("fit")

# Pull the chains ONCE and keep them on CPU. Every checkpoint is then scored on these same tensors.
raw_batches = []
it = iter(dm.val_dataloader())
for _ in range(args.skip_chains):
    next(it)
while len(raw_batches) < args.n_chains:
    raw_batches.append(next(it))
print(f"[data] cached {len(raw_batches)} validation batches after skipping {args.skip_chains} "
      f"(fixed across all checkpoints)", flush=True)

dev = "cuda"
model = ContactToCoordTrainer(model_cfg=MODEL_CFG).to(dev).eval()

fingerprints = {}   # chain key -> hash of ground-truth CA coords, to prove the set never moved
results = {}

with open(args.out, "w") as fh:
    for label, path in SPECS:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # ⛔⛔ WHY --weights MATTERS, measured 2026-09-18. A warm-started arm re-initialises its EMA
        # FROM the parent, so at decay 0.999 the PARENT's coefficient in the arm's EMA is 0.999^t --
        # 72.6% at 320 steps, 58.8% at 530. Reading the EMA of a young arm therefore measures mostly
        # the PARENT, which is exactly how both FAPE arms came out indistinguishable from their
        # ancestor while a same-era control differed on 16 of 64 chains. The control's own EMA has
        # been running 8,500 steps and carries no such init, so EMA-vs-EMA is NOT apples to apples
        # here. Use --weights raw to read what the arm's own gradients actually did.
        if args.weights == "ema" and "ema" in ck:
            sd = dict(ck["ema"]["params"])
            missing, unexpected = model.model.load_state_dict(sd, strict=False)
            decay = ck["ema"].get("decay")
            src = f"EMA(decay={decay})"
            gs = ck.get("global_step")
            if isinstance(decay, float) and isinstance(gs, int) and gs > 0:
                # decay**steps_since_warm_start = how much of this EMA is still the PARENT's weights
                src += f" parent_coeff={decay ** gs:.3f}"
        else:
            sd = ck["state_dict"] if "state_dict" in ck else ck
            missing, unexpected = model.load_state_dict(sd, strict=False)
            src = "RAW state_dict"
        gstep = ck.get("global_step")
        # to_hand_s exists only in the p_mirror-capable arms; it is UNUSED at p_mirror=0, so its
        # absence from the tbeta checkpoints is expected and harmless.
        assert len(missing) < 20, f"{label}: {len(missing)} missing params -- wrong MODEL_CFG?"
        print(f"[load] {label}: {src} global_step={gstep} missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        del ck, sd

        rows = []
        for bi, raw in enumerate(raw_batches):
            b = model._prepare(raw.to(dev), train=False)
            L = b["mask"].shape[1]
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                per_seed = []
                for si in range(args.n_seeds):
                    # ⛔ si=0 MUST give 1234+bi so multi-seed runs stay bit-comparable with every
                    # single-seed run already recorded. The offset only kicks in for si>=1.
                    torch.manual_seed(1234 + bi + si * 1_000_000)
                    per_seed.append(model.model.rollout(
                        s, z, b["mask"], b["ref_feats"], b["ref_pos"], b["atom_to_token"],
                        b["atom_mask"], b["ref_space_uid"], n_steps=args.steps))
            gt_all = b["atom_pos"].reshape(-1, L, 14, 3)
            for si, coords in enumerate(per_seed):
                gen_all = coords.reshape(-1, L, 14, 3)
                for j in range(gen_all.shape[0]):
                    m = b["mask"][j].bool().cpu().numpy()
                    g = gen_all[j, :, 1, :].float().cpu().numpy()[m]
                    t = gt_all[j, :, 1, :].float().cpu().numpy()[m]
                    if len(g) < 10:
                        continue
                    key = f"{bi}_{j}"
                    fp = hashlib.sha1(np.ascontiguousarray(t).tobytes()).hexdigest()[:12]
                    # ⛔ SAME POPULATION, asserted rather than assumed: a checkpoint scored on a
                    # different chain than its predecessor would make the pairing a lie.
                    if key in fingerprints:
                        assert fingerprints[key] == fp, f"chain {key} CHANGED between checkpoints"
                    else:
                        fingerprints[key] = fp
                    h = handedness_metrics(g, t)
                    if not h:
                        continue
                    dmae = float(np.abs(np.linalg.norm(g[:, None] - g[None], axis=-1)
                                        - np.linalg.norm(t[:, None] - t[None], axis=-1)).mean())
                    rec = dict(label=label, global_step=gstep, chain=key, seed_idx=si, fp=fp,
                               n_res=int(m.sum()),
                               rmsd_proper=float(h["rmsd_proper"]),
                               rmsd_reflected=float(h["rmsd_reflected"]),
                               dist_mae=dmae, is_mirrored=float(h["is_mirrored"]),
                               refl_sign=1.0 if h["rmsd_proper"] > h["rmsd_reflected"] else 0.0)
                    rows.append(rec)
                    fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        results[label] = rows
        print(f"[done] {label}: n={len(rows)} written", flush=True)

if torch.cuda.is_available():
    # Recorded so the real GPU-memory bound for this payload is known rather than estimated -- it
    # decides whether a 48 GB card can host this assay at all.
    print(f"\n[gpu] {torch.cuda.get_device_name(0)}  peak allocated "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB, peak reserved "
          f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB", flush=True)

print(f"\n{'checkpoint':>26} {'step':>7} {'n':>4} {'proper':>8} {'refl':>8} {'distMAE':>8} "
      f"{'refl-sign':>10} {'is_mirr':>8}")
for label, _ in SPECS:
    r = results.get(label, [])
    if not r:
        print(f"{label:>26} {'-':>7} {0:>4}")
        continue
    mean = lambda k: float(np.mean([x[k] for x in r]))
    print(f"{label:>26} {r[0]['global_step']:>7} {len(r):>4} {mean('rmsd_proper'):>8.2f} "
          f"{mean('rmsd_reflected'):>8.2f} {mean('dist_mae'):>8.2f} {mean('refl_sign'):>10.3f} "
          f"{mean('is_mirrored'):>8.3f}")


def exact_two_sided(b, c):
    """Exact two-sided sign test on the discordant pairs (McNemar, no normal approximation)."""
    n = b + c
    if n == 0:
        return 1.0, n
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), n


print("\nPAIRED refl-sign vs the first checkpoint (McNemar, exact two-sided):")
base_label = SPECS[0][0]
# ⛔ Key on (chain, seed_idx), NOT chain alone: with --n_seeds > 1 a chain-only key would silently
# keep just the last seed and quietly discard the rest of the sample.
pkey = lambda x: (x["chain"], x.get("seed_idx", 0))
base = {pkey(x): x for x in results.get(base_label, [])}
for label, _ in SPECS[1:]:
    cur = {pkey(x): x for x in results.get(label, [])}
    shared = sorted(set(base) & set(cur))
    b = sum(1 for k in shared if base[k]["refl_sign"] > cur[k]["refl_sign"])   # mirror -> fixed
    c = sum(1 for k in shared if base[k]["refl_sign"] < cur[k]["refl_sign"])   # fixed -> mirror
    p, nd = exact_two_sided(b, c)
    print(f"  {base_label} -> {label:>22}: paired n={len(shared)}, mirrored->fixed {b}, "
          f"fixed->mirrored {c}, discordant {nd}, p={p:.4f}")
print(f"\n[chains] {len(fingerprints)} distinct chains, fingerprints identical across all "
      f"{len(SPECS)} checkpoints")
