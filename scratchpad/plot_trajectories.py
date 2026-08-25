"""Side-by-side denoising trajectories: local_attn vs tri_mul, 4 chains.

Reads the sampletrace .npz snapshots (contact-map predictions every N steps, shape
[n_snapshots, nsamples, L, L]) and the .json per-step record, and renders one row per chain
with columns = trajectory time points, for each arm.

The scalar metrics cannot show WHY a strong single-step denoiser generates poorly. These maps
can: a model that never commits stays grey at every t, while one that resolves shows contacts
sharpening into a defined pattern.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves/traces"
OUT = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves/trajectories.png"
ARMS = [("la", "local_attn"), ("tri", "tri_mul")]
N_CHAINS = 4


def load(tag):
    npz = os.path.join(T, f"trace_{tag}_50_maps.npz")
    js = os.path.join(T, f"trace_{tag}_50.json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None
    z = np.load(npz)
    with open(js) as fh:
        rec = json.load(fh)
    # Snapshots accumulate across all 8 batches; the FIRST batch is the 4 chains we visualise.
    nsteps = int(rec["args"]["nsteps"])
    per_batch = len([s for s in z["steps"] if True]) // 8 if len(z["steps"]) >= 8 else len(z["steps"])
    return z, rec, per_batch


data = {}
for tag, _ in ARMS:
    d = load(tag)
    if d is None:
        print(f"MISSING trace for {tag}")
    data[tag] = d

avail = [t for t, _ in ARMS if data.get(t) is not None]
if not avail:
    raise SystemExit("no traces found")

# Column time points: take the first batch's snapshots
ref = data[avail[0]]
z0, rec0, per_batch = ref
sel = list(range(min(per_batch, 6)))
ncol = len(sel)

fig, axes = plt.subplots(N_CHAINS * len(avail), ncol, figsize=(2.1 * ncol, 2.1 * N_CHAINS * len(avail)))
fig.patch.set_facecolor("white")
if axes.ndim == 1:
    axes = axes[None, :]

row = 0
for tag, label in ARMS:
    if data.get(tag) is None:
        continue
    z, rec, pb = data[tag]
    maps, steps, ts = z["maps"], z["steps"], z["ts"]
    for ch in range(N_CHAINS):
        for ci, si in enumerate(sel):
            ax = axes[row, ci]
            m = maps[si]
            arr = m[ch] if m.ndim == 3 and m.shape[0] > ch else m
            ax.imshow(arr, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"{label}\nchain {ch}", fontsize=7)
            if row == 0:
                ax.set_title(f"t={ts[si]:.2f}", fontsize=8)
        row += 1

fig.suptitle("Denoising trajectories (contact probability, shared 0-1 colour scale)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
for tag, label in ARMS:
    if data.get(tag) is None:
        continue
    rec = data[tag][1]
    st = rec["steps"][:int(rec["args"]["nsteps"])]
    print(f"\n=== {label}: first-batch trajectory ===")
    for r in st[::10] + [st[-1]]:
        print(f"  step={r['step']:3d} t={r['t']:.2f} pred_mean={r['pred_mean']:.4f} "
              f"frac>0.5={r['pred_frac_gt_half']:.4f} sc={r['sc_present']} is_prev={r['sc_is_prev_pred']}")
