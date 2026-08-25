"""Numerically verify WHY the published trajectory figure shows padding for tri and not for local_attn.

Two claims to test against the saved sampletrace maps, rather than against source reading:

  C1. hier multiplies its output logits by the pair mask (contact_map_hier.py:1209) so its padded
      cells are EXACTLY sigmoid(0) = 0.5; ContactMapTriSiT does not (contact_map_tri.py:220-222),
      so its padded cells hold a learned constant != 0.5. plot_trajectories._crop_valid only
      removes cells within 1e-3 of 0.5, hence it cropped one arm and not the other.

  C2. tri's validation sampling overwrote the mask with a RANDOM length draw
      (model_trainer_base.py:3168-3180), so tri's valid region is not the chain's own length.
      The per-chain non-constant extent should match the "variable length L=" values in the log,
      not the chain lengths.
"""

import json
import os

import numpy as np

T = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves/traces"


def const_rows(a, value, tol=1e-3):
    """Indices whose whole row sits within tol of `value`."""
    return np.where(np.all(np.abs(a - value) < tol, axis=1))[0]


def report(tag):
    npz = os.path.join(T, f"trace_{tag}_50_maps.npz")
    js = os.path.join(T, f"trace_{tag}_50.json")
    if not os.path.exists(npz):
        print(f"[{tag}] MISSING {npz}")
        return
    z = np.load(npz)
    maps = z["maps"]
    print(f"\n===== {tag} =====")
    print(f"maps shape {maps.shape}  steps {z['steps'][:6]} ...  ts {z['ts'][:6]} ...")
    with open(js) as fh:
        rec = json.load(fh)
    print(f"args: {rec['args']}")

    snap = maps[0]  # first snapshot of the first traced batch
    for ch in range(snap.shape[0]):
        a = snap[ch]
        n = a.shape[-1]
        # The padded border is whatever constant fills the trailing rows.
        tail_val = float(a[-1, -1])
        rows_at_tail = const_rows(a, tail_val)
        rows_at_half = const_rows(a, 0.5)
        # First index of the trailing constant block.
        first_pad = n
        for i in range(n - 1, -1, -1):
            if np.all(np.abs(a[i] - tail_val) < 1e-3):
                first_pad = i
            else:
                break
        print(
            f"  chain {ch}: n={n}  trailing-const value={tail_val:.6f}  "
            f"|value-0.5|={abs(tail_val - 0.5):.2e}  valid extent={first_pad}  "
            f"pad width={n - first_pad}  rows==tail={len(rows_at_tail)}  rows==0.5={len(rows_at_half)}"
        )


for tag in ("la", "tri"):
    report(tag)
