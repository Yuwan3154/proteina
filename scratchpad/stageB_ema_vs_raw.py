"""EMA vs raw c2c weights on the Stage-B native-map arm: the user's condition for using EMA (2026-09-22).

Pre-registered rule (PLAN, written before any result): per c2c checkpoint, the --native arm is run with
--weights ema and --weights raw on the same chain list with identical seeds (paired by chain). Use EMA
unless raw is significantly better on refl-sign (exact McNemar, p < 0.05) or on TM-score (Wilcoxon
signed-rank, p < 0.05). Both are reported either way.

Usage: python scratchpad/stageB_ema_vs_raw.py <ema.jsonl> <raw.jsonl>
"""

import json
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

ALPHA = 0.05


def load(path):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    assert all(r["map_source"] == "native" for r in rows), f"{path}: not a --native run"
    by = {r["stem"]: r for r in rows}
    assert len(by) == len(rows), f"{path}: duplicate chains (n_seeds must be 1 here)"
    return by, rows[0]


ema, e0 = load(sys.argv[1])
raw, r0 = load(sys.argv[2])
assert e0["weights"] == "ema" and r0["weights"] == "raw", (e0["weights"], r0["weights"])
assert e0["ckpt"] == r0["ckpt"] and e0["global_step"] == r0["global_step"], "different checkpoints"
common = sorted(set(ema) & set(raw))
assert common and len(common) == len(ema) == len(raw), \
    f"chain sets differ: ema {len(ema)}, raw {len(raw)}, common {len(common)}"
assert all(ema[s]["torch_seed"] == raw[s]["torch_seed"] and ema[s]["seq_fp"] == raw[s]["seq_fp"]
           for s in common), "seeds or sequences differ between the two runs -- not paired"

n = len(common)
es = np.array([ema[s]["refl_sign"] for s in common])
rs = np.array([raw[s]["refl_sign"] for s in common])
etm = np.array([ema[s]["tm"] for s in common])
rtm = np.array([raw[s]["tm"] for s in common])
b = int(((es == 1) & (rs == 0)).sum())   # EMA mirrored, raw not
c = int(((es == 0) & (rs == 1)).sum())   # raw mirrored, EMA not
p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
d = rtm - etm
p_w = wilcoxon(rtm, etm).pvalue if np.any(d != 0) else 1.0

print(f"checkpoint {e0['ckpt']}  global_step {e0['global_step']}  chains {n}  dataset {e0['dataset']}")
print(f"{'':12s} {'EMA':>8s} {'raw':>8s}")
print(f"{'refl-sign':12s} {es.mean():8.3f} {rs.mean():8.3f}   discordant: EMA-only {b}, raw-only {c}, "
      f"exact McNemar p={p_mc:.4f}")
print(f"{'TM mean':12s} {etm.mean():8.3f} {rtm.mean():8.3f}   Wilcoxon p={p_w:.4f}")
print(f"{'TM median':12s} {np.median(etm):8.3f} {np.median(rtm):8.3f}")
for thr in (0.5, 0.7):
    print(f"{'TM>=' + str(thr):12s} {int((etm >= thr).sum()):8d} {int((rtm >= thr).sum()):8d}   (counts of {n})")
for k in ("rmsd_proper", "dist_mae"):
    ev, rv = (np.array([x[s][k] for s in common]) for x in (ema, raw))
    print(f"{k + ' med':12s} {np.median(ev):8.3f} {np.median(rv):8.3f}")
raw_better = (c < b and p_mc < ALPHA) or (np.median(d) > 0 and p_w < ALPHA)
print("VERDICT:", "use RAW (raw significantly better)" if raw_better else "use EMA (raw not significantly better)")
