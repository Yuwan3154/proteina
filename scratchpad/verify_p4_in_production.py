"""DOUBLE-CHECK that the P4 mlm_acc fix is working IN PRODUCTION, not just in the unit test.

The test proved Lightning's batch_size weighting behaves as assumed on a toy Trainer. It did NOT
prove the deployed metric is token-weighted on the live run -- a wrong batch_size, a stale checkout,
or an aggregation surprise would all still produce a plausible-looking number.

Three quantities that must line up if the fix is live:

  A  naive mean of mlm_acc_step          = what the OLD, diluted metric reported
  B  mlm_acc_epoch                       = what the FIXED metric reports
  C  sum(mlm_correct) / sum(mlm_n_masked) = the exact token-weighted accuracy, computed from two
                                            separately-logged raw quantities

If P4 is live, B should equal C (that is what the fix computes) and exceed A (zero-mask steps no
longer drag the average down). If B matched A instead, the fix would not be in effect.

⛔ mlm_correct and mlm_n_masked are logged as MEANS over their aggregation windows, not sums, so C is
formed from the epoch-level means over matched rows.
"""

import numpy as np
import wandb

api = wandb.Api()
r = api.run("kryst3154-massachusetts-institute-of-technology/protein_transformer_big_runs/tri_cb8synth_v5")
df = r.history(samples=100000, pandas=True)

STEP = "validation_loss/mlm_acc_step"
EPOCH = "validation_loss/mlm_acc_epoch"
CORR = "validation_loss/mlm_correct_epoch"
NMASK = "validation_loss/mlm_n_masked_epoch"

for c in (STEP, EPOCH, CORR, NMASK):
    print(f"  {c:46} {'present' if c in df.columns else '⛔ ABSENT'}")

a = df[STEP].dropna().to_numpy() if STEP in df.columns else np.array([])
b = df[EPOCH].dropna().to_numpy() if EPOCH in df.columns else np.array([])
print(f"\nA  naive mean of mlm_acc_step      = {a.mean():.4f}   (n={len(a)})  <- the OLD diluted value")
print(f"B  mean of mlm_acc_epoch           = {b.mean():.4f}   (n={len(b)})  <- the FIXED metric")

if CORR in df.columns and NMASK in df.columns:
    sub = df[[CORR, NMASK]].dropna()
    c_val = sub[CORR].to_numpy()
    n_val = sub[NMASK].to_numpy()
    c_ratio = float((c_val * 1.0).sum() / (n_val * 1.0).sum())
    print(f"C  sum(correct)/sum(n_masked)      = {c_ratio:.4f}   (n={len(sub)})  <- exact token-weighted")
    print(f"\n   |B - C| = {abs(b.mean() - c_ratio):.4f}   |B - A| = {abs(b.mean() - a.mean()):.4f}")
    if abs(b.mean() - c_ratio) < 0.02 and b.mean() > a.mean():
        print("   ⭐ P4 IS LIVE: the epoch metric matches the exact token-weighted value and")
        print("      exceeds the diluted step-mean, exactly as the fix intends.")
    elif abs(b.mean() - a.mean()) < 0.005:
        print("   ⛔ the epoch metric matches the DILUTED value -- the fix is NOT in effect.")
    else:
        print("   ⚠️ inconclusive: B matches neither cleanly. Investigate before trusting mlm_acc.")
else:
    print("\n⛔ mlm_correct / mlm_n_masked not both present -- cannot form the exact ratio.")

BASE = "validation_loss/mlm_acc_marginal_baseline_epoch"
if BASE in df.columns:
    bv = df[BASE].dropna().to_numpy()
    q = max(1, len(bv) // 4)
    print(f"\nP5 marginal baseline: n={len(bv)}  first-q {np.median(bv[:q]):.4f} -> "
          f"last-q {np.median(bv[-q:]):.4f}")
    print("   (a RUNNING most-common-token predictor: it should settle as the histogram fills,")
    print("    not drift arbitrarily)")
else:
    print(f"\n⛔ {BASE} ABSENT -- P5 not logging.")
