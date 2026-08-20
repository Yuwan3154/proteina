"""Was validation single-step p@L below its noisy-input floor BEFORE the overfitting set in?

If the model only drops below the floor late, sub-floor behaviour is a symptom of overfitting.
If it was at or below the floor from the start, the model never learned to denoise held-out
contact maps at all -- a modelling problem, not an overfitting one.

Compares model vs floor PAIRED WITHIN A t-BIN at the same logging row. That pairing matters: the
un-binned metric swings with whichever noise level t each logged batch happened to draw, so an
unpaired comparison of two separately-averaged series can invert the sign.
"""
import sys

import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "overfit_localattn_full_history.csv"
# local_attn validation loss bottomed at epoch 63; conv_next at 119. Pre-overfit window ends at the
# run's own minimum, passed in so this script is not hardcoded to one run.
PEAK_EPOCH = float(sys.argv[2]) if len(sys.argv) > 2 else 63.0

df = pd.read_csv(CSV, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()]

BINS = [
    ("all t", "validation_loss/contact_precision_at_L_single_step",
             "validation_loss/contact_precision_at_L_noisy_floor"),
    ("t low", "validation_loss/contact_precision_at_L_single_step_tlow",
              "validation_loss/contact_precision_at_L_noisy_floor_tlow"),
    ("t mid", "validation_loss/contact_precision_at_L_single_step_tmid",
              "validation_loss/contact_precision_at_L_noisy_floor_tmid"),
    ("t high", "validation_loss/contact_precision_at_L_single_step_thigh",
               "validation_loss/contact_precision_at_L_noisy_floor_thigh"),
]

print("Paired model-vs-floor, same row, within t-bin.  PEAK (val-loss minimum) = epoch %.0f\n" % PEAK_EPOCH)
print("%-8s %-26s %6s %9s %9s %9s %8s" % ("bin", "window", "n", "model", "floor", "delta", "%above"))

for label, mkey, fkey in BINS:
    if mkey not in df.columns or fkey not in df.columns:
        print("%-8s  MISSING (%s / %s)" % (label, mkey in df.columns, fkey in df.columns))
        continue
    sub = df[["epoch", mkey, fkey]].dropna()
    if len(sub) == 0:
        print("%-8s  no paired rows" % label)
        continue
    windows = [
        ("pre-overfit (<=peak)", sub[sub["epoch"] <= PEAK_EPOCH]),
        ("early post (peak..2x)", sub[(sub["epoch"] > PEAK_EPOCH) & (sub["epoch"] <= 2 * PEAK_EPOCH)]),
        ("late (last 20%% eps)", sub[sub["epoch"] >= 0.8 * sub["epoch"].max()]),
    ]
    for wname, w in windows:
        if len(w) == 0:
            print("%-8s %-26s %6d %9s" % (label, wname, 0, "-"))
            continue
        d = w[mkey] - w[fkey]
        print("%-8s %-26s %6d %9.4f %9.4f %+9.4f %7.1f%%"
              % (label, wname, len(w), w[mkey].mean(), w[fkey].mean(), d.mean(),
                 100.0 * (d > 0).mean()))
    print()

# The single sharpest number: earliest epoch at which the paired delta goes negative and STAYS
# negative on a rolling basis, per bin.
print("First sustained sub-floor epoch (rolling-20 mean of delta < 0):")
for label, mkey, fkey in BINS:
    if mkey not in df.columns or fkey not in df.columns:
        continue
    sub = df[["epoch", mkey, fkey]].dropna().sort_values("epoch")
    if len(sub) < 25:
        print("  %-8s insufficient paired rows (%d)" % (label, len(sub)))
        continue
    roll = (sub[mkey] - sub[fkey]).rolling(20, min_periods=20).mean()
    neg = sub["epoch"][roll < 0]
    print("  %-8s %s" % (label, ("epoch %.0f" % neg.iloc[0]) if len(neg) else "never sustained sub-floor"))
