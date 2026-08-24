"""Is tri_mul's single-step advantage concentrated in the LOW-NOISE regime?

Convention established from model_trainer_base.py:514,
    contact_map_pred = contact_map_t + (1 - t) * nn_pred
i.e. rectified-flow: t=1 is CLEAN data, t=0 is NOISE. So with T_BIN_EDGES = (1/3, 2/3):
    tlow  = t < 1/3   -> HIGH noise
    tmid  = 1/3..2/3
    thigh = t >= 2/3  -> LOW noise
Full sampling integrates from t=0 upward, so the HIGH-noise (tlow) end sets the fold.

Raw precision is NOT comparable across bins -- the noisy-input floor rises steeply as noise
falls. Only the GAP (model - floor), binned on the SAME t, is meaningful. The trainer logs a
per-bin floor for exactly this reason.
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
OUT = f"{CURVES}/tbin_analysis.png"
BINS = [("tlow", "t<1/3  (HIGH noise)"), ("tmid", "t 1/3-2/3"), ("thigh", "t>=2/3  (LOW noise)")]
RUNS = [("localattn_full384", "local_attn max384", "#2A6EBB"),
        ("tri_full384", "tri_mul max384", "#E8833A")]
MATCH_EP, WIN = 38, 4.0
SIG = "validation_loss/contact_precision_at_L_single_step_{b}"
FLR = "validation_loss/contact_precision_at_L_noisy_floor_{b}"

data = {r: pd.read_csv(f"{CURVES}/{r}_full_history.csv", low_memory=False) for r, _, _ in RUNS}

print("=== key availability (non-null counts) ===")
for run, lab, _ in RUNS:
    df = data[run]
    row = []
    for b, _ in BINS:
        for tmpl, nm in ((SIG, "sig"), (FLR, "flr")):
            k = tmpl.format(b=b)
            row.append(f"{b}.{nm}={int(df[k].notna().sum()) if k in df.columns else 'ABSENT'}")
    print(f"  {lab:20s} " + "  ".join(row))

def binned(run, b):
    df = data[run]
    ks, kf = SIG.format(b=b), FLR.format(b=b)
    if ks not in df.columns or kf not in df.columns:
        return None
    sub = df[["epoch", ks, kf]].apply(pd.to_numeric, errors="coerce").dropna()
    sub["gap"] = sub[ks] - sub[kf]
    return sub.sort_values("epoch")

print(f"\n=== PER-BIN GAP (model - floor) at MATCHED epoch {MATCH_EP} +/-{WIN:.0f} ===")
print(f"  {'bin':24s} {'local_attn':>12s} {'tri_mul':>12s} {'tri advantage':>15s}")
rows = {}
for b, blab in BINS:
    vals = {}
    for run, lab, _ in RUNS:
        sub = binned(run, b)
        if sub is None:
            vals[run] = float("nan"); continue
        sel = sub[(sub["epoch"] >= MATCH_EP - WIN) & (sub["epoch"] <= MATCH_EP + WIN)]
        vals[run] = float(sel["gap"].median()) if len(sel) else float("nan")
    la, tr = vals["localattn_full384"], vals["tri_full384"]
    rows[b] = (la, tr)
    print(f"  {blab:24s} {la:>+12.4f} {tr:>+12.4f} {tr - la:>+15.4f}")

print(f"\n=== raw floor level per bin (confirms the noise ordering) at ep {MATCH_EP} ===")
for b, blab in BINS:
    sub = binned("tri_full384", b)
    if sub is None: continue
    sel = sub[(sub["epoch"] >= MATCH_EP - WIN) & (sub["epoch"] <= MATCH_EP + WIN)]
    if len(sel):
        print(f"  {blab:24s} floor={float(sel[FLR.format(b=b)].median()):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
fig.patch.set_facecolor("white")
for ax, (b, blab) in zip(axes, BINS):
    for run, lab, color in RUNS:
        sub = binned(run, b)
        if sub is None: continue
        sm = sub["gap"].rolling(25, min_periods=5, center=True).median()
        ax.plot(sub["epoch"].values, sm.values, color=color, lw=1.9, label=lab)
    ax.axvline(MATCH_EP, color="#B0B0B0", lw=1.0, ls="-.", zorder=0)
    ax.axhline(0.0, color="#9A9A9A", lw=1.0, ls=":")
    ax.set_title(blab, fontsize=11)
    ax.set_xlabel("epoch")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axes[0].set_ylabel("precision@L above the SAME-t noisy floor")
axes[0].legend(frameon=False, fontsize=9)
fig.text(0.5, -0.04,
         "Gap = model - noisy-input floor, both measured at the SAME t, so bins are comparable to each other.\n"
         "t=1 is CLEAN, t=0 is NOISE (pred = x_t + (1-t)*v). Full sampling integrates from t=0, so the LEFT panel "
         "(high noise) is what governs generation quality.\n"
         "Dash-dot line = epoch 38, the matched-epoch comparison point (tri_mul's current maximum).",
         ha="center", va="top", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
print("\nwrote", OUT)
