import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
OUT = f"{CURVES}/localattn_vs_tri_comparison.png"

RUNS = [
    ("localattn_full512_1gpu", "local_attn  (max512, 1x L40S)", "#2A6EBB"),
    ("tri_full384", "tri_mul  (max384, 2x RTX PRO 6000)", "#E8833A"),
]
SMOOTH = 25  # rolling window used for the ~18x/epoch validation traces

K_TRAIN_LOSS = "train/loss_epoch"
K_VAL_LOSS = "validation_loss/loss_epoch"
K_SS = "validation_loss/contact_precision_at_L_single_step"
K_FLOOR = "validation_loss/contact_precision_at_L_noisy_floor"
K_SAMP = "validation_sampling/contact_precision_at_L_mean"

data = {}
for run_id, _, _ in RUNS:
    df = pd.read_csv(f"{CURVES}/{run_id}_full_history.csv", low_memory=False)
    keep = [c for c in (["epoch", K_TRAIN_LOSS, K_VAL_LOSS, K_SS, K_FLOOR, K_SAMP]) if c in df.columns]
    data[run_id] = df[keep].apply(pd.to_numeric, errors="coerce")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
fig.patch.set_facecolor("white")


def xy(df, key):
    sub = df[["epoch", key]].dropna().sort_values("epoch")
    return sub["epoch"].values, sub[key].values


ax = axes[0]
for run_id, label, color in RUNS:
    df = data[run_id]
    x, y = xy(df, K_TRAIN_LOSS)
    ax.plot(x, y, color=color, lw=1.9, ls="-", label=f"{label.split('  ')[0]} train")
    x, y = xy(df, K_VAL_LOSS)
    ys = pd.Series(y).rolling(SMOOTH, min_periods=5, center=True).median().values
    ax.plot(x, ys, color=color, lw=1.6, ls="--", label=f"{label.split('  ')[0]} val")
ax.set_xlabel("epoch")
ax.set_ylabel("total loss")
ax.set_title("No train/val divergence in either arm", fontsize=11)
ax.legend(frameon=False, fontsize=8)

# Panel 2: the gap above the noisy-input floor. Row-wise, since signal and floor are
# logged in the same validation row and both move with whichever t that batch drew.
ax = axes[1]
for run_id, label, color in RUNS:
    df = data[run_id]
    sub = df[["epoch", K_SS, K_FLOOR]].dropna().sort_values("epoch")
    gap = (sub[K_SS] - sub[K_FLOOR]).rolling(SMOOTH, min_periods=5, center=True).median()
    ax.plot(sub["epoch"].values, gap.values, color=color, lw=1.9, label=label.split("  ")[0])
ax.axhline(0.0, color="#9A9A9A", lw=1.0, ls=":")
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L above noisy floor")
ax.set_title("Single-step denoising: tri_mul is far stronger", fontsize=11)
ax.legend(frameon=False, fontsize=9)

ax = axes[2]
for run_id, label, color in RUNS:
    df = data[run_id]
    x, y = xy(df, K_SAMP)
    ax.plot(x, y, color=color, lw=1.0, alpha=0.30)
    ys = pd.Series(y).rolling(5, min_periods=2, center=True).median().values
    ax.plot(x, ys, color=color, lw=2.0, label=label.split("  ")[0])
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L (full sampling)")
ax.set_title("Full sampling: local_attn rising, tri_mul flat", fontsize=11)
ax.legend(frameon=False, fontsize=9)

for ax in axes:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.text(
    0.5, -0.02,
    "NOT a controlled head-to-head: different datasets and validation splits "
    "(tri_mul = max384, val 4158 chains, batch 1 x accum 16;  local_attn = max512, val 8082 chains, batch 4 x accum 8). "
    f"Validation traces are {SMOOTH}-point rolling medians (validation logs ~18x per epoch).",
    ha="center", va="top", fontsize=8.5, color="#555555",
)
fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
for run_id, label, _ in RUNS:
    df = data[run_id]
    print(f"{label}: max epoch {df['epoch'].max():.0f}")
