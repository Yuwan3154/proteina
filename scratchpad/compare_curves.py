import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"
OUT = f"{CURVES}/localattn_vs_tri_comparison.png"

# The first two share data/val-split/effective-batch and ARE comparable. max512 is a retired
# reference on a DIFFERENT val split (8082 vs 4158) -- grey + dashed so it never reads as a peer.
RUNS = [
    ("localattn_full384", "local_attn max384", "#2A6EBB", "-", 2.0),
    ("tri_full384", "tri_mul max384", "#E8833A", "-", 2.0),
    ("localattn_full512_1gpu", "local_attn max512 (ref)", "#9A9A9A", "--", 1.4),
]
SMOOTH = 25  # rolling window used for the ~18x/epoch validation traces

# ⛔ tri's validation_sampling history has a MEASUREMENT DISCONTINUITY, not a training one.
# Until 2026-08-25 the tri arm sampled with NO topology conditioning and at a RANDOM redrawn
# length (contact_map_tri_30M.yaml never declared topology_cond, so
# _build_self_reference_topology returned None and the trainer fell through to the variable-length
# branch). Everything before the fix measured a different, much harder task. Its last pre-fix EMA
# checkpoint was epoch 40 (read off the checkpoint itself in job 21222196), so that is the
# boundary. Drawing one continuous line across it would show a ~5x jump that reads as the model
# improving when it is the metric changing.
TRI_FIX_EPOCH = 40.0

K_TRAIN_LOSS = "train/loss_epoch"
K_VAL_LOSS = "validation_loss/loss_epoch"
K_SS = "validation_loss/contact_precision_at_L_single_step"
K_FLOOR = "validation_loss/contact_precision_at_L_noisy_floor"
K_SAMP = "validation_sampling/contact_precision_at_L_mean"

data = {}
for run_id, _, _, _, _ in RUNS:
    df = pd.read_csv(f"{CURVES}/{run_id}_full_history.csv", low_memory=False)
    keep = [c for c in (["epoch", K_TRAIN_LOSS, K_VAL_LOSS, K_SS, K_FLOOR, K_SAMP]) if c in df.columns]
    data[run_id] = df[keep].apply(pd.to_numeric, errors="coerce")

fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))
fig.patch.set_facecolor("white")


def xy(df, key):
    sub = df[["epoch", key]].dropna().sort_values("epoch")
    return sub["epoch"].values, sub[key].values


ax = axes[0]
for run_id, label, color, ls, lw in RUNS:
    df = data[run_id]
    x, y = xy(df, K_TRAIN_LOSS)
    ax.plot(x, y, color=color, lw=lw, ls="-", label=f"{label} train")
    x, y = xy(df, K_VAL_LOSS)
    ys = pd.Series(y).rolling(SMOOTH, min_periods=5, center=True).median().values
    ax.plot(x, ys, color=color, lw=lw * 0.85, ls=":", label=f"{label} val")
ax.set_xlabel("epoch")
ax.set_yscale("log")
ax.set_ylabel("total loss (log)")
ax.set_title("Loss (solid = train, dotted = validation)", fontsize=11)
ax.legend(frameon=False, fontsize=7)

# Panel 2: the gap above the noisy-input floor. Row-wise, since signal and floor are
# logged in the same validation row and both move with whichever t that batch drew.
ax = axes[1]
for run_id, label, color, ls, lw in RUNS:
    df = data[run_id]
    sub = df[["epoch", K_SS, K_FLOOR]].dropna().sort_values("epoch")
    gap = (sub[K_SS] - sub[K_FLOOR]).rolling(SMOOTH, min_periods=5, center=True).median()
    ax.plot(sub["epoch"].values, gap.values, color=color, lw=lw, ls=ls, label=label)
ax.axhline(0.0, color="#9A9A9A", lw=1.0, ls=":")
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L above noisy floor")
ax.set_title("Single-step denoising: gap above noisy floor", fontsize=11)
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
for run_id, label, color, ls, lw in RUNS:
    df = data[run_id]
    x, y = xy(df, K_SAMP)
    if run_id == "tri_full384":
        pre, post = x < TRI_FIX_EPOCH, x >= TRI_FIX_EPOCH
        # Pre-fix points are kept rather than deleted -- hiding them would erase the reason the
        # comparison changed -- but drawn faint and labelled so they cannot be read as performance.
        ax.plot(x[pre], y[pre], color=color, lw=1.0, ls=":", alpha=0.35,
                label="tri_mul (pre-fix: UNCONDITIONED, random length)")
        ax.plot(x[post], y[post], color=color, lw=0.9, alpha=0.28)
        yp = pd.Series(y[post]).rolling(5, min_periods=2, center=True).median().values
        ax.plot(x[post], yp, color=color, lw=lw, label=label + " (conditioned)")
        ax.axvline(TRI_FIX_EPOCH, color=color, lw=1.0, ls="--", alpha=0.55)
        ax.annotate("conditioning fixed", xy=(TRI_FIX_EPOCH, ax.get_ylim()[1]),
                    xytext=(3, -8), textcoords="offset points",
                    fontsize=7, color=color, rotation=90, va="top")
        continue
    ax.plot(x, y, color=color, lw=0.9, ls=ls, alpha=0.28)
    ys = pd.Series(y).rolling(5, min_periods=2, center=True).median().values
    ax.plot(x, ys, color=color, lw=lw, ls=ls, label=label)
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L (full sampling)")
ax.set_title("Full sampling: precision@L", fontsize=11)
ax.legend(frameon=False, fontsize=8)

tri_max_ep = float(data["tri_full384"]["epoch"].dropna().max())
for ax in (axes[1], axes[2]):
    ax.axvline(tri_max_ep, color="#B0B0B0", lw=1.0, ls="-.", zorder=0)
    ax.annotate(f"tri stops here (ep {tri_max_ep:.0f})\ncompare LEFT of this line",
                xy=(tri_max_ep, 0.02), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points",
                fontsize=7.5, color="#777777", va="bottom", ha="left")

for ax in axes:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.text(
    0.5, -0.03,
    "BLUE vs ORANGE is a CONTROLLED comparison: identical max384 data, identical 4158-chain validation split, "
    "identical effective batch 32, identical 13,152 chains/epoch.\nGREY (max512) is REFERENCE ONLY -- a retired run on a "
    "DIFFERENT 8082-chain validation split, so its absolute values are not comparable to the other two.\n"
    f"Validation traces are {SMOOTH}-point rolling medians (validation logs ~18x/epoch).   "
    "\nlocal_attn max384 carries two mid-run discontinuities: 1->2 GPU at ~epoch 3 and an LR-horizon correction "
    "(822k->411k cosine steps) at ~epoch 120; effective batch stayed 32 throughout.",
    ha="center", va="top", fontsize=8.0, color="#555555",
)
fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
print("wrote", OUT)
for run_id, label, _, _, _ in RUNS:
    df = data[run_id]
    print(f"{label}: max epoch {df['epoch'].max():.0f}")
