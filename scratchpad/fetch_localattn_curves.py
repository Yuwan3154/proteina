import json
import os

import matplotlib
import pandas as pd
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "kryst3154-massachusetts-institute-of-technology"
PROJECT = "protein_transformer_big_runs"
RUN_ID = "overfit_localattn"
OUT_DIR = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/curves"

KEYS = [
    "epoch",
    "trainer/global_step",
    "train/loss_epoch",
    "validation_loss/loss_epoch",
    "train/contact_map_loss_epoch",
    "validation_loss/contact_map_loss_epoch",
    "train/contact_precision_at_L_single_step_epoch",
    "validation_loss/contact_precision_at_L_single_step",
    "train/contact_precision_at_L_noisy_floor_epoch",
    "validation_loss/contact_precision_at_L_noisy_floor",
    "validation_sampling/contact_precision_at_L_mean",
    "validation_sampling/contact_precision_at_L_median",
    "validation_sampling/contact_f1_mean",
]

os.makedirs(OUT_DIR, exist_ok=True)
api = wandb.Api(timeout=120)
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
print("run:", run.name, "state:", run.state, "steps:", run.lastHistoryStep)

rows = list(run.scan_history(keys=None, page_size=2000))
df = pd.DataFrame(rows)
df = df.loc[:, ~df.columns.duplicated()]
print("history rows:", len(df), "cols:", len(df.columns))
print("ALL contact_precision columns present:")
for c in sorted(c for c in df.columns if "contact_precision" in c):
    print("   ", c, "  n_nonnull =", int(df[c].notna().sum()))
present = [k for k in KEYS if k in df.columns]
missing = [k for k in KEYS if k not in df.columns]
print("present:", present)
print("MISSING:", missing)

df.to_csv(f"{OUT_DIR}/{RUN_ID}_full_history.csv", index=False)


def series(key, xkey="epoch"):
    """Epoch-indexed, NaN-dropped (x, y) for one wandb key."""
    if key == xkey or key not in df.columns or xkey not in df.columns:
        return None, None
    sub = df[[xkey, key]].dropna()
    if len(sub) == 0:
        return None, None
    sub = sub.sort_values(xkey)
    return sub[xkey].values, sub[key].values


BLUE, ORANGE, GREY = "#2A6EBB", "#E8833A", "#9A9A9A"
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
fig.patch.set_facecolor("white")

ax = axes[0]
for key, color, label in [
    ("train/loss_epoch", BLUE, "train"),
    ("validation_loss/loss_epoch", ORANGE, "validation"),
]:
    x, y = series(key)
    if x is not None:
        ax.plot(x, y, color=color, lw=1.6, label=label)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("total loss (log)")
ax.set_title("Loss: train converged, validation diverging")
ax.legend(frameon=False)

ax = axes[1]
for key, color, style, label in [
    ("train/contact_precision_at_L_single_step_epoch", BLUE, "-", "train"),
    ("validation_loss/contact_precision_at_L_single_step", ORANGE, "-", "validation"),
    ("validation_loss/contact_precision_at_L_noisy_floor", GREY, "--", "noisy-input floor (val)"),
]:
    x, y = series(key)
    if x is None:
        continue
    # Raw trace is dominated by which t each logged batch happened to draw; the rolling
    # median is what carries the trend.
    ax.plot(x, y, color=color, lw=0.6, alpha=0.22)
    sm = pd.Series(y).rolling(25, min_periods=5, center=True).median().values
    ax.plot(x, sm, color=color, ls=style, lw=1.9, label=label)
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L")
ax.set_title("Single-step denoised precision@L")
ax.legend(frameon=False)

ax = axes[2]
for key, color, label in [
    ("validation_sampling/contact_precision_at_L_mean", BLUE, "mean"),
    ("validation_sampling/contact_precision_at_L_median", ORANGE, "median"),
]:
    x, y = series(key)
    if x is not None:
        ax.plot(x, y, color=color, lw=1.6, marker="o", ms=2.5, label=label)
ax.set_xlabel("epoch")
ax.set_ylabel("precision@L")
ax.set_title("Full-sampling precision@L (fixed 32 val chains)")
ax.legend(frameon=False)

for ax in axes:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/{RUN_ID}_curves.png", dpi=160, facecolor="white")
print("wrote", f"{OUT_DIR}/{RUN_ID}_curves.png")

# Tail statistics: is it still improving, and where did the best value occur?
report = {}
for key in present:
    x, y = series(key)
    if x is None or len(y) < 4:
        continue
    n = len(y)
    last_q = y[max(0, n - max(2, n // 5)) :]
    report[key] = {
        "n_points": int(n),
        "last_epoch": float(x[-1]),
        "final": float(y[-1]),
        "best": float(y.max()) if "precision" in key else float(y.min()),
        "best_epoch": float(x[int(y.argmax() if "precision" in key else y.argmin())]),
        "last20pct_mean": float(last_q.mean()),
        "last20pct_std": float(last_q.std()),
    }
with open(f"{OUT_DIR}/{RUN_ID}_convergence.json", "w") as fh:
    json.dump(report, fh, indent=2)
print(json.dumps(report, indent=2))
