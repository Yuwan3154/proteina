"""Report the auxiliary-weight sweep for tri_cb8synth from wandb (project tri_cb8_sweep).

Pre-registered rule: pick the LARGEST weights whose MAIN-objective trajectory (train/contact_map_loss
and train/contact_precision_at_L_single_step) is not worse than the w=0 control over the last
`--tail` fraction of common steps, beyond the control's own step-to-step noise (2 sd of its tail).
Arms whose aux losses do not decrease are also flagged: a head that does not learn is not doing
anything at that weight.

usage: python sweep_tri_w_report.py [--entity E] [--project tri_cb8_sweep] [--prefix tri_cb8sw_] [--tail 0.25]
"""

import argparse

import numpy as np
import wandb

MAIN_KEYS = ("train/contact_map_loss", "train/contact_precision_at_L_single_step")
AUX_KEYS = ("train/align_loss", "train/align_precision_at_q", "train/mlm_loss", "train/mlm_acc",
            "train/topology_missing_ref_frac", "validation_loss/loss")


def resolve(df, key):
    """Lightning's on_step/on_epoch logging renames `train/x` to `train/x_step` and `train/x_epoch`
    in wandb, so the exact name is usually ABSENT. Prefer the per-step series, then the epoch one,
    then the bare name; return None when the metric never appeared (a missing column must not raise
    -- that was the KeyError this pre-flight caught)."""
    for cand in (f"{key}_step", key, f"{key}_epoch"):
        if cand in df.columns:
            return cand
    return None


def tail_stats(df, key, frac, upto=None):
    """Tail statistics of `key`. `upto` caps the step range so arms that ran different numbers of
    steps (a preempted arm restarts at 0) are compared over the SAME window -- otherwise the control
    could be read at step 800 against an arm at 1500 and the difference would be schedule, not weight."""
    col = resolve(df, key)
    if col is None or "trainer/global_step" not in df.columns:
        return None
    d = df[["trainer/global_step", col]].dropna()
    if upto is not None:
        d = d[d["trainer/global_step"] <= upto]
    if d.empty:
        return None
    hi = d["trainer/global_step"].max()
    t = d[d["trainer/global_step"] >= hi * (1 - frac)][col].to_numpy()
    return float(t.mean()), float(t.std(ddof=1) if len(t) > 1 else 0.0), int(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="kryst3154-massachusetts-institute-of-technology")
    ap.add_argument("--project", default="tri_cb8_sweep")
    ap.add_argument("--prefix", default="tri_cb8sw_")
    ap.add_argument("--tail", type=float, default=0.25)
    ap.add_argument("--exclude", default="smoke", help="comma-separated arm names to ignore (default: the 20-step smoke)")
    args = ap.parse_args()
    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity}/{args.project}") if r.name.startswith(args.prefix)]
    skip = set(args.exclude.split(",")) if args.exclude else set()
    print(f"{len(runs)} runs in {args.project} (excluding arms: {sorted(skip) or 'none'})")
    hist = {}
    for r in runs:
        arm = r.name[len(args.prefix):]
        if arm in skip:
            continue
        # no keys= filter: wandb drops every row lacking ANY requested key, and the _step/_epoch
        # renaming means the bare names are usually absent entirely
        df = r.history(pandas=True, samples=100000)
        n = int(df["trainer/global_step"].max()) if "trainer/global_step" in df.columns and not df["trainer/global_step"].dropna().empty else -1
        # ⛔ An arm can have SEVERAL wandb runs (each failed/preempted attempt made one). Keep the
        # one that got furthest, or a 19-step crashed attempt silently becomes "the arm" and drags
        # the common comparison window down with it.
        if arm in hist and hist[arm][2] >= n:
            print(f"  {r.name}: {n} steps -- superseded by a longer run of the same arm, skipped")
            continue
        if arm in hist:
            print(f"  {r.name}: {n} steps -- replaces a shorter run of the same arm")
        hist[arm] = (df, r.config, n)
    hist = {k: (v[0], v[1]) for k, v in hist.items()}
    if "ctrl" not in hist:
        print("no ctrl arm found -- nothing to compare against")
    ctrl = hist.get("ctrl")
    # compare every arm over the window they all reached
    common = min((int(df["trainer/global_step"].max()) for df, _ in hist.values()
                  if "trainer/global_step" in df.columns and not df["trainer/global_step"].dropna().empty),
                 default=None)
    print(f"common step window: 0-{common} (arms are compared only where ALL of them have data)")
    print(f"\n{'arm':>12} {'steps':>6} | {'cm_loss tail':>14} {'d vs ctrl':>10} | {'precL tail':>11} {'d vs ctrl':>10} | align_loss  prec@Q | mlm_loss  acc | missing_ref")
    for arm, (df, cfg) in sorted(hist.items()):
        row = [f"{arm:>12}"]
        cm = tail_stats(df, MAIN_KEYS[0], args.tail, common)
        pl = tail_stats(df, MAIN_KEYS[1], args.tail, common)
        steps = cm[2] if cm else 0
        row.append(f"{steps:6d} |")
        for key, st in ((MAIN_KEYS[0], cm), (MAIN_KEYS[1], pl)):
            if st is None:
                row.append(f"{'n/a':>14} {'':>10} |")
                continue
            d = ""
            if ctrl is not None and arm != "ctrl":
                cs = tail_stats(ctrl[0], key, args.tail, common)
                if cs:
                    delta = st[0] - cs[0]
                    flag = "" if abs(delta) <= 2 * max(cs[1], 1e-9) else (" WORSE" if (delta > 0) == (key.endswith("loss")) else " better")
                    d = f"{delta:+.4f}{flag}"
            row.append(f"{st[0]:8.4f}±{st[1]:.4f} {d:>10} |")
        a = tail_stats(df, "train/align_loss", args.tail, common)
        q = tail_stats(df, "train/align_precision_at_q", args.tail, common)
        m = tail_stats(df, "train/mlm_loss", args.tail, common)
        acc = tail_stats(df, "train/mlm_acc", args.tail, common)
        mr = tail_stats(df, "train/topology_missing_ref_frac", 1.0)
        row.append(f" {a[0] if a else float('nan'):9.4f} {q[0] if q else float('nan'):6.3f} | {m[0] if m else float('nan'):8.4f} {acc[0] if acc else float('nan'):5.3f} | {mr[0] if mr else float('nan'):.4f}")
        print(" ".join(row))
    print("\nRULE: choose the largest weights with no WORSE flag on either main metric; a WORSE flag = beyond 2 sd of the control's own tail.")


if __name__ == "__main__":
    main()
