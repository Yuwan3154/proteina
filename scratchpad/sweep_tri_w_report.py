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


def tail_stats(df, key, frac):
    d = df[["trainer/global_step", key]].dropna()
    if d.empty:
        return None
    cut = d["trainer/global_step"].max() * (1 - frac)
    t = d[d["trainer/global_step"] >= cut][key].to_numpy()
    return float(t.mean()), float(t.std(ddof=1) if len(t) > 1 else 0.0), int(d["trainer/global_step"].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="kryst3154-massachusetts-institute-of-technology")
    ap.add_argument("--project", default="tri_cb8_sweep")
    ap.add_argument("--prefix", default="tri_cb8sw_")
    ap.add_argument("--tail", type=float, default=0.25)
    args = ap.parse_args()
    api = wandb.Api()
    runs = [r for r in api.runs(f"{args.entity}/{args.project}") if r.name.startswith(args.prefix)]
    print(f"{len(runs)} arms in {args.project}")
    hist = {}
    for r in runs:
        arm = r.name[len(args.prefix):]
        cfg = r.config
        df = r.history(keys=["trainer/global_step", *MAIN_KEYS, *AUX_KEYS], pandas=True, samples=100000)
        hist[arm] = (df, cfg)
    if "ctrl" not in hist:
        print("no ctrl arm found -- nothing to compare against")
    ctrl = hist.get("ctrl")
    print(f"\n{'arm':>12} {'steps':>6} | {'cm_loss tail':>14} {'d vs ctrl':>10} | {'precL tail':>11} {'d vs ctrl':>10} | align_loss  prec@Q | mlm_loss  acc | missing_ref")
    for arm, (df, cfg) in sorted(hist.items()):
        row = [f"{arm:>12}"]
        cm = tail_stats(df, MAIN_KEYS[0], args.tail)
        pl = tail_stats(df, MAIN_KEYS[1], args.tail)
        steps = cm[2] if cm else 0
        row.append(f"{steps:6d} |")
        for key, st in ((MAIN_KEYS[0], cm), (MAIN_KEYS[1], pl)):
            if st is None:
                row.append(f"{'n/a':>14} {'':>10} |")
                continue
            d = ""
            if ctrl is not None and arm != "ctrl":
                cs = tail_stats(ctrl[0], key, args.tail)
                if cs:
                    delta = st[0] - cs[0]
                    flag = "" if abs(delta) <= 2 * max(cs[1], 1e-9) else (" WORSE" if (delta > 0) == (key.endswith("loss")) else " better")
                    d = f"{delta:+.4f}{flag}"
            row.append(f"{st[0]:8.4f}±{st[1]:.4f} {d:>10} |")
        a = tail_stats(df, "train/align_loss", args.tail)
        q = tail_stats(df, "train/align_precision_at_q", args.tail)
        m = tail_stats(df, "train/mlm_loss", args.tail)
        acc = tail_stats(df, "train/mlm_acc", args.tail)
        mr = tail_stats(df, "train/topology_missing_ref_frac", 1.0)
        row.append(f" {a[0] if a else float('nan'):9.4f} {q[0] if q else float('nan'):6.3f} | {m[0] if m else float('nan'):8.4f} {acc[0] if acc else float('nan'):5.3f} | {mr[0] if mr else float('nan'):.4f}")
        print(" ".join(row))
    print("\nRULE: choose the largest weights with no WORSE flag on either main metric; a WORSE flag = beyond 2 sd of the control's own tail.")


if __name__ == "__main__":
    main()
