"""T8 alignment-bias study, step 2 (local): what the current ConFind tri's alignment head predicts vs the TM-align target.

Per residue, the head's choice is argmax over (T element logits, "none" logit), exactly the softmax-CE classes it was
trained on. Reported for the LAST network call of the sampling run (the deployed state) and the FIRST (pure noise):
  * residue accuracy (incl. none); precision@Q of the head and of the position-only baseline (nearest element midpoint,
    the same baseline the trainer logs);
  * signed errors on residues that truly align and are predicted to an element: element-index error and the residue
    offset between predicted and true element midpoints;
  * how often a prediction equals the position-only choice, split by correct / wrong (does it just read position?);
  * accuracy vs relative position i/L; "none" over/under-calling; accuracy vs element type, length and template TM.
Figures: example heatmaps (best / median / worst by last-call accuracy) and the aggregate panels; y axes on the
1/2/2.5/5 x 10^k ladder.
Usage: python t8_align_bias_plot.py COLLECT.npz OUT_DIR
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H_TOK = range(2, 23)   # vocab-44 alphabet (types helix, strand): helix tokens 2..22, strand 23..43
BLUE, GREY, ORANGE = "#2a78d6", "#8a8f98", "#d2691e"


def nice_axis(lo, hi, target=5):
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    s = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag * 1.0000001), 10 * mag) if raw > 0 else 1.0
    a = math.floor(lo / s + 1e-9) * s
    b = math.ceil(hi / s - 1e-9) * s
    return a, (b if b > a else a + s), s


def ticks(ax, vals):
    lo, hi, s = nice_axis(min(vals), max(vals))
    n = int(round((hi - lo) / s))
    ax.set_ylim(lo, hi)
    ax.set_yticks([lo + k * s for k in range(n + 1)])


def classes(a, n):
    """argmax over [elements..., none]; none = T."""
    return np.concatenate([a, n[:, None]], axis=1).argmax(1)


def prec_at_q(score, tgt, T):
    A = np.zeros_like(score)
    al = tgt >= 0
    A[np.where(al)[0], tgt[al]] = 1
    q = int(al.sum())
    if q == 0:
        return None
    top = np.argsort(-score.ravel(), kind="stable")[:q]
    return float(A.ravel()[top].mean())


def analyse(recs, which):
    out = {k: [] for k in ("acc", "pq", "pq_pos", "none_true", "none_pred", "tm", "L")}
    err_e, err_res, rel_ok, rel_n = [], [], np.zeros(10), np.zeros(10)
    agree_ok, agree_bad = [], []
    by_type = {"helix": [0, 0], "strand": [0, 0]}
    for r in recs:
        a, n, tgt, T, L = r[f"a_{which}"], r[f"n_{which}"], r["target"], r["T"], r["L"]
        pos = r["he_pos_raw"]
        cls = classes(a, n)
        truth = np.where(tgt >= 0, tgt, T)
        ok = cls == truth
        out["acc"].append(ok.mean())
        out["pq"].append(prec_at_q(a, tgt, T))
        nearest = np.abs(np.arange(L)[:, None] - pos[None, :]).argmin(1)
        out["pq_pos"].append(prec_at_q(-np.abs(np.arange(L)[:, None] - pos[None, :]).astype(float), tgt, T))
        out["none_true"].append(float((tgt < 0).mean()))
        out["none_pred"].append(float((cls == T).mean()))
        out["tm"].append(r["template_tm"])
        out["L"].append(L)
        m = (tgt >= 0) & (cls < T)
        err_e += list(cls[m] - tgt[m])
        err_res += list(pos[cls[m]] - pos[tgt[m]])
        b = np.minimum((np.arange(L) * 10) // L, 9)
        np.add.at(rel_ok, b, ok)
        np.add.at(rel_n, b, 1)
        agree = cls[m] == nearest[m]
        agree_ok += list(agree[ok[m]])
        agree_bad += list(agree[~ok[m]])
        for e_true, o in zip(tgt[tgt >= 0], ok[tgt >= 0]):
            key = "helix" if int(r["he_tokens"][e_true]) in H_TOK else "strand"
            by_type[key][0] += int(o)
            by_type[key][1] += 1
    out = {k: np.array([x for x in v if x is not None], dtype=float) for k, v in out.items()}
    return out, np.array(err_e), np.array(err_res), rel_ok / np.maximum(rel_n, 1), np.array(agree_ok), np.array(agree_bad), by_type


def main(collect, out_dir):
    recs = list(np.load(collect, allow_pickle=True)["recs"])
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"samples {len(recs)}; network calls per run {sorted({r['n_calls'] for r in recs})}"]
    res = {}
    for which in ("last", "first"):
        o, ee, er, rel, ag_ok, ag_bad, bt = analyse(recs, which)
        res[which] = (o, ee, er, rel, ag_ok, ag_bad, bt)
        lines += [f"[{which} call] residue accuracy (incl. none) median {np.median(o['acc']):.3f} mean {o['acc'].mean():.3f}",
                  f"  precision@Q head median {np.median(o['pq']):.3f} (mean {o['pq'].mean():.3f}) vs position-only baseline "
                  f"{np.median(o['pq_pos']):.3f} (mean {o['pq_pos'].mean():.3f}); "
                  f"head > baseline on {int((o['pq'] > o['pq_pos']).sum())}/{len(o['pq'])} chains",
                  f"  'none': true fraction mean {o['none_true'].mean():.3f}, predicted {o['none_pred'].mean():.3f}",
                  f"  element-index error on aligned+predicted residues: exact {np.mean(ee == 0):.3f}, |err|=1 {np.mean(np.abs(ee) == 1):.3f}, "
                  f"mean signed {ee.mean():+.3f} (n={len(ee)})",
                  f"  residue offset (pred - true midpoint): median {np.median(er):+.1f}, mean {er.mean():+.2f}, p5 {np.percentile(er, 5):+.1f}, p95 {np.percentile(er, 95):+.1f}",
                  f"  prediction == position-only choice: when correct {ag_ok.mean():.3f}, when wrong {ag_bad.mean():.3f}",
                  f"  accuracy by true element type: " + ", ".join(f"{k} {v[0] / max(v[1], 1):.3f} (n={v[1]})" for k, v in bt.items()),
                  "  accuracy by relative position decile: " + " ".join(f"{x:.2f}" for x in rel)]
    o = res["last"][0]
    q = np.quantile(o["tm"], [0, 1 / 3, 2 / 3, 1])
    for j, (lo, hi) in enumerate(zip(q[:-1], q[1:])):
        m = (o["tm"] >= lo) & ((o["tm"] < hi) if j < 2 else (o["tm"] <= hi))  # half-open bins, last one closed
        lines.append(f"[last] template TM {lo:.2f}-{hi:.2f}: accuracy median {np.median(o['acc'][m]):.3f} (n={int(m.sum())})")
    open(os.path.join(out_dir, "align_bias_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # aggregate figure
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    o, ee, er, rel, ag_ok, ag_bad, bt = res["last"]
    of = res["first"][0]
    ax[0, 0].scatter(o["pq_pos"], o["pq"], s=10, color=BLUE, label="last call")
    ax[0, 0].scatter(of["pq_pos"], of["pq"], s=10, color=GREY, label="first call (noise)")
    ax[0, 0].plot([0, 1], [0, 1], color="#cccccc", lw=1)
    ax[0, 0].set_xlabel("position-only baseline precision@Q")
    ax[0, 0].set_ylabel("head precision@Q")
    ax[0, 0].set_title("per chain: head vs position-only")
    ax[0, 0].legend(frameon=False, fontsize=9)
    vals, cnt = np.unique(np.clip(ee, -5, 5), return_counts=True)
    ax[0, 1].bar(vals, cnt / cnt.sum(), color=BLUE)
    ax[0, 1].set_xlabel("predicted - true element index (clipped at +/-5)")
    ax[0, 1].set_ylabel("fraction of residues")
    ax[0, 1].set_title("element-index error (aligned residues)")
    ticks(ax[0, 1], list(cnt / cnt.sum()) + [0])
    ax[0, 2].hist(np.clip(er, -40, 40), bins=81, color=BLUE)
    ax[0, 2].set_xlabel("predicted - true element midpoint (residues, clipped at +/-40)")
    ax[0, 2].set_ylabel("residues")
    ax[0, 2].set_title("signed offset")
    ax[1, 0].bar(np.arange(10) + 0.5, rel, width=0.9, color=BLUE)
    ax[1, 0].set_xlabel("relative position along the chain (deciles of i / L)")
    ax[1, 0].set_ylabel("residue accuracy")
    ax[1, 0].set_title("accuracy along the chain (last call)")
    ticks(ax[1, 0], list(rel) + [0])
    ax[1, 1].scatter(o["none_true"], o["none_pred"], s=10, color=BLUE)
    ax[1, 1].plot([0, 1], [0, 1], color="#cccccc", lw=1)
    ax[1, 1].set_xlabel("true 'none' fraction per chain")
    ax[1, 1].set_ylabel("predicted 'none' fraction")
    ax[1, 1].set_title("'none' calibration")
    ax[1, 2].scatter(o["tm"], o["acc"], s=10, color=BLUE)
    ax[1, 2].set_xlabel("template TM to native")
    ax[1, 2].set_ylabel("residue accuracy (last call)")
    ax[1, 2].set_title("accuracy vs template quality")
    ticks(ax[1, 2], list(o["acc"]) + [0])
    for a in ax.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "align_bias_aggregate.png"), dpi=140)

    # example heatmaps: best / median / worst by last-call accuracy, 2 each
    order = np.argsort([classes(r["a_last"], r["n_last"]).__eq__(np.where(r["target"] >= 0, r["target"], r["T"])).mean() for r in recs])
    picks = [order[-1], order[-2], order[len(order) // 2], order[len(order) // 2 + 1], order[0], order[1]]
    fig, ax = plt.subplots(len(picks), 2, figsize=(12, 3.2 * len(picks)))
    for row, i in enumerate(picks):
        r = recs[i]
        T, L = r["T"], r["L"]
        tgt1h = np.zeros((L, T + 1))
        tgt1h[np.arange(L), np.where(r["target"] >= 0, r["target"], T)] = 1
        z = np.concatenate([r["a_last"], r["n_last"][:, None]], 1)
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        for c, (m, title) in enumerate(((tgt1h, "TM-align target"), (p, "head softmax, last call"))):
            ax[row, c].imshow(m.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            ax[row, c].set_title(f"{r['stem']} (template TM {r['template_tm']:.2f}): {title}", fontsize=9)
            ax[row, c].set_ylabel("element (top = 0; last row = none)", fontsize=8)
            ax[row, c].set_xlabel("query residue", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "align_bias_examples.png"), dpi=120)
    print(f"[plot] -> {out_dir}")


if __name__ == "__main__":
    assert len(sys.argv) == 3, __doc__
    main(*sys.argv[1:])
