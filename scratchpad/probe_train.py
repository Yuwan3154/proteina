"""Train the frozen-feature alignment probe and score it against the left-alignment floor.

Question: has tri's trunk ALREADY learned to realign the query onto the reference, given only a
clipped left-aligned relative position?

Three scorers, all trained the same way on the same cells, evaluated on held-out CHAINS:

  FLOOR   Embedding(2*max_rel+2 -> 1) on the clipped offset alone. This is the mandatory baseline:
          left-alignment is a decent prior, so beating chance proves nothing. A per-offset lookup
          matches the capacity the model itself has for representing left-alignment, so the
          comparison is fair rather than rigged in the probe's favour.
  PROBE   Linear(320 -> 1) on the frozen z[:, :L, L:] block.
  BOTH    the two concatenated, to show whether the trunk adds anything ON TOP of the offset
          rather than merely re-encoding it.

Metric is precision@Q per the user's instruction: with Q true alignments for a chain, score the
top-Q predicted cells. Q bounds what is achievable, so precision@Q is on a 0-1 scale where 1 is
attainable. Averaged over chains, never pooled over cells -- pooling would let long chains
dominate.

⚠️ The split is by CHAIN, not by cell. Cells within one chain are massively correlated (they share
a query, a reference and a trunk forward), so a cell-level split leaks and would inflate every
number.
"""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F


def precision_at_q(score, gt):
    """Fraction of the top-Q scored cells that are true, where Q = number of true rows."""
    q = int((gt.sum(dim=1) > 0).sum())
    if q == 0:
        return None
    flat_s = score.flatten()
    flat_g = gt.flatten().float()
    k = min(q, flat_s.numel())
    top = torch.topk(flat_s, k).indices
    return float(flat_g[top].sum() / k)


def load_samples(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "s*.pt"))):
        s = torch.load(p, map_location="cpu", weights_only=False)
        out.append(s)
    return out


def fit(samples, mode, n_off, dim, epochs, lr, dev, quiet=False):
    if mode == "floor":
        model = torch.nn.Embedding(n_off, 1)
        torch.nn.init.zeros_(model.weight)
    elif mode == "probe":
        model = torch.nn.Linear(dim, 1)
    else:
        model = torch.nn.Linear(dim + n_off, 1)
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        tot, nb = 0.0, 0
        for s in samples:
            gt = s["gt"].to(dev).float()
            if mode == "floor":
                logit = model(s["off"].to(dev)).squeeze(-1)
            elif mode == "probe":
                logit = model(s["feat"].to(dev).float()).squeeze(-1)
            else:
                oh = F.one_hot(s["off"].to(dev), n_off).float()
                logit = model(torch.cat([s["feat"].to(dev).float(), oh], dim=-1)).squeeze(-1)
            # Positives are ~1% of cells, so an unweighted BCE would learn the all-zero solution.
            pos = gt.sum().clamp(min=1.0)
            w = (gt.numel() - pos) / pos
            loss = F.binary_cross_entropy_with_logits(logit, gt, pos_weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        if not quiet:
            print(f"    [{mode}] epoch {ep + 1}/{epochs} loss={tot / max(nb, 1):.4f}", flush=True)
    return model


def evaluate(model, samples, mode, n_off, dev):
    vals = []
    with torch.no_grad():
        for s in samples:
            if mode == "floor":
                logit = model(s["off"].to(dev)).squeeze(-1)
            elif mode == "probe":
                logit = model(s["feat"].to(dev).float()).squeeze(-1)
            else:
                oh = F.one_hot(s["off"].to(dev), n_off).float()
                logit = model(torch.cat([s["feat"].to(dev).float(), oh], dim=-1)).squeeze(-1)
            p = precision_at_q(logit.cpu(), s["gt"])
            if p is not None:
                vals.append(p)
    t = torch.tensor(vals)
    return float(t.mean()), float(t.median()), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    samples = load_samples(args.data)
    assert samples, f"no samples in {args.data}"
    if args.dedupe:
        # Self-conditioning forwards the same batch twice per step, so the same (query, reference)
        # pair can be captured twice -- with DIFFERENT features (the second forward receives the
        # model's own prediction as contact_map_sc), so this is near-duplication, not duplication.
        # Keeping both gives those chains double weight; --dedupe keeps the first forward only.
        seen, kept = set(), []
        for x in samples:
            k = (x["query"], x["ref"])
            if k in seen:
                continue
            seen.add(k)
            kept.append(x)
        print(f"[dedupe] {len(samples)} forwards -> {len(kept)} distinct (query, ref) pairs")
        samples = kept
    dim = samples[0]["feat"].shape[-1]
    n_off = int(max(int(s["off"].max()) for s in samples)) + 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ⛔ Split by UNIQUE QUERY CHAIN, not by sample index. The same query recurs across samples
    # (the sampler can draw it more than once with a different reference), so an index split puts
    # the same chain on both sides and the probe can memorise it. A first attempt did exactly that
    # -- 31 chains overlapped -- and inflated the probe score.
    queries = sorted({s["query"] for s in samples})
    print(f"samples={len(samples)} unique_queries={len(queries)} dim={dim} n_off={n_off}")
    qs = torch.tensor([float(s["Q"]) for s in samples])
    ls = torch.tensor([float(s["L"]) for s in samples])
    cells = sum(int(x["L"]) * int(x["T"]) for x in samples)
    pos = sum(int(x["gt"].sum()) for x in samples)
    print(f"Q/L: mean={float((qs / ls).mean()):.1%}  Q median={int(qs.median())}")
    print(f"cells: {cells:,} total  positives: {pos:,} ({pos/max(cells,1):.2%})  "
          f"L median={int(ls.median())}  T median={int(torch.tensor([float(x['T']) for x in samples]).median())}")
    print(f"params: floor={n_off}  probe={dim+1}  both={dim+n_off+1}\n")

    # K-FOLD over chains. One split gave a single number with no error bar; folds give a spread, and
    # the spread is what says whether probe-minus-floor is bigger than run-to-run variation.
    K = args.folds
    folds = [{q for i, q in enumerate(queries) if i % K == k} for k in range(K)]
    per_mode = {m: [] for m in ("floor", "probe", "both")}
    per_train = {m: [] for m in ("floor", "probe", "both")}
    for k in range(K):
        test_q = folds[k]
        test = [s for s in samples if s["query"] in test_q]
        train = [s for s in samples if s["query"] not in test_q]
        overlap = len({s["query"] for s in train} & {s["query"] for s in test})
        if overlap != 0:
            print(f"FAIL: fold {k} leaks {overlap} chains", flush=True)
            return 3
        if not train or not test:
            print(f"FAIL: fold {k} empty", flush=True)
            return 4
        print(f"  --- fold {k + 1}/{K}: train={len(train)} test={len(test)} chains_overlap=0 ---", flush=True)
        for mode in ("floor", "probe", "both"):
            m = fit(train, mode, n_off, dim, args.epochs, args.lr, dev, quiet=True)
            # Evaluate on TRAIN as well as TEST: with a 320-parameter head on millions of cells
            # overfitting is unlikely, but that should be measured rather than asserted.
            tr_mean, _, tr_n = evaluate(m, train, mode, n_off, dev)
            mean, med, n = evaluate(m, test, mode, n_off, dev)
            per_mode[mode].append(mean)
            per_train[mode].append(tr_mean)
            print(f"      {mode:6s} train={tr_mean:.4f} (n={tr_n})  test={mean:.4f} "
                  f"median={med:.4f} (n={n})   gap={tr_mean - mean:+.4f}", flush=True)

    print("\n=== precision@Q, {}-fold over held-out chains ===".format(K))
    summ = {}
    for mode in ("floor", "probe", "both"):
        t = torch.tensor(per_mode[mode])
        tr = torch.tensor(per_train[mode])
        summ[mode] = (float(t.mean()), float(t.std()))
        print(f"  {mode:6s} TEST mean={t.mean():.4f} sd={t.std():.4f} | TRAIN mean={tr.mean():.4f} "
              f"| train-test gap={float(tr.mean() - t.mean()):+.4f}")
        print(f"         test folds={list(round(float(x),4) for x in t)}")
    d = summ["probe"][0] - summ["floor"][0]
    # Propagate both spreads rather than quoting the gap as if it were exact.
    sd = (summ["probe"][1] ** 2 + summ["floor"][1] ** 2) ** 0.5
    print(f"\n  probe - floor = {d:+.4f}  (combined sd {sd:.4f})")
    print("  ^ this difference, not the probe number, is the answer to "
          "'has the trunk already learned to realign'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
