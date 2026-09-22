"""CPU-only checks for the tri_full384 CA-feature ablation (no GPU, no model build, no dataloader).

1. The LIVE topology index's standardisation constants for min_ca_dist / mean_ca_dist
   (the plan's 19.83+-12.51 / 25.08+-12.56 come from an older build, job 20612782).
2. The 71,950 EMA checkpoint's cell_in weight shape and per-column norms (which columns are 4, 5).
3. Numeric proof that (a) zeroing the standardised input channels, (b) zeroing the weight columns and
   (c) imputing the raw value with the index mean give BIT-IDENTICAL cell_in outputs, fp32 and bf16.
4. The mean STANDARDISED value of each CA channel over the reference-reference cells the model sees
   (each row's block truncated to max_topology_he_len, uniform over rows) -- how far "0" is from the
   input distribution's own mean.

Run with PYTHONPATH=<proteina_cmhier checkout> so PAIR_FEATURE_NAMES comes from commit 30e4ec8.
"""

import argparse

import torch

from proteinfoundation.datasets.sse_topology import (
    N_PAIR_FEATURES,
    PAIR_FEATURE_NAMES,
    STRUCTURAL_PAIR_FEATURES,
)

CA_FEATURES = ("min_ca_dist", "mean_ca_dist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_he", type=int, default=64)  # = max_topology_he_len in the dataset + nn config
    args = ap.parse_args()

    idx = torch.load(args.index, map_location="cpu", weights_only=False, mmap=True)
    print("pair_feature_names      ", list(idx["pair_feature_names"]))
    print("structural_feature_names", list(idx["structural_feature_names"]))
    assert list(idx["pair_feature_names"]) == list(PAIR_FEATURE_NAMES)
    assert list(idx["structural_feature_names"]) == list(STRUCTURAL_PAIR_FEATURES)
    mean, std = idx["pair_feature_mean"].float(), idx["pair_feature_std"].float().clamp(min=1e-6)
    for n in PAIR_FEATURE_NAMES:
        k = PAIR_FEATURE_NAMES.index(n)
        print(f"  {n:<26s} mean={float(mean[k]):10.4f} std={float(std[k]):10.4f}")
    print("stats_from:", idx.get("stats_from", "(own)"))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    print("global_step:", ck.get("global_step"), "epoch:", ck.get("epoch"))
    keys = [k for k in sd if "cell_in" in k]
    print("cell_in keys:", keys)
    w = sd[[k for k in keys if k.endswith("weight")][0]].float()
    b = sd[[k for k in keys if k.endswith("bias")][0]].float()
    print("cell_in.weight", tuple(w.shape), "bias", tuple(b.shape))
    assert w.shape[1] == 2 + N_PAIR_FEATURES, "pair_ref_features != 'both'?"
    cols = [2 + PAIR_FEATURE_NAMES.index(n) for n in CA_FEATURES]
    names = ["cm_t", "cm_sc"] + list(PAIR_FEATURE_NAMES)
    for c, nm in enumerate(names):
        print(f"  col {c:2d} {nm:<26s} |W_col|={float(w[:, c].norm()):.4f}")
    print("CA columns:", cols)

    # (3) bit-identity of the three removal options on real-scale inputs
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, w.shape[1], generator=g)
    lin = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    xa = x.clone()
    xa[:, cols] = 0.0
    # (c) as the transform would compute it: only the CA channels change, raw := index mean, then (raw - mean) / std
    kc = [c - 2 for c in cols]
    xc = x.clone()
    xc[:, cols] = ((mean[kc] - mean[kc]) / std[kc]).expand(x.shape[0], -1)
    lin_b = torch.nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        lin_b.weight.copy_(w)
        lin_b.weight[:, cols] = 0.0
        lin_b.bias.copy_(b)
    with torch.no_grad():
        ya, yb, yc = lin(xa), lin_b(x), lin(xc)
        print("fp32  (a)==(b):", torch.equal(ya, yb), " (a)==(c):", torch.equal(ya, yc),
              " max|a-b|:", float((ya - yb).abs().max()))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ya16, yb16 = lin(xa), lin_b(x)
        print("bf16  (a)==(b):", torch.equal(ya16, yb16))
        raw0 = x[:, 2:] * std + mean
        raw0[:, [c - 2 for c in cols]] = 0.0
        x0 = x.clone()
        x0[:, 2:] = (raw0 - mean) / std
        print("standardised value of raw 0 A:", [round(float(v), 3) for v in x0[0, cols]])
        print("raw=0 A (NOT removal) max|diff vs (a)|:", float((lin(x0) - ya).abs().max()))

    # (4) mean standardised CA channels over the cells the model actually receives
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    sc = [STRUCTURAL_PAIR_FEATURES.index(n) for n in CA_FEATURES]
    fo, hs, ff = idx["feat_offset"], idx["he_size"], idx["feat_flat"]
    s_trunc = torch.zeros(2, dtype=torch.float64)
    s_diag0 = torch.zeros(2, dtype=torch.float64)
    n_cells, n_diag, n_rows, n_bad = 0, 0, 0, 0
    for r in range(len(hs)):
        T = int(hs[r])
        a, e = int(fo[r]), int(fo[r + 1])
        if T <= 0 or e <= a:
            continue
        if e - a != T * T * n_struct:
            n_bad += 1
            continue
        blk = ff[a:e].reshape(T, T, n_struct).float()[: args.max_he, : args.max_he][..., sc]
        k = blk.shape[0]
        z = (blk - mean[[PAIR_FEATURE_NAMES.index(n) for n in CA_FEATURES]]) / std[
            [PAIR_FEATURE_NAMES.index(n) for n in CA_FEATURES]]
        s_trunc += z.reshape(-1, 2).sum(0).double()
        s_diag0 += torch.diagonal(blk, dim1=0, dim2=1).T.eq(0).sum(0).double()
        n_cells += k * k
        n_diag += k
        n_rows += 1
    print(f"rows used={n_rows} size-mismatch rows={n_bad} cells={n_cells}")
    for j, n in enumerate(CA_FEATURES):
        print(f"  {n:<14s} E[standardised | truncated to {args.max_he}] = {float(s_trunc[j] / n_cells):+.4f}"
              f"   diagonal raw==0 fraction = {float(s_diag0[j] / n_diag):.4f}")


if __name__ == "__main__":
    main()
