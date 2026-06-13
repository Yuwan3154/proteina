import numpy as np

# UDLM position-bias schedule (proteinfoundation/flow_matching/discrete_md4.py)
# alpha_t(t, d_pos; k) = alpha_cos(t) * alpha_sig(t, d_pos; k)
# t=0 clean, t=1 fully noised. d_pos = |i-j|/(L-1) in [0,1]. k = tightness.
EPS = 1e-4


def alpha_cos(t):
    return (1 - 2 * EPS) * (1 - np.cos(np.pi / 2 * (1 - t))) + EPS


def _raw_sig(x, k, d):
    return 1.0 - 1.0 / (1.0 + np.exp(-k * (x - d)))


def alpha_sig(t, d, k):
    a = _raw_sig(t, k, d)
    a0 = _raw_sig(0.0, k, d)
    a1 = _raw_sig(1.0, k, d)
    denom = max(a0 - a1, 1e-12)
    return np.clip((a - a1) / denom, EPS, 1 - EPS)


def alpha_t(t, d, k):
    return alpha_cos(t) * alpha_sig(t, d, k)


t = np.linspace(0, 1, 1001)
clean = t <= 0.3  # near the clean end (where fine corrections matter)
ks = [1, 2, 3, 5, 8, 10, 20]
ds = [0.0, 0.25, 0.5, 0.75, 1.0]

print("Per-band corruption exposure  E_t[1-alpha_t]  (full t in [0,1])")
print("  also [clean-end] = mean over t<=0.3 (near the clean/data end)")
print(f"{'k':>4} | " + " | ".join(f"d={d:<4}" for d in ds) + " || off/near ratio (d=1 vs d=0, full)")
print("-" * 86)
for k in ks:
    full = {d: float(np.mean(1 - alpha_t(t, d, k))) for d in ds}
    cl = {d: float(np.mean((1 - alpha_t(t, d, k))[clean])) for d in ds}
    cells = " | ".join(f"{full[d]:.3f}/{cl[d]:.3f}" for d in ds)
    ratio = full[1.0] / max(full[0.0], 1e-9)
    print(f"{k:>4} | {cells} || {ratio:.3f}")

print()
print("Reading: each cell = full-t exposure / clean-end(t<=0.3) exposure.")
print("Want: off-diagonal (d=1) clean-end exposure NOT ~0 (so the model practices")
print("correcting off-diagonal contacts near the clean end), WHILE keeping near>off")
print("(d=0 noised more than d=1). k=10 is the current (too-tight) setting.")
print()
# A simple score: maximize off-diagonal clean-end exposure subject to keeping a
# clear near>off ordering (near-diag full exposure at least ~1.5x off-diag).
print("Candidate scan (off-diag clean-end exposure & bias ordering):")
for k in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]:
    off_clean = float(np.mean((1 - alpha_t(t, 1.0, k))[clean]))
    near_full = float(np.mean(1 - alpha_t(t, 0.0, k)))
    off_full = float(np.mean(1 - alpha_t(t, 1.0, k)))
    mid_clean = float(np.mean((1 - alpha_t(t, 0.5, k))[clean]))
    bias = near_full / max(off_full, 1e-9)
    print(f"  k={k:>4}: off-diag clean-end={off_clean:.3f}  mid clean-end={mid_clean:.3f}  "
          f"near/off(full)={bias:.2f}")
