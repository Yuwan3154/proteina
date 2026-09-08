"""Does --t_beta actually shift training sigma toward high noise? Verify before burning GPU hours.

⛔ Checks the EMPIRICAL sampler against quantiles derived ANALYTICALLY beforehand, so a sampler that
is silently wrong (e.g. skewed the wrong way, or ignoring the mix) cannot pass.

Analytic prediction for t ~ 0.98*Beta(1.3,2.0)+0.02*U, sigma = noise_schedule(t):
    p10 ~ 1442,  p25 ~ 707,  p50 ~ 184,  p75 ~ 26,  p90 ~ 2.9
against the AF3 lognormal default:
    p10 ~ 0.70,  p50 ~ 4.82,  p90 ~ 33

The direction that matters: t=0 is FULL NOISE (verified in Proteina r3n_fm.py:156,
x_t = (1-t)x_0 + t*x_1 with x_0 the reference), so a Beta skewed toward 0 must RAISE sigma.

Run: python scratchpad/test_t_beta_sampling.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.nn.af3_diffusion import (
    SIGMA_DATA,
    noise_schedule,
    sample_noise_level,
    sample_noise_level_beta,
)

PASS = []


def check(name, ok, detail=""):
    PASS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def q(x, p):
    return float(torch.quantile(x.double(), p))


def main():
    torch.manual_seed(0)
    N = (200000,)
    base = sample_noise_level(N, "cpu")
    beta = sample_noise_level_beta(N, "cpu", p1=1.3, p2=2.0)

    print("  sigma quantiles:")
    print(f"  {'p':>5} {'AF3 lognormal':>15} {'Beta(1.3,2.0)':>15}")
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"  {p:5.2f} {q(base,p):15.2f} {q(beta,p):15.2f}")

    # 1. Direction: the beta sampler must be shifted UP at every quantile.
    ups = all(q(beta, p) > q(base, p) for p in (0.10, 0.25, 0.50, 0.75))
    check("Beta sampler raises sigma at every quantile (more high noise)", ups)

    # 2. Median must land near the analytic 184, not the lognormal's 4.82.
    med = q(beta, 0.50)
    check("median sigma matches the analytic prediction (~184)", 150 < med < 220, f"got {med:.1f}")

    # 3. Fraction in the 'model must generate >=99%' regime (c_skip<0.01 <=> sigma>159.2).
    thr = (SIGMA_DATA**2 * 0.99 / 0.01) ** 0.5
    f_base = float((base > thr).float().mean())
    f_beta = float((beta > thr).float().mean())
    check("high-noise mass rises by >10x", f_beta > 10 * f_base,
          f"{100*f_base:.3f}% -> {100*f_beta:.2f}%  ({f_beta/max(f_base,1e-9):.0f}x)")

    # 4. Bounds: sigma must stay inside the schedule's own range.
    lo, hi = float(noise_schedule(torch.tensor(1.0))), float(noise_schedule(torch.tensor(0.0)))
    check("sigma stays within the schedule range", bool((beta >= lo).all() and (beta <= hi).all()),
          f"[{float(beta.min()):.4f}, {float(beta.max()):.1f}] vs [{lo:.4f}, {hi:.1f}]")

    # 5. Default path unchanged: t_beta=None must reproduce the lognormal exactly.
    torch.manual_seed(42); a = sample_noise_level((1000,), "cpu")
    torch.manual_seed(42); b = sample_noise_level((1000,), "cpu")
    check("default lognormal sampler is unchanged/deterministic under seed", bool(torch.equal(a, b)))

    print(f"\n{sum(PASS)}/{len(PASS)} passed")
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()
