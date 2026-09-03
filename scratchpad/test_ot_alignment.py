"""Standalone CPU checks for proteinfoundation/nn/ot_alignment.py. Run: python scratchpad/test_ot_alignment.py"""

import sys

import torch

# Loaded by path, not by package: proteinfoundation.nn.__init__ pulls in the whole model stack,
# while ot_alignment.py is pure torch. Importing it this way keeps the check runnable on a laptop
# and proves the module has no heavy dependencies.
import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "ot_alignment",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "proteinfoundation", "nn", "ot_alignment.py"),
)
_ot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ot)

OTAlignmentHead = _ot.OTAlignmentHead
gw_const_term = _ot.gw_const_term
gw_tensor_product = _ot.gw_tensor_product
gw_tensor_product_naive = _ot.gw_tensor_product_naive
max_normalise = _ot.max_normalise
order_preserving_cost = _ot.order_preserving_cost
sinkhorn = _ot.sinkhorn

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def make_batch(B=3, L=9, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    q_mask = torch.zeros(B, L, dtype=torch.bool)
    r_mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        q_mask[b, : L - b] = True          # ragged, so padding is actually exercised
        r_mask[b, : T - (b % 3)] = True
    c_q = torch.rand(B, L, L, generator=g)
    c_q = 0.5 * (c_q + c_q.transpose(1, 2))
    c_r = torch.rand(B, T, T, generator=g)
    c_r = 0.5 * (c_r + c_r.transpose(1, 2))
    return q_mask, r_mask, c_q, c_r


torch.manual_seed(0)
q_mask, r_mask, c_q, c_r = make_batch()
B, L = q_mask.shape
T = r_mask.shape[1]
mask = q_mask[:, :, None] & r_mask[:, None, :]
mu = q_mask.float() / q_mask.sum(1, keepdim=True)
nu = r_mask.float() / r_mask.sum(1, keepdim=True)

# 1. max_normalise puts valid entries in [0,1] and leaves the peak at exactly 1
cost = torch.rand(B, L, T) * 7.0 + 0.5
norm = max_normalise(cost, mask)
peak = norm.masked_fill(~mask, float("-inf")).amax(dim=(-2, -1))
check("max_normalise: valid entries <= 1", bool((norm[mask] <= 1.0 + 1e-6).all()))
check("max_normalise: peak == 1", torch.allclose(peak, torch.ones(B), atol=1e-6))
flat = max_normalise(torch.zeros(B, L, T), mask)
check("max_normalise: all-zero cost does not divide by zero", bool(torch.isfinite(flat).all()))

# 2. Sinkhorn marginals and masking
pi = sinkhorn(norm, mask, mu, nu, eps=0.1, n_iter=200)
check("sinkhorn: row marginals == mu", torch.allclose(pi.sum(2), mu, atol=1e-5),
      f"max err {(pi.sum(2) - mu).abs().max():.2e}")
check("sinkhorn: col marginals == nu", torch.allclose(pi.sum(1), nu, atol=1e-5),
      f"max err {(pi.sum(1) - nu).abs().max():.2e}")
check("sinkhorn: masked cells exactly 0", bool((pi[~mask] == 0).all()))
check("sinkhorn: total mass 1", torch.allclose(pi.sum((1, 2)), torch.ones(B), atol=1e-5))

# 3. THE ONE THAT CAN FAIL SILENTLY: factorised GW == naive L^2 T^2 tensor
pi_ref = sinkhorn(norm, mask, mu, nu, eps=0.2, n_iter=100)
mu_pi, nu_pi = pi_ref.sum(2), pi_ref.sum(1)
fact = gw_tensor_product(c_q, c_r, pi_ref, gw_const_term(c_q, c_r, mu_pi, nu_pi))
naive = gw_tensor_product_naive(c_q, c_r, pi_ref)
err = (fact - naive).abs().max().item()
check("GW: factorised == naive 4-tensor", err < 1e-4, f"max abs err {err:.3e}")

# 4. Zero-init means the head contributes exactly nothing at step 0
for mode in ("sinkhorn", "fgw"):
    head = OTAlignmentHead(dim=16, mode=mode, eps=0.1, n_iter=10, n_outer=3)
    q_feat, r_feat = torch.randn(B, L, 16), torch.randn(B, T, 16)
    inj = head(q_feat=q_feat, r_feat=r_feat, c_q=c_q, c_r=c_r, q_mask=q_mask, r_mask=r_mask)
    check(f"{mode}: zero-init injection is exactly 0", bool((inj == 0).all()),
          f"shape {tuple(inj.shape)}")

# 5. Gradients still reach the cost projections in fgw mode (the detach must not sever them)
for mode in ("sinkhorn", "fgw"):
    head = OTAlignmentHead(dim=16, mode=mode, eps=0.1, n_iter=10, n_outer=3)
    torch.nn.init.normal_(head.project.weight, std=0.1)   # un-zero so a gradient can exist
    q_feat, r_feat = torch.randn(B, L, 16), torch.randn(B, T, 16)
    head(q_feat=q_feat, r_feat=r_feat, c_q=c_q, c_r=c_r, q_mask=q_mask, r_mask=r_mask).sum().backward()
    gq = head.q_proj.weight.grad
    check(f"{mode}: gradient reaches q_proj", gq is not None and bool((gq.abs() > 0).any()),
          f"|grad| = {gq.abs().max():.3e}" if gq is not None else "grad is None")
    if mode == "fgw":
        ga = head.alpha_logit.grad
        check("fgw: gradient reaches learned alpha", ga is not None and bool(ga.abs() > 0),
              f"|grad| = {ga.abs().item():.3e}" if ga is not None else "grad is None")

# 6. Order-preserving cost is monotone-favouring: the diagonal band must be cheapest
op = order_preserving_cost(q_mask, r_mask, None, lambda1=50.0, lambda2=0.1, delta=1.0,
                           index_mode="rank")
b = 0
nq, nr = int(q_mask[b].sum()), int(r_mask[b].sum())
sub = op[b, :nq, :nr]
argmin_per_row = sub.argmin(dim=1)
# Compare COSTS, not indices: with nq=9 and nr=5 several rows sit exactly between two elements
# (|3/8 - 1/4| == |3/8 - 2/4|), so argmin and round legitimately pick different ties.
frac = torch.arange(nq).float() / max(nq - 1, 1) * max(nr - 1, 1)
expected = frac.round().long()
got_cost = sub.gather(1, argmin_per_row[:, None]).squeeze(1)
exp_cost = sub.gather(1, expected[:, None]).squeeze(1)
check("order-preserving: cheapest match per row is the order-matched one",
      torch.allclose(got_cost, exp_cost, atol=1e-6),
      f"argmin {argmin_per_row.tolist()} vs order-matched {expected.tolist()}, "
      f"max cost gap {(got_cost - exp_cost).abs().max():.2e}")
check("order-preserving: monotone rows give a monotone argmin",
      bool((argmin_per_row[1:] >= argmin_per_row[:-1]).all()))

# 7. 'raw' index mode runs and stays finite
he_pos = torch.rand(B, T) * 300.0
op_raw = order_preserving_cost(q_mask, r_mask, he_pos, 50.0, 0.1, 1.0, "raw")
check("order-preserving: index_mode='raw' finite", bool(torch.isfinite(op_raw).all()))

print()
print(f"{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
