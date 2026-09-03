# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Optimal-transport couplings between query residues and topology-reference SSE elements.

ContactMapTriSiT establishes the query <-> reference correspondence in exactly one place: the
relative position `rel` on the joint token axis, clipped to +/-max_rel_pos. With L up to 384 and
reference elements sitting at their own-chain midpoints, |i - p_e| routinely exceeds the clip, so
for most query-reference pairs the offset is not representable at all and every such cell collapses
onto one saturated embedding. This module produces a soft coupling pi in R^{L x T} to add
alongside that embedding on the query-reference block.

Two heads, both reducing to Sinkhorn on a cost matrix:

* `sinkhorn` -- entropic OT on a learned feature cost, optionally with the Su & Hua (CVPR 2017)
  order-preserving terms. Both of those terms are LINEAR in pi, so they fold into the cost rather
  than needing a bespoke solver; the KL prior's entropy part merges with the entropic regulariser,
  giving an effective eps of `eps + lambda2`.
* `fgw` -- fused Gromov-Wasserstein, which needs no shared coordinate system and consumes the two
  contact maps directly. Uses the Peyre, Cuturi & Solomon (ICML 2016, PMLR v48 pp. 2664-2672)
  Remark-1 factorisation throughout; the naive 4-tensor is L^2 T^2 = 2.42 GB at L=384.

Convention note: alpha weights the STRUCTURE term, `E = (1-alpha) M + alpha L(C1,C2) (x) pi`
(Vayer et al., ICML 2019 Thm 3.1). This is the opposite of the common assumption and matches POT.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TINY = 1e-30


def max_normalise(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scale each sample's cost so its valid entries land in [0, 1].

    Entropic-OT eps is not scale-free, so every published eps value is meaningless without the
    normalisation it was measured under. Max-normalisation is the convention in the closest
    literature (SCOT, Pamona, novoSpaRc, FUGW); fixing it here is what makes eps=0.1 transferable.
    """
    masked = cost.masked_fill(~mask, float("-inf"))
    peak = masked.amax(dim=(-2, -1), keepdim=True)
    peak = torch.where(torch.isfinite(peak) & (peak.abs() > TINY), peak, torch.ones_like(peak))
    return cost / peak


def unit_range_normalise(cost: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Affinely map each sample's valid entries onto [0, 1].

    Needed whenever a term can make the cost NEGATIVE. `max_normalise` only bounds the positive
    side, so `exp(-cost/eps)` OVERFLOWS on a negative cost -- at the published lambda1=50 the cost
    reaches about -50 and exp(500) is inf, which then gives nan in the pi product. Underflow was the
    only direction this module originally reasoned about.

    The SHIFT is free: Sinkhorn's fixed point is unchanged by adding a constant to the cost, because
    exp(-(C+k)/eps) is a global rescale of the kernel and the scalars are absorbed into u and v. The
    subsequent divide by the range is the same deliberate choice `max_normalise` makes -- it is what
    keeps a published eps meaningful -- and it bounds the exponent to 1/eps.
    """
    neg_inf = cost.masked_fill(~mask, float("inf"))
    lo = neg_inf.amin(dim=(-2, -1), keepdim=True)
    lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    shifted = cost - lo
    hi = shifted.masked_fill(~mask, float("-inf")).amax(dim=(-2, -1), keepdim=True)
    hi = torch.where(torch.isfinite(hi) & (hi.abs() > TINY), hi, torch.ones_like(hi))
    return shifted / hi


def sinkhorn(
    cost: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    eps: float,
    n_iter: int,
) -> torch.Tensor:
    """Plain (non-log-domain) Sinkhorn-Knopp, `n_iter` fixed iterations. Returns pi [B, L, T].

    Log-domain is deliberately not used: fp32 exp() goes subnormal at cost/eps > 87.3, and on a
    max-normalised cost the worst case here is 1/0.05 = 20.

    ⛔ That reasoning covers UNDERFLOW only, and a NEGATIVE cost overflows instead -- exp(+500) is
    inf. Every caller must therefore hand this a cost already mapped into [0,1]; use
    `unit_range_normalise` rather than `max_normalise` whenever any term can go negative.
    """
    gibbs = torch.exp(-cost / eps) * mask
    u = mu.clone()
    v = nu.clone()
    for _ in range(n_iter):
        v = nu / torch.einsum("blt,bl->bt", gibbs, u).clamp_min(TINY)
        u = mu / torch.einsum("blt,bt->bl", gibbs, v).clamp_min(TINY)
    return u[:, :, None] * gibbs * v[:, None, :]


def _normalised_index(
    q_mask: torch.Tensor, r_mask: torch.Tensor, he_pos: Optional[torch.Tensor], index_mode: str
):
    """Per-token positions in [0, 1] for the order-preserving terms.

    `rank` uses each element's ORDER among the valid elements; `raw` uses its own-chain midpoint
    scaled by the largest valid midpoint. Order preservation is a statement about order, so `rank`
    is the faithful reading.

    ⚠️ Do NOT read this as "the 2026-08-27 objection to `topology_he_pos` does not apply here". Only
    half of it is avoided. This prior does not rewrite the reference's PAIR FEATURES into query
    coordinates, so it escapes the specific incoherence that was objected to -- features and
    positions are not put on different length scales. But BOTH index modes are uniform
    fraction-along-chain maps, so the uniform-stretch assumption that MOTIVATED that objection is
    still imported here, just as a soft bias on the coupling rather than as a hard encoding.

    ⛔⛔ And unlike the fused-GW mixing weight, it is not self-attenuating. `alpha` is an
    `nn.Parameter`, so a model that finds the relational term unhelpful can drive it to 0 and recover
    the feature-only solution on its own. `lambda1`/`lambda2`/`delta` are plain floats. Where the
    uniform-stretch assumption is wrong for a given query/reference pair there is no parameter with
    which to back away from it -- the trunk can only route around it. Since this prior is also the
    ONLY source of query-index dependence in `sinkhorn` mode (at `lambda1 -> 0` the cost collapses
    back to `f(aatype_i, token_e)`, at most `n_residue_types` distinct rows), that mode's entire
    positional capability is bought with an assumption the model cannot unlearn.
    """
    assert index_mode in ("rank", "raw"), index_mode
    B, L = q_mask.shape
    device = q_mask.device
    q_idx = torch.arange(L, device=device, dtype=torch.float32)[None].expand(B, L)
    q_last = (q_mask.sum(dim=1, keepdim=True).float() - 1.0).clamp_min(1.0)
    a = (q_idx / q_last).clamp(0.0, 1.0)

    if index_mode == "rank":
        r_rank = (r_mask.float().cumsum(dim=1) - 1.0).clamp_min(0.0)
        r_last = (r_mask.sum(dim=1, keepdim=True).float() - 1.0).clamp_min(1.0)
        b = (r_rank / r_last).clamp(0.0, 1.0)
    else:
        assert he_pos is not None, "index_mode='raw' needs topology_he_pos_raw"
        pos = he_pos.float().masked_fill(~r_mask, 0.0)
        b = pos / pos.amax(dim=1, keepdim=True).clamp_min(1.0)
    return a, b


def order_preserving_cost(
    q_mask: torch.Tensor,
    r_mask: torch.Tensor,
    he_pos: Optional[torch.Tensor],
    lambda1: float,
    lambda2: float,
    delta: float,
    index_mode: str,
) -> torch.Tensor:
    """Su & Hua's two order terms as one additive cost [B, L, T], both linear in pi.

    `lambda1` scales the inverse-difference-moment (local homogeneity, rewarded so it enters
    negatively); `lambda2` scales the KL prior against a Gaussian band around the diagonal.
    """
    a, b = _normalised_index(q_mask, r_mask, he_pos, index_mode)
    d = a[:, :, None] - b[:, None, :]
    idm = 1.0 / (d * d + 1.0)
    log_prior = -(d * d) / (2.0 * delta * delta)
    return -lambda1 * idm - lambda2 * log_prior


def gw_const_term(
    c_q: torch.Tensor, c_r: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor
) -> torch.Tensor:
    """The pi-independent half of the factorisation: f1(C_q) mu 1^T + 1 nu^T f2(C_r)^T."""
    left = torch.einsum("bij,bj->bi", c_q * c_q, mu)[:, :, None]
    right = torch.einsum("bts,bs->bt", c_r * c_r, nu)[:, None, :]
    return left + right


def gw_tensor_product(
    c_q: torch.Tensor, c_r: torch.Tensor, pi: torch.Tensor, const: torch.Tensor
) -> torch.Tensor:
    """The square-loss GW tensor product L(C_q, C_r) (x) pi, factorised.

    Peyre et al. ICML 2016 Remark 1: `const - 2 C_q pi C_r^T`, O(L^2 T + L T^2), never
    materialising the L^2 T^2 tensor.
    """
    return const - 2.0 * torch.einsum("bij,bjt->bit", c_q, torch.einsum("bjs,bts->bjt", pi, c_r))


def gw_tensor_product_naive(
    c_q: torch.Tensor, c_r: torch.Tensor, pi: torch.Tensor
) -> torch.Tensor:
    """The same product from the explicit L^2 T^2 tensor. TESTING ONLY -- 2.42 GB at L=384."""
    l4 = (c_q[:, :, None, :, None] - c_r[:, None, :, None, :]) ** 2  # [B, L, T, L, T]
    return torch.einsum("bitjs,bjs->bit", l4, pi)


def fused_gw(
    c_q: torch.Tensor,
    c_r: torch.Tensor,
    feat_cost: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    eps: float,
    n_iter: int,
    n_outer: int,
) -> torch.Tensor:
    """Fused-GW coupling by block-coordinate descent, returning a differentiable pi [B, L, T].

    The outer BCD iterations run under `no_grad` -- that is where memory would accumulate, and no
    paper in this literature backpropagates through a GW solve (TFGW, NeurIPS 2022, holds T* fixed
    by the envelope theorem). ⛔ The envelope theorem alone is NOT sufficient here: it
    differentiates the optimal VALUE, whereas this head injects pi ITSELF as a feature. So the
    final iteration is re-run with gradients on -- one Sinkhorn of n_iter steps, ~1 MB of graph at
    K=10 -- which keeps the head's projections trainable at bounded cost.
    """
    const = gw_const_term(c_q, c_r, mu, nu)
    with torch.no_grad():
        pi = mu[:, :, None] * nu[:, None, :] * mask
        for _ in range(max(n_outer - 1, 0)):
            grad = gw_tensor_product(c_q, c_r, pi, const)
            total = (1.0 - alpha) * feat_cost + alpha * grad
            pi = sinkhorn(max_normalise(total, mask), mask, mu, nu, eps, n_iter)
    grad = gw_tensor_product(c_q, c_r, pi, const)
    total = (1.0 - alpha) * feat_cost + alpha * grad
    return sinkhorn(max_normalise(total, mask), mask, mu, nu, eps, n_iter)


class OTAlignmentHead(nn.Module):
    """Query <-> reference coupling, projected and added onto the query-reference pair block.

    Zero-initialised on purpose: with the head enabled the model is numerically identical to the
    baseline at step 0, so any later difference is attributable to the coupling alone.
    """

    def __init__(
        self,
        dim: int,
        mode: str = "sinkhorn",
        eps: float = 0.1,
        n_iter: int = 10,
        n_outer: int = 10,
        alpha_init: float = 0.5,
        order_preserving: bool = False,
        lambda1: float = 50.0,
        lambda2: float = 0.1,
        delta: float = 1.0,
        index_mode: str = "rank",
        cost_dim: int = 64,
    ):
        super().__init__()
        assert mode in ("sinkhorn", "fgw"), mode
        self.mode = mode
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.n_outer = int(n_outer)
        self.order_preserving = bool(order_preserving)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.delta = float(delta)
        self.index_mode = index_mode

        self.q_proj = nn.Linear(dim, cost_dim)
        self.r_proj = nn.Linear(dim, cost_dim)
        # alpha is learned in [0,1] via a sigmoid, per TFGW; it silently absorbs the ratio of the
        # feature and structure terms' raw scales and so is not transferable across cost scalings.
        self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(float(alpha_init))))
        self.project = nn.Linear(1, dim)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def feature_cost(self, q_feat: torch.Tensor, r_feat: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.q_proj(q_feat), dim=-1)
        r = F.normalize(self.r_proj(r_feat), dim=-1)
        return 1.0 - torch.einsum("bld,btd->blt", q, r)

    def coupling(
        self,
        q_feat: torch.Tensor,
        r_feat: torch.Tensor,
        c_q: torch.Tensor,
        c_r: torch.Tensor,
        q_mask: torch.Tensor,
        r_mask: torch.Tensor,
        he_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = q_mask[:, :, None] & r_mask[:, None, :]
        mu = q_mask.float() / q_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        nu = r_mask.float() / r_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        cost = max_normalise(self.feature_cost(q_feat, r_feat), mask)
        if self.order_preserving:
            cost = cost + order_preserving_cost(
                q_mask, r_mask, he_pos, self.lambda1, self.lambda2, self.delta, self.index_mode
            )
            # ⛔ MANDATORY: the order terms enter NEGATIVELY, so the sum leaves [0,1] and
            # exp(-cost/eps) overflows. Measured nan at lambda1=50 AND at 10, finite at 1.0.
            cost = unit_range_normalise(cost, mask)
        cost = cost.masked_fill(~mask, 0.0)

        if self.mode == "sinkhorn":
            return sinkhorn(cost, mask, mu, nu, self.eps, self.n_iter)
        alpha = torch.sigmoid(self.alpha_logit)
        c_q = c_q * (q_mask[:, :, None] & q_mask[:, None, :]).to(c_q.dtype)
        c_r = c_r * (r_mask[:, :, None] & r_mask[:, None, :]).to(c_r.dtype)
        return fused_gw(c_q, c_r, cost, mask, mu, nu, alpha, self.eps, self.n_iter, self.n_outer)

    def forward(self, **kwargs) -> torch.Tensor:
        """Returns the injection [B, L, T, dim] to add onto the query-reference pair block."""
        pi = self.coupling(**kwargs)
        return self.project(pi[..., None])
