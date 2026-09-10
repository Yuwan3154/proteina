"""Batched TM-score for use inside the training loop (no subprocess, no file I/O).

Scope is TM-SCORE, not TM-align: prediction, template and native all share the residue numbering
inside a training batch (`random_crop_to_size` applies the SAME residue window to `all_atom_*` and
to every `template_*` field), so the correspondence is FIXED and the expensive half of TM-align --
the alignment search -- is not needed. What remains is the superposition search that maximizes

    TM = (1/L_norm) * sum_i 1 / (1 + (d_i / d0)^2),   d0 = 1.24*(L_norm-15)^(1/3) - 1.8

i.e. Kabsch superposition plus Zhang & Skolnick iterative extension: seed on a contiguous fragment,
superpose, keep residues inside a growing distance cutoff, re-superpose, repeat, and take the best
TM over all seeds.

⭐ Seeds are evaluated as one batch (folded into the batch dimension) rather than in a Python loop,
so a whole training batch costs `n_iter` batched 3x3 SVDs instead of `n_seeds * n_iter` of them.
The reference loop implementation lives in `tests/test_tm_score.py` and the two are gated against
each other there, alongside a check against `USalign -TMscore 5` values.

⛔ `norm_mask` vs `mask` matters and is not cosmetic. `mask` selects residues that contribute to
the sum AND drive the superposition (a residue absent from either structure cannot be superposed);
`norm_mask` sets L_norm and d0. Leaving `norm_mask` as the NATIVE's coverage while `mask` is the
pairwise overlap is the standard convention (partial coverage is penalized rather than hidden) and
matches how every other TM number in this project is taken -- normalized by the true native, see
[[feedback_usalign_field_native]].
"""

from __future__ import annotations

import torch

# Seed schedules, measured against `USalign -TMscore 5` on 60 real pairs spanning TM 0.223-0.999
# (A6000, 2026-08-14). ⛔ REFERENCE is the default for the T4 gate; FAST is NOT accurate enough:
#
#   schedule    seeds   max err   mean err   cost @L=384,B=1 (CPU)   % of an 8.3 s T1 step (2 calls)
#   REFERENCE     247    0.0008    0.0001              30.9 ms                      0.75%
#   FAST           26    0.0440    0.0012               5.3 ms                      0.13%
#
# ⛔ FAST's error is NOT uniform -- it is concentrated at LOW TM (worst case 0.1874 vs USalign's
# 0.2314 on 2gmq_A), because min_seed_len=32 skips exactly the short seeds needed to find a small
# alignable core. An earlier measurement reported FAST at max err 0.0018; that sample simply did
# not reach down to TM ~ 0.2. Confirmed NOT a dtype effect and NOT a seed-batching effect: fp32,
# fp64 and the per-seed loop all agree to the digit at each schedule.
REFERENCE_KWARGS = {"n_iter": 20, "min_seed_len": 4}
FAST_KWARGS = {"n_iter": 10, "min_seed_len": 32}


def _kabsch(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted Kabsch. P,Q: (B,L,3); w: (B,L) non-negative. Returns (R, t) mapping P -> Q.

    ⛔⛔ RUNS WITH AUTOCAST DISABLED, not merely with float32 inputs. `torch.linalg.svd` has no bf16
    CUDA kernel, so under `--precision bf16` this raised `"svd_cuda_gesvdjBatched" not implemented for
    'BFloat16'` the first time the T4 gate ever ran on a GPU (every prior measurement was CPU-only).
    ⛔ Casting P/Q/w with .float() is NOT sufficient, and was the first failed fix: autocast intercepts
    `torch.einsum` and returns bf16 even from fp32 operands, so `H` came back bf16 and the SVD died at
    the identical line. ⭐ fp32 is also the right precision on the
    merits, not merely a workaround: this is a 3x3 SVD whose output feeds a d0-normalised TM sum, and
    fp32/fp64 were measured to agree to the printed digit, whereas bf16 has ~3 decimal digits of
    mantissa -- far too coarse for a rotation that decides a promotion at delta = 0.05 TM.
    The result is cast back to the caller's dtype so nothing downstream changes shape or type.
    """
    _in_dtype = P.dtype
    # ⛔ autocast(enabled=False) is the load-bearing part -- see the docstring.
    with torch.autocast(device_type=P.device.type, enabled=False):
        P, Q, w = P.float(), Q.float(), w.float()
        wsum = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        Pc = (P * w[..., None]).sum(dim=1) / wsum
        Qc = (Q * w[..., None]).sum(dim=1) / wsum
        P0, Q0 = P - Pc[:, None], Q - Qc[:, None]
        H = torch.einsum("bli,bl,blj->bij", P0, w, Q0)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)
        Ut = U.transpose(-2, -1)
        # reflection correction: det(V U^T) must be +1, else flip the last singular direction
        d = torch.sign(torch.linalg.det(V @ Ut))
        D = torch.eye(3, device=P.device, dtype=P.dtype).expand(P.shape[0], 3, 3).clone()
        D[:, 2, 2] = d
        R = V @ D @ Ut
        t = Qc - torch.einsum("bij,bj->bi", R, Pc)
    return R.to(_in_dtype), t.to(_in_dtype)


def _seed_windows(L: int, min_seed_len: int, max_seeds: int | None) -> list[tuple[int, int]]:
    """Zhang & Skolnick seeds: contiguous fragments of halving length at half-length strides."""
    seed_lens, fl = [], L
    while fl >= min_seed_len:
        seed_lens.append(fl)
        fl //= 2
    if not seed_lens:                       # crop shorter than min_seed_len: one full-length seed
        seed_lens = [L]
    seeds = []
    for fl in seed_lens:
        for st in range(0, max(L - fl + 1, 1), max(fl // 2, 1)):
            seeds.append((fl, st))
    if max_seeds is not None and len(seeds) > max_seeds:
        step = len(seeds) / max_seeds
        seeds = [seeds[int(i * step)] for i in range(max_seeds)]
    return seeds


def tm_score(
    pred: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
    norm_mask: torch.Tensor | None = None,
    n_iter: int = 20,
    min_seed_len: int = 4,
    max_seeds: int | None = None,
) -> torch.Tensor:
    """TM-score of `pred` against `ref` with fixed correspondence, normalized by `norm_mask`.

    Args:
        pred: (B,L,3) CA coordinates of the model / template.
        ref:  (B,L,3) CA coordinates of the native.
        mask: (B,L) 1 where the residue exists in BOTH structures (superposition + numerator).
        norm_mask: (B,L) 1 where the NATIVE has the residue; sets L_norm and d0. Defaults to `mask`.
        n_iter: refinement iterations per seed.
        min_seed_len / max_seeds: accuracy-vs-speed knobs; see FAST_KWARGS.

    Returns:
        (B,) TM-score in [0,1].
    """
    # fp32 regardless of the training precision -- an SVD in bf16 is not worth debugging
    pred = pred.float()
    ref = ref.float()
    B, L, _ = pred.shape

    if mask is None:
        mask = torch.ones(B, L, device=pred.device, dtype=pred.dtype)
    mask = mask.float()
    norm_mask = mask if norm_mask is None else norm_mask.float()

    L_norm = norm_mask.sum(dim=1).clamp_min(1.0)                          # (B,)
    d0 = (1.24 * (L_norm - 15.0).clamp_min(1e-6) ** (1.0 / 3.0) - 1.8).clamp_min(0.5)

    seeds = _seed_windows(L, min_seed_len, max_seeds)
    S = len(seeds)
    idx = torch.arange(L, device=pred.device)
    seed_w = torch.stack(
        [((idx >= st) & (idx < st + fl)).to(pred.dtype) for fl, st in seeds]
    )                                                                     # (S,L)

    # fold the seeds into the batch dimension: everything below is one (B*S) problem
    P = pred[:, None].expand(B, S, L, 3).reshape(B * S, L, 3)
    Q = ref[:, None].expand(B, S, L, 3).reshape(B * S, L, 3)
    m = mask[:, None].expand(B, S, L).reshape(B * S, L)
    d0f = d0[:, None].expand(B, S).reshape(B * S)
    d0sq = (d0f ** 2)[:, None]
    Lf = L_norm[:, None].expand(B, S).reshape(B * S)

    w = seed_w[None].expand(B, S, L).reshape(B * S, L) * m
    best = torch.zeros(B * S, device=pred.device, dtype=pred.dtype)
    # a seed with <3 paired residues cannot define a superposition; it stays frozen at TM 0.
    # The full-length seed always survives (the caller guarantees >=4 paired residues), so `best`
    # is never all-zero for a real input.
    active = w.sum(1) >= 3

    for it in range(n_iter):
        if not bool(active.any()):
            break
        R, t = _kabsch(P, Q, torch.where(active[:, None], w, torch.ones_like(w)))
        moved = torch.einsum("bij,blj->bli", R, P) + t[:, None]
        dsq = ((moved - Q) ** 2).sum(-1)
        tm = ((1.0 / (1.0 + dsq / d0sq)) * m).sum(1) / Lf
        best = torch.where(active, torch.maximum(best, tm), best)

        cutoff = (d0f + 0.5 + it * 0.25)[:, None] ** 2
        w_new = (dsq < cutoff).to(pred.dtype) * m
        # freeze (rather than break, which in a batched setting would stop every element) any seed
        # that has collapsed below 4 residues or has converged
        keep = active & (w_new.sum(1) >= 4) & (w_new != w).any(dim=1)
        w = torch.where(keep[:, None], w_new, w)
        active = keep

    return best.reshape(B, S).max(dim=1).values


def tm_score_ca(
    pred37: torch.Tensor,
    ref37: torch.Tensor,
    ref_mask37: torch.Tensor,
    pred_mask37: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """TM of atom37 `pred37` vs native `ref37` on CA (atom index 1), normalized by native coverage.

    `pred_mask37` is the predicted/template structure's own atom mask -- pass it for templates
    (which have real gaps); a model prediction covers every residue and can leave it None.
    """
    ref_ca = ref_mask37[..., 1] > 0
    both = ref_ca if pred_mask37 is None else (ref_ca & (pred_mask37[..., 1] > 0))
    return tm_score(
        pred37[:, :, 1, :], ref37[:, :, 1, :],
        mask=both.to(pred37.dtype), norm_mask=ref_ca.to(pred37.dtype), **kwargs,
    )
