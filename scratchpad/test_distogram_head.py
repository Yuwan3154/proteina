"""Gate for the ContactMapTriSiT distogram head.

Checks the things that would otherwise fail silently in a multi-day run: that the baseline stays
bit-identical without the head, that the head emits the exact shape proteina.py asserts on, that it
reads the QUERY block rather than the joint (L+T) grid, and that predict_coords no longer trips the
structure-from-distogram path that both backends raise NotImplementedError on.
"""

import sys

import torch

from proteinfoundation.datasets.sse_topology import N_PAIR_FEATURES
from proteinfoundation.nn.contact_map_tri import ContactMapTriSiT

B, L, T, DIM, BUCKETS = 2, 16, 5, 64, 39

BASE = dict(
    pair_dim=DIM,
    tri_hidden=DIM,
    n_blocks=2,
    transition_n=2,
    dim_cond=32,
    max_rel_pos=8,
    topology_cond=True,
    max_topology_he_len=T,
    topology_vocab_size=T + 1,
    n_residue_types=22,
    pair_ref_features="both",
    contact_map_mode=True,
    contact_map_input_dim=1,
    non_contact_value=0,
)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, L)
    mask[1, L - 3:] = 0.0  # a padded sample, so masking is actually exercised
    # he_valid is derived inside the model as (he_tokens > 0), so a 0 token IS the padding.
    he_tokens = torch.randint(1, T + 1, (B, T), generator=g)
    he_tokens[1, T - 2:] = 0
    return {
        "contact_map_t": torch.rand(B, L, L, generator=g),
        "contact_map_sc": torch.rand(B, L, L, generator=g),
        "residue_type": torch.randint(0, 21, (B, L), generator=g),
        "mask": mask,
        "t": torch.rand(B, generator=g),
        "topology_he_tokens": he_tokens,
        "topology_he_pos_raw": torch.arange(T).float()[None].repeat(B, 1),
        "topology_he_feat": torch.rand(B, T, T, N_PAIR_FEATURES, generator=g),
    }


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    torch.manual_seed(0)
    batch = make_batch()
    results = []

    print("1. baseline (no num_buckets_predict_pair)")
    torch.manual_seed(0)
    base = ContactMapTriSiT(**BASE).eval()
    with torch.no_grad():
        out_base = base(dict(batch))
    results.append(check("no pair_logits emitted", "pair_logits" not in out_base))
    results.append(check("dist_head is None", base.dist_head is None))
    results.append(check("predict_coords is None (was False)", base.predict_coords is None))
    results.append(check("predict_coords is NOT False", base.predict_coords is not False))

    print("2. with the head")
    torch.manual_seed(0)
    head = ContactMapTriSiT(**BASE, num_buckets_predict_pair=BUCKETS).eval()
    with torch.no_grad():
        out_head = head(dict(batch))
    pl = out_head.get("pair_logits")
    results.append(check("pair_logits emitted", pl is not None))
    results.append(check(f"shape == [B, L, L, {BUCKETS}] (query block, not L+T)",
                         tuple(pl.shape) == (B, L, L, BUCKETS)))
    results.append(check("last dim == loss.num_dist_buckets (proteina.py asserts this)",
                         pl.shape[-1] == BUCKETS))
    results.append(check("symmetric in i<->j",
                         torch.allclose(pl, pl.transpose(1, 2), atol=1e-6)))
    results.append(check("finite", torch.isfinite(pl).all().item()))

    q = batch["mask"].bool()
    qpair = q[:, :, None] & q[:, None, :]
    results.append(check("padded pairs are exactly zero",
                         (pl[~qpair] == 0).all().item()))
    results.append(check("valid pairs are not all zero",
                         pl[qpair].abs().sum().item() > 0))

    print("3. the head must not disturb the contact output")
    results.append(check("contact_map_logits bit-identical to baseline",
                         torch.equal(out_base["contact_map_logits"],
                                     out_head["contact_map_logits"])))

    print("4. gradients reach the head")
    head.train()
    out = head(dict(batch))
    out["pair_logits"].square().mean().backward()
    gnorm = sum(p.grad.abs().sum().item() for p in head.dist_head.parameters()
                if p.grad is not None)
    results.append(check(f"dist_head grad norm > 0 ({gnorm:.3e})", gnorm > 0))

    print("5. the shape the aux loss will index")
    bs, n = pl.shape[0], pl.shape[1]
    flat = pl.view(bs * n * n, BUCKETS)
    results.append(check("view(bs*n*n, buckets) succeeds -- matches proteina.py:617",
                         flat.shape == (bs * n * n, BUCKETS)))

    print()
    n_ok, n_all = sum(results), len(results)
    print(f"{n_ok}/{n_all} checks pass")
    return 0 if n_ok == n_all else 3


if __name__ == "__main__":
    sys.exit(main())
