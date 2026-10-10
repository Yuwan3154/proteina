"""Gate for proteinfoundation/nn/contact_map_tri_v2.py (T8 tri). CPU. Module equalities in float64; the full model in
float32 (its time embedding casts to float32, as ContactMapTriSiT's does), tolerance 1e-5.

1. Fused triangle updates == OpenFold's unfused forward (same parameters), outgoing and incoming.
2. SwiGLUTransition == the AF3 Alg. 11 formula written out; zero output at init (final init).
3. Every block is the identity at init (zero-init FiLM + 'final' output layers), so an untrained model's logits are 0.
4. With all weights randomised: padding invariance (an extra masked residue and an extra masked element leave every
   valid output unchanged), the pad-bucket path equals the unpadded path, logits are symmetric and zero off-mask.
5. The align / MLM heads read the END of stage A: their loss gives gradient to stage A, none to stage B; the contact
   loss reaches both.
6. Parameter count at the user's sizes (4 + 4 blocks, width 128, tri hidden 128, SwiGLU 512, 96 elements).
Run: python scratchpad/test_contact_map_tri_v2.py
"""

import torch

from proteinfoundation.nn.contact_map_tri_v2 import (
    ContactMapTriV2,
    SwiGLUTransition,
    TriMulIncomingFused,
    TriMulOutgoingFused,
)
from proteinfoundation.openfold_stub.model.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

torch.manual_seed(0)
DT = torch.float32
CFG = dict(pair_dim=128, tri_hidden=128, n_blocks_ref=4, n_blocks_query=4, transition_hidden=512, dim_cond=128,
           max_topology_he_len=96, max_rel_pos=32, topology_vocab_size=45, n_elem_features=1, n_residue_types=22,
           pair_ref_features="both", align_head={"enabled": True}, mlm_head={"enabled": True})


def randomise_(m):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.05)


def test_fused_trimul():
    for ref_cls, fused_cls in ((TriangleMultiplicationOutgoing, TriMulOutgoingFused),
                               (TriangleMultiplicationIncoming, TriMulIncomingFused)):
        ref, fused = ref_cls(c_z=16, c_hidden=8).double(), fused_cls(c_z=16, c_hidden=8).double()
        randomise_(ref)
        fused.load_state_dict(ref.state_dict())
        z = torch.randn(2, 7, 7, 16, dtype=torch.float64)
        mask = torch.ones(2, 7, 7, dtype=torch.float64)
        mask[1, 5:, :] = 0
        mask[1, :, 5:] = 0
        d = (ref(z, mask) - fused(z, mask)).abs().max().item()
        assert d < 1e-12, (ref_cls.__name__, d)
        print(f"PASS fused {ref_cls.__name__}: max |diff| {d:.1e}")


def test_swiglu():
    t = SwiGLUTransition(16, 32).double()
    z = torch.randn(3, 5, 5, 16, dtype=torch.float64)
    m = torch.ones(3, 5, 5, dtype=torch.float64)
    assert t(z, m).abs().max().item() == 0.0, "final init must give zero output"
    randomise_(t)
    h = t.layer_norm(z)
    want = (torch.nn.functional.silu(h @ t.linear_a.weight.T) * (h @ t.linear_b.weight.T)) @ t.linear_out.weight.T
    d = (t(z, m) - want).abs().max().item()
    assert d < 1e-12 and t.linear_a.bias is None and t.linear_out.bias is None, d
    print(f"PASS SwiGLU == AF3 Alg. 11 formula (max |diff| {d:.1e}), bias-free, zero at init")


def batch(B=2, L=11, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(2, 45, (B, T), generator=g)
    cm = (torch.rand(B, L, L, generator=g) > 0.8)
    return {
        "contact_map_t": 0.5 * (cm + cm.transpose(1, 2)).float(),
        "contact_map_sc": torch.rand(B, L, L, generator=g, dtype=DT),
        "mask": torch.ones(B, L),
        "residue_type": torch.randint(0, 20, (B, L), generator=g),
        "t": torch.rand(B, generator=g, dtype=DT),
        "topology_he_tokens": tok,
        "topology_he_pos_raw": torch.sort(torch.rand(B, T, generator=g) * L, dim=1).values,
        "topology_he_feat": torch.randn(B, T, T, 8, generator=g, dtype=DT),
        "topology_he_elem_feat": torch.randn(B, T, 1, generator=g, dtype=DT),
    }


def pad_one(b):
    """Append one masked residue and one masked (token 0) element."""
    o = dict(b)
    P = torch.nn.functional.pad
    o["contact_map_t"] = P(b["contact_map_t"], (0, 1, 0, 1), value=0.7)
    o["contact_map_sc"] = P(b["contact_map_sc"], (0, 1, 0, 1), value=0.3)
    o["mask"] = P(b["mask"], (0, 1))
    o["residue_type"] = P(b["residue_type"], (0, 1), value=5)
    o["topology_he_tokens"] = P(b["topology_he_tokens"], (0, 1))
    o["topology_he_pos_raw"] = P(b["topology_he_pos_raw"], (0, 1), value=3.0)
    o["topology_he_feat"] = P(b["topology_he_feat"], (0, 0, 0, 1, 0, 1), value=1.5)
    o["topology_he_elem_feat"] = P(b["topology_he_elem_feat"], (0, 0, 0, 1), value=2.0)
    return o


def test_identity_at_init():
    m = ContactMapTriV2(**CFG).eval()
    z = torch.randn(2, 9, 9, 128)
    pm = torch.ones(2, 9, 9)
    cond = torch.randn(2, 128)
    with torch.no_grad():
        for blk in list(m.blocks_ref) + list(m.blocks_query):
            assert torch.equal(blk(z, pm, cond), z), "a fresh block must be the identity"
    print("PASS every fresh block is exactly the identity (zero-init FiLM, final-init tri-mul and SwiGLU outputs)")
    m2 = ContactMapTriV2(**{**CFG, "align_head": {"enabled": False}, "mlm_head": {"enabled": False}})
    assert m2.mid_norm is None and m2(batch())["contact_map_logits"].shape == (2, 11, 11)
    print("PASS no heads -> no mid_norm (nothing unconsumed for DDP)")


def test_model_invariances():
    m = ContactMapTriV2(**CFG).eval()
    randomise_(m)
    b = batch()
    with torch.no_grad():
        o1 = m(b)
        o2 = m(pad_one(b))
    L, T = b["contact_map_t"].shape[1], b["topology_he_tokens"].shape[1]
    for k, sl in (("contact_map_logits", (slice(None), slice(0, L), slice(0, L))),
                  ("align_logits", (slice(None), slice(0, L), slice(0, T))),
                  ("align_none_logits", (slice(None), slice(0, L))),
                  ("mlm_logits", (slice(None), slice(0, T)))):
        d = (o1[k] - o2[k][sl]).abs().max().item()
        assert d < 1e-5, (k, d)
    assert o2["contact_map_logits"][:, L, :].abs().max().item() == 0.0
    assert o2["align_logits"][:, :, T].abs().max().item() == 0.0
    lg = o1["contact_map_logits"]
    assert (lg - lg.transpose(1, 2)).abs().max().item() < 1e-6
    assert lg.abs().max().item() > 0
    print(f"PASS padding invariance on all 4 outputs (residue + element), symmetric, |logit| max {lg.abs().max():.3f}")

    m.pad_len_buckets, m.pad_topo_buckets = [16], [8]
    with torch.no_grad():
        o3 = m(b)
    for k in o1:
        d = (o1[k] - o3[k]).abs().max().item()
        assert o1[k].shape == o3[k].shape and d < 1e-5, (k, d)
    print("PASS pad-bucket path (L 11->16, T 6->8) == unpadded path on every output")


def test_head_gradients():
    m = ContactMapTriV2(**CFG)
    randomise_(m)
    b = batch()

    def grads(loss):
        m.zero_grad(set_to_none=True)
        loss.backward()
        ref = sum(p.grad.abs().sum().item() for p in m.blocks_ref.parameters() if p.grad is not None)
        qry = [p.grad for p in m.blocks_query.parameters()]
        return ref, sum(0.0 if g is None else g.abs().sum().item() for g in qry)

    o = m(b)
    ref, qry = grads(o["align_logits"].sum() + o["align_none_logits"].sum() + o["mlm_logits"].sum())
    assert ref > 0 and qry == 0.0, (ref, qry)
    o = m(b)
    ref2, qry2 = grads(o["contact_map_logits"].sum())
    assert ref2 > 0 and qry2 > 0, (ref2, qry2)
    print(f"PASS heads at the end of stage A: align+MLM grad -> stage A {ref:.2e}, stage B {qry}; contact -> both")


def test_param_count():
    m = ContactMapTriV2(**CFG)
    n = sum(p.numel() for p in m.parameters())
    per = sum(p.numel() for p in m.blocks_ref[0].parameters())
    print(f"INFO params {n:,} total; per block {per:,} "
          f"(tri-out {sum(p.numel() for p in m.blocks_ref[0].tri_out.parameters()):,}, "
          f"SwiGLU {sum(p.numel() for p in m.blocks_ref[0].transition.parameters()):,}, "
          f"FiLM {sum(p.numel() for p in m.blocks_ref[0].mod.parameters()):,}); 8 blocks {8 * per:,}")
    assert len(m.blocks_ref) == 4 and len(m.blocks_query) == 4


if __name__ == "__main__":
    test_fused_trimul()
    test_swiglu()
    test_identity_at_init()
    test_model_invariances()
    test_head_gradients()
    test_param_count()
    print("ALL PASS")
