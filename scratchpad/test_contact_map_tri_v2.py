"""Gate for proteinfoundation/nn/contact_map_tri_v2.py (T8 tri). CPU. Module equalities in float64; the full model in
float32 (its time embedding casts to float32, as ContactMapTriSiT's does), tolerance 1e-5.

1. Fused triangle updates == OpenFold's unfused forward (same parameters), outgoing and incoming.
2. SwiGLUTransition == the AF3 Alg. 11 formula written out; 'relu' (He) input init; zero output when unconditioned.
3. The AF pieces match the AF3 code: one-hot Linear lookup, adaptive_layernorm, Fourier constants and the conditioning
   pipeline, init values (zero AdaLN linears, zero-weight gates with bias -2, default-init conditioned outputs,
   final-init heads, 2k+1 relpos bins); at init the time conditioning has no effect on a block (as in AF3).
4. With all weights randomised: padding invariance (an extra masked residue and an extra masked element leave every
   valid output unchanged), the pad-bucket path equals the unpadded path, logits are symmetric and zero off-mask.
5. The align / MLM heads read the END of stage A: their loss gives gradient to stage A, none to stage B; the contact
   loss reaches both.
6. With all weights randomised, every parameter gets a gradient from the summed losses (nothing unused for DDP).
7. Parameter count at the user's sizes (4 + 4 blocks, width 128, tri hidden 128, SwiGLU 512, 96 elements).
Run: python scratchpad/test_contact_map_tri_v2.py
"""

import math

import torch

from proteinfoundation.nn.af3_fourier_constants import AF3_FOURIER_BIAS, AF3_FOURIER_WEIGHT
from proteinfoundation.nn.contact_map_tri_v2 import (
    AdaLN,
    ContactMapTriV2,
    FourierEmbedding,
    OneHotLinear,
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
    big = SwiGLUTransition(128, 512)
    sd = big.linear_a.weight.std().item()
    assert abs(sd / math.sqrt(2 / 128) - 1) < 0.05, sd
    assert SwiGLUTransition(16, 32, out_init="default").linear_out.weight.abs().max() > 0
    print(f"PASS SwiGLU == AF3 Alg. 11 formula (max |diff| {d:.1e}), bias-free, zero at init; "
          f"input std {sd:.4f} vs He sqrt(2/128) {math.sqrt(2 / 128):.4f}")


def test_alphafold_pieces():
    oh = OneHotLinear(7, 5).double()
    idx = torch.tensor([[0, 3, 6], [2, 2, 1]])
    want = torch.nn.functional.one_hot(idx, 7).double() @ oh.linear.weight.t() + oh.linear.bias
    assert torch.allclose(oh(idx), want) and oh.linear.bias.abs().max() == 0
    ad = AdaLN(16, 8).double()
    randomise_(ad)
    a, s = torch.randn(2, 4, 4, 16, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64)
    sn = torch.nn.functional.layer_norm(s, (8,), weight=ad.ln_s.weight)
    an = torch.nn.functional.layer_norm(a, (16,))
    want = torch.sigmoid(sn @ ad.linear_s.weight.t() + ad.linear_s.bias)[:, None, None] * an + (sn @ ad.linear_nobias_s.weight.t())[:, None, None]
    assert torch.allclose(ad(a, s), want) and ad.ln_s.bias is None and ad.ln_a.weight is None
    fe = FourierEmbedding()
    t = torch.tensor([0.0, 0.3, 1.0])
    w, b = torch.tensor(AF3_FOURIER_WEIGHT), torch.tensor(AF3_FOURIER_BIAS)
    assert torch.allclose(fe(t), torch.cos(2 * math.pi * (t[:, None] * w + b))) and len(fe.state_dict()) == 0
    m = ContactMapTriV2(**CFG)
    randomise_(m)
    c = torch.nn.functional.layer_norm(fe(t), (256,), weight=m.fourier_ln.weight) @ m.cond_in.weight.t()
    for tr in m.cond_transitions:
        h = tr.t.layer_norm(c)
        c = c + (torch.nn.functional.silu(h @ tr.t.linear_a.weight.t()) * (h @ tr.t.linear_b.weight.t())) @ tr.t.linear_out.weight.t()
    cm = m.cond_in(m.fourier_ln(m.fourier(t)))
    for tr in m.cond_transitions:
        cm = tr(cm)
    assert torch.allclose(cm, c, atol=1e-5) and m.fourier_ln.bias is None and m.cond_in.bias is None
    m = ContactMapTriV2(**CFG)
    blocks = list(m.blocks_ref) + list(m.blocks_query)
    for blk in blocks:
        assert all(bool((g.bias == -2.0).all()) and g.weight.abs().max() == 0 for g in blk.gate)
        assert all(x.linear_s.weight.abs().max() == 0 and x.linear_s.bias.abs().max() == 0
                   and x.linear_nobias_s.weight.abs().max() == 0 for x in blk.adaln)
        for lin in (blk.tri_out.linear_z, blk.tri_in.linear_z, blk.transition.linear_out):
            assert lin.weight.abs().max() > 0, "conditioned output projections take the default init (AF3)"
        assert blk.tri_out.linear_g.weight.abs().max() == 0 and bool((blk.tri_out.linear_g.bias == 1.0).all())
    for tr in m.cond_transitions:
        assert tr.t.linear_out.weight.abs().max() == 0, "unconditioned transition: final init"
    assert m.rel_pos_emb.linear.weight.shape[1] == 2 * CFG["max_rel_pos"] + 1
    for h in (m.align_head, m.align_none, m.mlm_head, m.out):
        assert h.weight.abs().max() == 0 and h.bias.abs().max() == 0
    assert not any(isinstance(x, torch.nn.LayerNorm) and n.startswith(("mid", "out")) for n, x in m.named_modules())
    print("PASS AF pieces: one-hot lookup; adaptive_layernorm; AF3 Fourier constants (not in state_dict) + conditioning "
          "pipeline; inits (AdaLN zeros, gates 0/-2, conditioned outputs default, trimul gating 0/1, heads final, "
          "relpos 2k+1); no LayerNorm before the heads")


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


def test_block_at_init():
    m = ContactMapTriV2(**CFG).eval()
    z = torch.randn(2, 9, 9, 128)
    pm = torch.ones(2, 9, 9)
    c1, c2 = torch.randn(2, 128), torch.randn(2, 128)
    with torch.no_grad():
        for blk in list(m.blocks_ref) + list(m.blocks_query):
            a, b = blk(z, pm, c1), blk(z, pm, c2)
            assert torch.equal(a, b) and not torch.allclose(a, z), "fresh block: active, but independent of the condition"
            x = blk.adaln[0](z, c1)
            assert torch.allclose(x, 0.5 * torch.nn.functional.layer_norm(z, (128,)), atol=1e-6)
    print("PASS fresh blocks: non-identity (default-init outputs, gate sigmoid(-2)), condition-independent, AdaLN = 0.5 LN(z)")
    m2 = ContactMapTriV2(**{**CFG, "align_head": {"enabled": False}, "mlm_head": {"enabled": False}})
    assert m2(batch())["contact_map_logits"].shape == (2, 11, 11)
    print("PASS no heads -> model runs")


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


def test_every_param_gets_grad():
    m = ContactMapTriV2(**CFG)
    randomise_(m)
    o = m(batch())
    (o["contact_map_logits"].sum() + o["align_logits"].sum() + o["align_none_logits"].sum() + o["mlm_logits"].sum()).backward()
    names = [n for n, p in m.named_parameters()]
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, dead
    print(f"PASS all {len(names)} parameter tensors get a non-zero gradient")


def test_param_count():
    m = ContactMapTriV2(**CFG)
    n = sum(p.numel() for p in m.parameters())
    per = sum(p.numel() for p in m.blocks_ref[0].parameters())
    print(f"INFO params {n:,} total; per block {per:,} "
          f"(tri-out {sum(p.numel() for p in m.blocks_ref[0].tri_out.parameters()):,}, "
          f"SwiGLU {sum(p.numel() for p in m.blocks_ref[0].transition.parameters()):,}, "
          f"AdaLN+gates {sum(p.numel() for p in list(m.blocks_ref[0].adaln.parameters()) + list(m.blocks_ref[0].gate.parameters())):,}); 8 blocks {8 * per:,}")
    assert len(m.blocks_ref) == 4 and len(m.blocks_query) == 4


if __name__ == "__main__":
    test_fused_trimul()
    test_swiglu()
    test_alphafold_pieces()
    test_block_at_init()
    test_model_invariances()
    test_head_gradients()
    test_every_param_gets_grad()
    test_param_count()
    print("ALL PASS")
