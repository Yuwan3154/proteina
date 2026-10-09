"""CPU gates for CATemplateCompress1D (the four 2D -> 1D decompressors)."""

import pytest
import torch
import torch.nn.functional as F

from proteinfoundation.nn.ca_template_compress import DECOMPRESSORS, STRIDE, WIN, CATemplateCompress1D, n_windows, window_slots

NB = 39
CFG = dict(
    token_dim=32, pair_repr_dim=16, dim_cond=32, nheads=4, n_pre=1, n_mid=1, n_post=1, mid_dim=16, mid_tri_hidden=16,
    mid_dim_cond=16, num_buckets_predict_pair=NB, opm_dim=4, spectral_heads=2, spectral_r=3, spectral_hidden=8,
    xattn_heads=4, topology_vocab_size=44,
    feats_init_seq=["res_seq_pdb_idx", "chain_break_per_res", "x_sc"], feats_cond_seq=["time_emb"],
    feats_pair_repr=["rel_seq_sep", "x_sc_pair_dists", "xt_pair_dists"], feats_pair_cond=["time_emb"],
    residue_type_emb_init_seq=True, seq_emb_dim=16, t_emb_dim=16, idx_emb_dim=16, seq_sep_dim=15, xt_pair_dist_dim=8,
    xt_pair_dist_min=0.1, xt_pair_dist_max=3, x_sc_pair_dist_dim=8, x_sc_pair_dist_min=0.1, x_sc_pair_dist_max=3,
    strict_feats=False, use_qkln=True,
)


def _batch(L=20, T=4, seed=0):
    torch.manual_seed(seed)
    B = 2
    dssp = torch.tensor([[0, 0, 1, 1, 1, 1, 0, -1, 0, 2, 2, 2, 0, 0, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 0, 0, 0]])
    mask = torch.ones(B, L)
    mask[1, 17:] = 0
    tok = torch.tensor([[5, 30, 7, 0], [12, 3, 40, 9]])  # 0 = padded template element
    return dict(
        x_t=torch.randn(B, L, 3), x_sc=torch.randn(B, L, 3), t=torch.rand(B), mask=mask,
        residue_type=torch.randint(0, 20, (B, L)), dssp_target=dssp,
        topology_he_tokens=tok, topology_he_pos_raw=torch.rand(B, T) * L, topology_he_feat=torch.rand(B, T, T, 8),
    )


def _model(dec, perturb=False):
    torch.manual_seed(1)
    m = CATemplateCompress1D(decompress=dec, **CFG)
    if perturb:  # zero-initialised readouts would hide the 2D -> 1D gradient path at step 0
        with torch.no_grad():
            for n, p in m.decomp.named_parameters():
                if p.abs().sum() == 0:
                    p.normal_(std=0.1)
    return m


def _loss(out, b):
    v = b["mask"] > 0.5
    loss = F.mse_loss(out["coords_pred"][v], b["x_t"][v])
    loss = loss + F.cross_entropy(out["pair_logits"].reshape(-1, NB), torch.randint(0, NB, (out["pair_logits"].numel() // NB,)))
    loss = loss + F.cross_entropy(out["dssp_logits"].reshape(-1, 3), b["dssp_target"].reshape(-1), ignore_index=-1)
    return loss + out["align_logits"].square().mean() + out["align_none_logits"].square().mean() + out["mlm_logits"].square().mean()


@pytest.mark.parametrize("dec", DECOMPRESSORS)
def test_forward_backward_finite(dec):
    b = _batch()
    m = _model(dec)
    out = m(b)
    assert out["coords_pred"].shape == (2, 20, 3) and out["pair_logits"].shape == (2, 20, 20, NB)
    assert out["align_logits"].shape == (2, 20, 4) and out["mlm_logits"].shape == (2, 4, 44)
    _loss(out, b).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 50 and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("dec", DECOMPRESSORS)
def test_coords_loss_reaches_mid_and_pre_through_decompressor(dec):
    b = _batch()
    m = _model(dec, perturb=True)
    out = m(b)
    v = b["mask"] > 0.5
    F.mse_loss(out["coords_pred"][v], b["x_t"][v]).backward()  # coordinates only: no aux head can carry the gradient
    mid = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("mid_blocks") and p.grad is not None)
    pre = sum(float(p.grad.abs().sum()) for n, p in m.named_parameters() if n.startswith("pre_layers") and p.grad is not None)
    assert mid > 0 and pre > 0, (mid, pre)


def _membership(valid):
    K_per = n_windows(valid.sum(1))
    w, ok, p = window_slots(valid, K_per)
    A = (F.one_hot(w, int(K_per.max())).float() * ok[..., None].float()).sum(0)
    return K_per, A, p, ok


def test_windows_cover_residues_as_stride5_win9():
    assert [int(n_windows(torch.tensor(n))) for n in (1, 9, 10, 14, 15, 17, 20, 384)] == [1, 1, 2, 2, 3, 3, 4, 76]
    valid = _batch()["mask"].bool()
    K_per, A, p, ok = _membership(valid)
    assert K_per.tolist() == [4, 3]
    for b in range(2):
        n = int(valid[b].sum())
        ref = torch.zeros(valid.shape[1], A.shape[2])
        for k in range(int(K_per[b])):
            ref[k * STRIDE: min(k * STRIDE + WIN, n), k] = 1  # window k = residues 5k .. 5k+8, right-padded
        assert torch.equal(A[b], ref), b
    assert bool(((p >= 0) & (p < WIN))[ok].all())


def test_window_block_mean_pooling():
    valid = torch.ones(1, 14, dtype=torch.bool)
    _, A, _, _ = _membership(valid)                                   # windows 0..8 and 5..13
    z = torch.randn(1, 14, 14, 3)
    n = A.sum(1)
    zk = torch.einsum("bik,bijc,bjl->bklc", A, z, A) / (n[:, :, None, None] * n[:, None, :, None])
    assert torch.allclose(zk[0, 0, 1], z[0, 0:9, 5:14].mean((0, 1)), atol=1e-6)


def test_window_tokens_ignore_masked_residues():
    torch.manual_seed(0)
    m = _model("basis_pool").eval()
    valid = _batch()["mask"].bool()
    K_per = n_windows(valid.sum(1))
    K, L = int(K_per.max()), valid.shape[1]
    win_idx = torch.arange(K)[:, None] * STRIDE + torch.arange(WIN)[None]
    win_mask = (win_idx < L)[None] & valid[:, win_idx.clamp(max=L - 1)] & (torch.arange(K)[None, :, None] < K_per[:, None, None])
    s = torch.randn(2, L, CFG["token_dim"])
    s2 = s.clone()
    s2[1, 17:] += 100.0                                                 # masked residues of sample 1
    with torch.no_grad():
        t1 = m.win_pool(s, win_idx.clamp(max=L - 1)[None].expand(2, -1, -1), win_mask)
        t2 = m.win_pool(s2, win_idx.clamp(max=L - 1)[None].expand(2, -1, -1), win_mask)
    assert int(win_mask[1].sum()) == 9 + 9 + 7 and torch.allclose(t1, t2, atol=1e-5)  # 0..8, 5..13, 10..16
    assert bool((t1[1, 3] == 0).all())                                  # window 3 does not exist for sample 1


@pytest.mark.parametrize("dec,varies", [("pair_row_xattn", True), ("basis_pool", True), ("spectral", False)])
def test_within_window_variation(dec, varies):
    b = _batch()
    m = _model(dec, perturb=True).eval()
    captured = {}
    m.decomp.register_forward_hook(lambda mod, inp, out: captured.setdefault("o", out))
    with torch.no_grad():
        m(b)
    o = captured["o"][0]
    run = o[0:5]  # first call = low slot; residues 0..4 read window 0 (clamped) at offsets 0..4
    differs = bool((run - run[0]).abs().max() > 1e-6)
    assert differs == varies


def test_spectral_degenerate_grads_finite():
    b = _batch()
    b["topology_he_tokens"] = torch.zeros_like(b["topology_he_tokens"])  # no template
    b["x_t"] = torch.zeros_like(b["x_t"])                                # symmetric, repetitive input
    b["x_sc"] = torch.zeros_like(b["x_sc"])
    b["residue_type"] = torch.zeros_like(b["residue_type"])
    m = _model("spectral", perturb=True)
    out = m(b)
    v = b["mask"] > 0.5
    F.mse_loss(out["coords_pred"][v], torch.randn_like(out["coords_pred"][v])).backward()
    bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, bad
