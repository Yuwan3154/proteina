"""CPU gates for CATemplateCompress1D (the four 2D -> 1D decompressors)."""

import pytest
import torch
import torch.nn.functional as F

from proteinfoundation.nn.ca_template_compress import DECOMPRESSORS, CATemplateCompress1D, runs_from_labels

NB = 39
CFG = dict(
    token_dim=32, pair_repr_dim=16, dim_cond=32, nheads=4, n_pre=1, n_mid=1, n_post=1, mid_dim=16, mid_tri_hidden=16,
    mid_dim_cond=16, num_buckets_predict_pair=NB, opm_dim=4, spectral_heads=2, spectral_r=3, spectral_hidden=8,
    xattn_heads=4, topology_vocab_size=44, true_seg_until_step=10,
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
        sse_use_true=True,
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


def test_segmentation_is_dssp_runs_with_minus_one_breaks():
    b = _batch()
    m = _model("basis_pool")
    out = m(b)
    y = out["sse_labels_used"]
    assert torch.equal(torch.where(b["dssp_target"] >= 0, b["dssp_target"], y), y)
    # row 0: 0 0 | 1 1 1 1 | 0 | -1 | 0 | 2 2 2 | 0 0 | 1 1 1 | 0 0 0  -> 9 runs (the -1 residue is its own run)
    assert int(out["sse_K"][0]) == 9, out["sse_K"]
    assert int(out["sse_K"][1]) == 8, out["sse_K"]  # 1 1 1|0 0|2 2|0 0 0|1 1 1 1|0|2 2|0 (17 valid residues)


def test_runs_block_mean_pooling():
    y = torch.tensor([[0, 0, 1, 1, 1, 2]])
    valid = torch.ones_like(y, dtype=torch.bool)
    _, seg_id, K = runs_from_labels(y, valid, torch.zeros_like(valid))
    A = F.one_hot(seg_id, int(K)).float()
    z = torch.randn(1, 6, 6, 3)
    n = A.sum(1)
    zk = torch.einsum("bik,bijc,bjl->bklc", A, z, A) / (n[:, :, None, None] * n[:, None, :, None])
    assert torch.allclose(zk[0, 1, 2], z[0, 2:5, 5:6].mean((0, 1)))
    assert torch.allclose(zk[0, 0, 1], z[0, 0:2, 2:5].mean((0, 1)))


@pytest.mark.parametrize("dec,varies", [("pair_row_xattn", True), ("basis_pool", True), ("spectral", False)])
def test_within_run_variation(dec, varies):
    b = _batch()
    m = _model(dec, perturb=True).eval()
    captured = {}
    m.decomp.register_forward_hook(lambda mod, inp, out: captured.setdefault("o", out))
    with torch.no_grad():
        m(b)
    o = captured["o"][0]
    run = o[2:6]  # residues 2..5 form one helix run in sample 0
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
