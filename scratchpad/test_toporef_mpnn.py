"""CPU unit tests for Task 3: ProteinMPNN 9-residue chunk topology reference (toporef_mpnn)."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, gather_nodes
from proteinfoundation.datasets.sse_topology import MASK_TOKEN, N_PAIR_FEATURES
from proteinfoundation.nn.contact_map_tri_v2 import ContactMapTriV2
from proteinfoundation.nn.mpnn_topology_encoder import (
    CHUNK_SIZE,
    MPNNBackboneEncoder,
    chunk9_coarse_reference_from_template,
)


def make_synthetic_backbone(B: int, L: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    ca = torch.cumsum(torch.randn(B, L, 3, generator=g) * 1.5 + torch.tensor([3.2, 0.4, 0.3]), dim=1)
    n = ca + torch.tensor([-1.45, 0.0, 0.0])
    c = ca + torch.tensor([1.52, 0.0, 0.0])
    o = c + torch.tensor([0.0, 1.23, 0.0])
    x4 = torch.stack([n, ca, c, o], dim=2)  # [B, L, 4, 3]
    mask = torch.ones(B, L, dtype=torch.float32)
    return x4, mask


def test_mpnn_backbone_encoder_matches_upstream():
    torch.manual_seed(0)
    ref_mpnn = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
    ).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "v_48_020_mock.pt")
        torch.save({"model_state_dict": ref_mpnn.state_dict(), "num_edges": 48}, ckpt_path)
        enc = MPNNBackboneEncoder(k_neighbors=48, ckpt_path=ckpt_path)

    # Ensure .train(True) keeps frozen MPNN in eval mode and all params frozen
    enc.train(True)
    assert not enc.training
    for name, p in enc.named_parameters():
        assert not p.requires_grad, f"MPNN param {name} should be frozen"

    x4, mask = make_synthetic_backbone(B=2, L=41, seed=7)
    res_idx = torch.arange(41)[None].expand(2, 41)
    chain_lbl = torch.ones(2, 41, dtype=torch.long)

    h_V, h_E, E_idx, mask_attend = enc(x4, mask, residue_idx=res_idx, chain_labels=chain_lbl)

    with torch.no_grad():
        E_ref, E_idx_ref = ref_mpnn.features(x4, mask, res_idx, chain_lbl)
        h_V_ref = torch.zeros((E_ref.shape[0], E_ref.shape[1], E_ref.shape[-1]))
        h_E_ref = ref_mpnn.W_e(E_ref)
        ma_ref = gather_nodes(mask.unsqueeze(-1), E_idx_ref).squeeze(-1) * mask.unsqueeze(-1)
        for layer in ref_mpnn.encoder_layers:
            h_V_ref, h_E_ref = layer(h_V_ref, h_E_ref, E_idx_ref, mask, ma_ref)

    assert torch.allclose(h_V, h_V_ref, atol=1e-6), f"max diff h_V: {(h_V - h_V_ref).abs().max().item()}"
    assert torch.allclose(h_E, h_E_ref, atol=1e-6), f"max diff h_E: {(h_E - h_E_ref).abs().max().item()}"
    assert torch.equal(E_idx, E_idx_ref)
    assert torch.allclose(mask_attend, ma_ref, atol=1e-6)
    print("[PASS] MPNNBackboneEncoder bit-for-bit match + frozen weights verified")


def test_contact_map_tri_v2_mpnn_chunk9_forward_backward():
    torch.manual_seed(123)
    B = 2
    L_q = 50
    L_tmpl = 41  # ceil(41 / 9) = 5 chunks (last chunk has 5 valid residues)
    T = (L_tmpl + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert T == 5

    model = ContactMapTriV2(
        pair_dim=128,
        tri_hidden=128,
        n_blocks_ref=2,
        n_blocks_query=2,
        transition_hidden=512,
        dim_cond=128,
        max_topology_he_len=96,
        max_rel_pos=64,
        topology_vocab_size=45,
        n_elem_features=1,
        pair_ref_features="both",
        query_pos="index",
        reference_mode="mpnn_chunk9",
        align_head={"enabled": True},
        mlm_head={"enabled": True},  # Should be forcibly disabled in mpnn_chunk9 mode!
        compile={"enabled": False},
    )
    model.train(True)
    assert model.mlm_head is None, "mlm_head must be disabled in mpnn_chunk9 mode"
    assert not model.mpnn_topo.mpnn.training, "Frozen MPNN encoder must remain in eval mode"

    # Build coarse-grained 9-residue chunk features (i) + template backbone coords (ii)
    x4_tmpl, mask_tmpl = make_synthetic_backbone(B=B, L=L_tmpl, seed=99)
    atom37 = torch.zeros(L_tmpl, 37, 3)
    atom37[:, [0, 1, 2, 4], :] = x4_tmpl[0]
    atom37_mask = torch.zeros(L_tmpl, 37)
    atom37_mask[:, [0, 1, 2, 4]] = 1.0
    ca = x4_tmpl[0, :, 1, :]
    cm_tmpl = ((ca[:, None, :] - ca[None, :, :]).norm(dim=-1) < 8.0).float()
    cm_tmpl.fill_diagonal_(0.0)
    coarse = chunk9_coarse_reference_from_template(cm_tmpl, atom37, atom37_mask)

    he_tokens = coarse["topology_he_tokens"][None].expand(B, T).clone()
    # Mask chunk 2 in sample 0 to test random chunk token masking without MLM loss
    he_tokens[0, 2] = MASK_TOKEN

    batch = {
        "contact_map_t": torch.rand(B, L_q, L_q),
        "mask": torch.ones(B, L_q, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (B, L_q)),
        "t": torch.tensor([0.35, 0.75], dtype=torch.float32),
        "topology_he_tokens": he_tokens,
        "topology_he_pos_raw": coarse["topology_he_pos_raw"][None].expand(B, T).clone(),
        "topology_he_feat": coarse["topology_he_feat"][None].expand(B, T, T, N_PAIR_FEATURES).clone(),
        "topology_he_elem_feat": coarse["topology_he_elem_feat"][None].expand(B, T, 1).clone(),
        "template_backbone_coords": x4_tmpl,
        "template_backbone_mask": mask_tmpl,
    }

    out = model(batch)
    assert "mlm_logits" not in out, "mlm_logits must not be emitted in mpnn_chunk9 mode"
    assert out["contact_map_logits"].shape == (B, L_q, L_q)
    assert out["align_logits"].shape == (B, L_q, T)
    assert out["align_none_logits"].shape == (B, L_q)

    # Note: AF2 'final' init sets head weights (out, align_head, align_none) to 0 at step 0;
    # give heads non-zero weights (as after 1 optimizer step) so upstream gradients are non-zero.
    with torch.no_grad():
        torch.nn.init.normal_(model.out.weight, std=0.02)
        torch.nn.init.normal_(model.align_head.weight, std=0.02)
        torch.nn.init.normal_(model.align_none.weight, std=0.02)

    out = model(batch)
    target_cm = torch.randint(0, 2, (B, L_q, L_q)).float()
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(out["contact_map_logits"], target_cm)
        + out["align_logits"].mean()
        + out["align_none_logits"].mean()
    )
    loss.backward()

    for name, p in model.mpnn_topo.mpnn.named_parameters():
        assert p.grad is None, f"Frozen MPNN param {name} should have p.grad is None"

    for name, p in model.named_parameters():
        if name.startswith("mpnn_topo.mpnn."):
            continue
        assert p.grad is not None, f"Trainable param {name} has None gradient in normal mode"
        assert torch.isfinite(p.grad).all(), f"Trainable param {name} has non-finite gradient"

    assert model.mpnn_topo.chunk_transformer.mask_chunk_emb.grad.abs().max().item() > 0.0
    assert model.mpnn_topo.edge_ffn.fc1.weight.grad.abs().max().item() > 0.0
    assert model.mpnn_topo.edge_ffn.fc2.weight.grad.abs().max().item() > 0.0
    print("[PASS] ContactMapTriV2(reference_mode='mpnn_chunk9') normal mode forward/backward verified")

    # Case B: 0-edge / unconditional dropped reference (_set_empty) where all ceil(L_q / 9) chunks are masked
    model.zero_grad(set_to_none=True)
    T_empty = (L_q + CHUNK_SIZE - 1) // CHUNK_SIZE  # ceil(50 / 9) = 6
    assert T_empty == 6
    batch_empty = {
        "contact_map_t": torch.rand(B, L_q, L_q),
        "mask": torch.ones(B, L_q, dtype=torch.bool),
        "residue_type": torch.randint(0, 20, (B, L_q)),
        "t": torch.tensor([0.2, 0.8], dtype=torch.float32),
        "topology_he_tokens": torch.full((B, T_empty), MASK_TOKEN, dtype=torch.long),
        "topology_he_pos_raw": (torch.arange(T_empty, dtype=torch.float32) + 0.5) * (L_q / T_empty)
        * torch.ones(B, 1),
        "topology_he_feat": torch.zeros(B, T_empty, T_empty, N_PAIR_FEATURES),
        "topology_he_elem_feat": torch.zeros(B, T_empty, 1),
        "template_backbone_coords": torch.zeros(B, 0, 4, 3),
        "template_backbone_mask": torch.zeros(B, 0),
    }
    out_empty = model(batch_empty)
    loss_empty = (
        torch.nn.functional.binary_cross_entropy_with_logits(out_empty["contact_map_logits"], target_cm)
        + out_empty["align_logits"].mean()
        + out_empty["align_none_logits"].mean()
    )
    loss_empty.backward()

    for name, p in model.named_parameters():
        if name.startswith("mpnn_topo.mpnn."):
            continue
        assert p.grad is not None, f"Trainable param {name} has None gradient in 0-edge / empty mode (DDP break!)"
        assert torch.isfinite(p.grad).all(), f"Trainable param {name} has non-finite gradient in 0-edge mode"

    assert model.mpnn_topo.chunk_transformer.mask_chunk_emb.grad.abs().max().item() > 0.0
    print("[PASS] ContactMapTriV2(reference_mode='mpnn_chunk9') 0-edge / unconditional mode DDP gradient flow verified")


if __name__ == "__main__":
    test_mpnn_backbone_encoder_matches_upstream()
    test_contact_map_tri_v2_mpnn_chunk9_forward_backward()
    print("ALL TOPOREF MPNN CPU TESTS PASSED")
