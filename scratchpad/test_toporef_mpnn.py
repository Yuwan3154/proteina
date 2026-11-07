"""CPU unit tests for Task 3: ProteinMPNN 9-residue chunk topology reference (toporef_mpnn)."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, gather_nodes
from proteinfoundation.datasets.sse_topology import MASK_TOKEN, N_PAIR_FEATURES
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
from proteinfoundation.nn.contact_map_tri_v2 import ContactMapTriV2
from proteinfoundation.nn.mpnn_topology_encoder import (
    CHUNK_SIZE,
    TEMPLATE_SEQ_MASK_TOKEN,
    MPNNBackboneEncoder,
    chunk9_coarse_reference_from_template,
    chunk9_runs,
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

    tmpl_aatype = torch.randint(0, 20, (B, L_tmpl), dtype=torch.long)
    # Mask ~25% of template residues i.i.d. with TEMPLATE_SEQ_MASK_TOKEN (21)
    tmpl_aatype[0, ::4] = TEMPLATE_SEQ_MASK_TOKEN

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
        "template_aatype": tmpl_aatype,
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
    assert model.mpnn_topo.chunk_transformer.seq_emb.weight.grad.abs().max().item() > 0.0
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


def test_topology_reference_transform_mpnn_chunk9_seq_masking():
    from torch_geometric.data import Data

    L_t = 36  # 4 chunks of 9
    runs = chunk9_runs(L_t, chunk_size=9)
    T = len(runs)
    assert T == 4
    x4_tmpl, m4_tmpl = make_synthetic_backbone(B=1, L=L_t, seed=55)
    aatype_tmpl = torch.arange(L_t, dtype=torch.long) % 20

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = os.path.join(tmpdir, "mock_chunk9_index.pt")
        index = {
            "ids": ["test_A", "test_A@rw150#0"],
            "cluster_of": torch.tensor([0, 0], dtype=torch.int32),
            "members_flat": torch.tensor([1], dtype=torch.int32),
            "members_offset": torch.tensor([0, 1], dtype=torch.int64),
            "seq_hash": torch.tensor([123, 123], dtype=torch.int64),
            "runs_flat": torch.tensor(runs + runs, dtype=torch.int16),
            "runs_offset": torch.tensor([0, T, 2 * T], dtype=torch.int64),
            "he_offset": torch.tensor([0, T * T, 2 * T * T], dtype=torch.int64),
            "he_size": torch.tensor([T, T], dtype=torch.int16),
            "he_flat": torch.ones(2 * T * T, dtype=torch.uint8),
            "feat_offset": torch.tensor([0, T * T * 2, 2 * T * T * 2], dtype=torch.int64),
            "feat_flat": torch.zeros(2 * T * T * 2, dtype=torch.float16),
            "align_offset": torch.tensor([0, 0, L_t], dtype=torch.int64),
            "align_flat": (torch.arange(L_t, dtype=torch.int16) // 9),
            "row_tm": torch.tensor([1.0, 0.72], dtype=torch.float16),
            "row_is_native": torch.tensor([True, False], dtype=torch.bool),
            "row_rewind": torch.tensor([0, 150], dtype=torch.int16),
            "row_usalign_tm": torch.tensor([1.0, 0.72], dtype=torch.float16),
            "pair_feature_mean": torch.zeros(N_PAIR_FEATURES, dtype=torch.float32),
            "pair_feature_std": torch.ones(N_PAIR_FEATURES, dtype=torch.float32),
            "elem_feature_mean": torch.tensor(9.0, dtype=torch.float32),
            "elem_feature_std": torch.tensor(1.0, dtype=torch.float32),
            "element_types": (1,),
            "dssp_def": "chunk9",
            "tmpl_res_offset": torch.tensor([0, L_t, 2 * L_t], dtype=torch.int64),
            "tmpl_atom4_flat": torch.cat([x4_tmpl[0], x4_tmpl[0]], dim=0).to(torch.float16),
            "tmpl_mask_flat": torch.cat([m4_tmpl[0], m4_tmpl[0]], dim=0).to(torch.bool),
            "tmpl_aatype_flat": torch.cat([aatype_tmpl, aatype_tmpl], dim=0).to(torch.int8),
        }
        torch.save(index, idx_path)

        tf = TopologyReferenceTransform(
            index_path=idx_path,
            reference_source="synthetic",
            reference_mode="mpnn_chunk9",
            element_types=(1,),
            sse_types=(1, 2),
            elem_features=True,
            token_mask_prob=0.1,
            template_seq_mask_prob=0.25,
            drop_prob=0.0,
            seed=1234,
        )

        masked_fracs = []
        for i in range(40):
            g = Data(
                coords=torch.zeros(L_t, 37, 3),
                coord_mask=torch.ones(L_t, 37, dtype=torch.bool),
                protein_id="test_A",
            )
            g = tf(g)
            # Coordinate features must NEVER be masked
            assert torch.allclose(g.template_backbone_coords, x4_tmpl[0].half().float(), atol=1e-3)
            assert torch.equal(g.template_backbone_mask, torch.ones(L_t, dtype=torch.float32))
            is_seq_masked = g.template_aatype == TEMPLATE_SEQ_MASK_TOKEN
            masked_fracs.append(is_seq_masked.float().mean().item())
            # Unmasked positions must equal original template sequence
            assert torch.equal(g.template_aatype[~is_seq_masked], aatype_tmpl[~is_seq_masked])
            # MLM target must remain all zeros in mpnn_chunk9 mode
            assert int((g.topology_he_tokens_target != 0).sum().item()) == 0

        mean_mask_rate = sum(masked_fracs) / len(masked_fracs)
        assert 0.18 <= mean_mask_rate <= 0.32, f"Expected ~0.25 seq mask rate, got {mean_mask_rate:.3f}"

        # Self/nonself validation reference (augment=False) must have 0% sequence masking
        ref_val, _ = tf.nonself_reference("test_A", L_t, seed=0)
        assert torch.equal(ref_val["template_aatype"], aatype_tmpl)
        print(f"[PASS] TopologyReferenceTransform(mpnn_chunk9) 25% i.i.d. seq masking verified (mean={mean_mask_rate:.3f})")


def test_rebuild_confind_indices_script():
    import subprocess
    import zlib
    import numpy as np
    from torch_geometric.data import Data

    L_t = 36
    old_runs = [(1, 10), (0, 2), (1, 10), (0, 4), (2, 10)]  # 3 SSE elements (2 helix, 1 strand)
    T_old = 3
    x4_tmpl, _ = make_synthetic_backbone(B=1, L=L_t, seed=77)
    aatype_tmpl = (torch.arange(L_t, dtype=torch.long) % 20).numpy().astype(np.int8)

    with tempfile.TemporaryDirectory() as tmpdir:
        src_idx = os.path.join(tmpdir, "src_index.pt")
        src_elig = os.path.join(tmpdir, "eligible_ids_tm0.5-0.9.txt")
        with open(src_elig, "w") as fh:
            fh.write("test_A\n")

        index = {
            "ids": ["test_A", "test_A@rw150#0"],
            "cluster_of": torch.tensor([0, 0], dtype=torch.int32),
            "members_flat": torch.tensor([1], dtype=torch.int32),
            "members_offset": torch.tensor([0, 1], dtype=torch.int64),
            "seq_hash": torch.tensor([123, 123], dtype=torch.int64),
            "runs_flat": torch.tensor(old_runs + old_runs, dtype=torch.int16),
            "runs_offset": torch.tensor([0, len(old_runs), 2 * len(old_runs)], dtype=torch.int64),
            "he_offset": torch.tensor([0, T_old * T_old, 2 * T_old * T_old], dtype=torch.int64),
            "he_size": torch.tensor([T_old, T_old], dtype=torch.int16),
            "he_flat": torch.ones(2 * T_old * T_old, dtype=torch.uint8),
            "feat_offset": torch.tensor([0, T_old * T_old * 2, 2 * T_old * T_old * 2], dtype=torch.int64),
            "feat_flat": torch.full((2 * T_old * T_old * 2,), 0.25, dtype=torch.float16),
            "align_offset": torch.tensor([0, 0, L_t], dtype=torch.int64),
            "align_flat": (torch.arange(L_t, dtype=torch.int16) % T_old),
            "row_tm": torch.tensor([1.0, 0.72], dtype=torch.float16),
            "row_is_native": torch.tensor([True, False], dtype=torch.bool),
            "row_rewind": torch.tensor([0, 150], dtype=torch.int16),
            "row_usalign_tm": torch.tensor([1.0, 0.72], dtype=torch.float16),
            "pair_feature_mean": torch.zeros(N_PAIR_FEATURES, dtype=torch.float32),
            "pair_feature_std": torch.ones(N_PAIR_FEATURES, dtype=torch.float32),
            "elem_feature_mean": torch.tensor(10.0, dtype=torch.float32),
            "elem_feature_std": torch.tensor(1.0, dtype=torch.float32),
            "element_types": (1, 2),
            "dssp_def": "h_bond",
        }
        torch.save(index, src_idx)

        # Create mock template pool with shardXXXX/test_auth_A.npz
        pool_dir = os.path.join(tmpdir, "pool")
        auth_stem = "test_auth_A"
        shard_dir = os.path.join(pool_dir, f"shard{zlib.crc32(auth_stem.encode()) % 1000:04d}")
        os.makedirs(shard_dir, exist_ok=True)
        amask = np.zeros((L_t, 37), dtype=bool)
        amask[:, [0, 1, 2, 4]] = True
        full = np.zeros((L_t, 37, 3), dtype=np.float32)
        full[:, [0, 1, 2, 4], :] = x4_tmpl[0].numpy()
        tcoords = full[amask][None, :, :]  # [1, N_present, 3]
        np.savez(
            os.path.join(shard_dir, f"{auth_stem}.npz"),
            coords=tcoords,
            atom_mask=amask,
            aatype=aatype_tmpl,
            residue_index=np.arange(1, L_t + 1, dtype=np.int32),
        )

        alias_path = os.path.join(tmpdir, "auth_key.tsv")
        with open(alias_path, "w") as fh:
            fh.write("label_id\tauth_id\ntest_A\ttest_auth_A\n")

        # Create mock native pack
        pack_path = os.path.join(tmpdir, "tri_v2.pack")
        g_nat = Data(
            coords=x4_tmpl[0],
            coord_mask=torch.ones(L_t, 4, dtype=torch.bool),
            residue_type=torch.from_numpy(aatype_tmpl).long(),
            contact_map_confind=torch.eye(L_t, dtype=torch.float32),
            protein_id="test_A",
        )
        with open(pack_path, "wb") as fh:
            torch.save(g_nat, fh)
            ln = fh.tell()
        np.savez(
            pack_path + ".idx.npz",
            stems=np.array(["test_A"], dtype=object),
            offsets=np.array([0], dtype=np.int64),
            lengths=np.array([ln], dtype=np.int64),
        )

        out_smooth = os.path.join(tmpdir, "smooth", "topology_index_confind_synth_tmalign_smooth.pt")
        out_chunk9 = os.path.join(tmpdir, "chunk9", "topology_index_confind_synth_mpnn_chunk9.pt")
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild_confind_indices_a6000.py")
        cmd = [
            sys.executable,
            script,
            "--src-index", src_idx,
            "--src-eligible", src_elig,
            "--native-pack", pack_path,
            "--template-pool", pool_dir,
            "--chain-alias", alias_path,
            "--out-smooth", out_smooth,
            "--out-chunk9", out_chunk9,
            "--workers", "2",
            "--num-chunks", "2",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "REBUILD_INDICES_DONE" in res.stdout

        # Verify TopologyReferenceTransform loads the rebuilt chunk9 index + pool + pack cleanly
        tf_c9 = TopologyReferenceTransform(
            index_path=out_chunk9,
            reference_source="synthetic",
            reference_mode="mpnn_chunk9",
            element_types=(1,),
            sse_types=(1, 2),
            elem_features=True,
            token_mask_prob=0.1,
            template_seq_mask_prob=0.25,
            template_pool_dir=pool_dir,
            chain_alias_path=alias_path,
            native_pack_path=pack_path,
            seed=99,
        )
        g_q = Data(
            coords=torch.zeros(L_t, 37, 3),
            coord_mask=torch.ones(L_t, 37, dtype=torch.bool),
            protein_id="test_A",
        )
        g_q = tf_c9(g_q)
        assert g_q.topology_he_tokens.shape[0] == 4
        assert g_q.template_backbone_coords.shape == (L_t, 4, 3)
        assert torch.allclose(g_q.template_backbone_coords, x4_tmpl[0], atol=1e-5)
        print("[PASS] rebuild_confind_indices_a6000.py end-to-end build + TopologyReferenceTransform verified")


if __name__ == "__main__":
    test_mpnn_backbone_encoder_matches_upstream()
    test_contact_map_tri_v2_mpnn_chunk9_forward_backward()
    test_topology_reference_transform_mpnn_chunk9_seq_masking()
    test_rebuild_confind_indices_script()
    print("ALL TOPOREF MPNN CPU TESTS PASSED")
