# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""ProteinMPNN-based 9-residue chunk topology reference encoder for ContactMapTriV2 (Task 3 / toporef_mpnn)."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ProteinMPNN.protein_mpnn_utils import EncLayer, ProteinFeatures, gather_nodes
from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    N_PAIR_FEATURES,
    assemble_pair_features,
    element_lengths,
    element_positions,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.openfold_stub.model.primitives import LayerNorm, Linear, lecun_normal_init_

CHUNK_SIZE = 9
MPNN_DIM = 128
MPNN_ATOM4_INDICES = (0, 1, 2, 4)  # N, CA, C, O in atom37 numbering


def extract_backbone_atom4(
    coords: torch.Tensor, coord_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract [..., L, 4, 3] (N, CA, C, O) and [..., L] validity mask from atom37 or atom4 tensors."""
    if coords.shape[-2] == 4:
        x4 = coords.float()
        m4 = (coord_mask > 0.5).all(dim=-1).float() if coord_mask.ndim == coords.ndim - 1 else coord_mask.float()
        return x4, m4
    x4 = coords[..., list(MPNN_ATOM4_INDICES), :].float()
    m4 = (coord_mask[..., list(MPNN_ATOM4_INDICES)] > 0.5).all(dim=-1).float()
    return x4, m4


def chunk9_runs(length: int, chunk_size: int = CHUNK_SIZE):
    """Partition a chain of `length` residues into ceil(length / chunk_size) contiguous chunks."""
    if length <= 0:
        return []
    n_chunks = (length + chunk_size - 1) // chunk_size
    return [(DSSP_HELIX, min(chunk_size, length - c * chunk_size)) for c in range(n_chunks)]


def chunk9_coarse_reference_from_template(
    contact_map: torch.Tensor,
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    feat_mean: Optional[torch.Tensor] = None,
    feat_std: Optional[torch.Tensor] = None,
    elem_mean: float = 0.0,
    elem_std: float = 1.0,
    chunk_size: int = CHUNK_SIZE,
) -> Dict[str, torch.Tensor]:
    """Compute coarse-grained 9-residue chunk pair features (i), midpoints, and chunk lengths."""
    L = int(contact_map.shape[0])
    runs = chunk9_runs(L, chunk_size=chunk_size)
    T = len(runs)
    if T == 0:
        return {
            "topology_he_tokens": torch.ones(1, dtype=torch.long),
            "topology_he_pos_raw": torch.zeros(1, dtype=torch.float32),
            "topology_he_contact": torch.zeros(1, 1, dtype=torch.float32),
            "topology_he_feat": torch.zeros(1, 1, N_PAIR_FEATURES, dtype=torch.float32),
            "topology_he_elem_feat": torch.zeros(1, 1, dtype=torch.float32),
        }
    he_contact, keep = sse_contact_reference(contact_map.cpu(), runs, keep_types=(DSSP_HELIX,))
    structural = sse_structural_pair_features(
        contact_map.cpu(), coords.cpu(), coord_mask.cpu(), runs, keep
    )
    raw_feat = assemble_pair_features(he_contact, structural, runs, keep)
    if feat_mean is not None and feat_std is not None:
        raw_feat = (raw_feat - feat_mean.cpu().float()) / feat_std.cpu().float().clamp(min=1e-6)
    pos_raw = element_positions(runs, target_len=None)
    lengths = element_lengths(runs, keep)
    elem_feat = ((lengths - float(elem_mean)) / max(float(elem_std), 1e-6))[:, None]
    return {
        "topology_he_tokens": torch.full((T,), 2, dtype=torch.long),
        "topology_he_pos_raw": pos_raw,
        "topology_he_contact": he_contact,
        "topology_he_feat": raw_feat,
        "topology_he_elem_feat": elem_feat,
    }


class MPNNBackboneEncoder(nn.Module):
    """Frozen pretrained ProteinMPNN backbone encoder (N, CA, C, O + virtual CB -> h_V, h_E, E_idx)."""

    def __init__(
        self,
        node_features: int = MPNN_DIM,
        edge_features: int = MPNN_DIM,
        hidden_dim: int = MPNN_DIM,
        num_encoder_layers: int = 3,
        k_neighbors: int = 48,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        self.features = ProteinFeatures(
            edge_features=edge_features,
            node_features=node_features,
            num_positional_embeddings=16,
            num_rbf=16,
            top_k=k_neighbors,
            augment_eps=0.0,
        )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.encoder_layers = nn.ModuleList(
            [EncLayer(hidden_dim, hidden_dim * 2, dropout=0.0) for _ in range(num_encoder_layers)]
        )

        if ckpt_path is not None:
            self.load_pretrained(ckpt_path)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Keep frozen ProteinMPNN encoder in eval mode at all times.
        return super().train(False)

    def load_pretrained(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        enc_prefixes = ("features.", "W_e.", "encoder_layers.")
        filtered = {k: v for k, v in state.items() if k.startswith(enc_prefixes)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        missing_learned = set(missing) - {"features.D_mu", "features.D_sigma"}
        if missing_learned or unexpected:
            raise RuntimeError(
                f"ProteinMPNN encoder state_dict mismatch from {ckpt_path}: "
                f"missing={sorted(missing_learned)}, unexpected={sorted(unexpected)}"
            )

    @torch.no_grad()
    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: Optional[torch.Tensor] = None,
        chain_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = X.shape[0], X.shape[1]
        device = X.device
        if L == 0:
            empty_v = X.new_zeros((B, 0, self.hidden_dim))
            empty_e = X.new_zeros((B, 0, 0, self.hidden_dim))
            empty_idx = torch.zeros((B, 0, 0), dtype=torch.long, device=device)
            empty_m = X.new_zeros((B, 0, 0))
            return empty_v, empty_e, empty_idx, empty_m

        X_f = X.float()
        mask_f = mask.float()
        if residue_idx is None:
            residue_idx = torch.arange(L, device=device, dtype=torch.long)[None].expand(B, L)
        else:
            residue_idx = residue_idx.long()
        if chain_labels is None:
            chain_labels = torch.ones((B, L), device=device, dtype=torch.long)
        else:
            chain_labels = chain_labels.long()

        E, E_idx = self.features(X_f, mask_f, residue_idx, chain_labels)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device, dtype=E.dtype)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask_f.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask_f.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask_f, mask_attend)
        return h_V, h_E, E_idx, mask_attend


class Chunk9SwiGLU(nn.Module):
    """Pre-norm SwiGLU FFN with 4x expansion (512 for d=128)."""

    def __init__(self, d: int, expansion: int = 4):
        super().__init__()
        self.hidden = d * expansion
        self.norm = LayerNorm(d)
        self.linear_a = Linear(d, self.hidden, bias=False, init="relu")
        self.linear_b = Linear(d, self.hidden, bias=False, init="relu")
        self.linear_out = Linear(self.hidden, d, bias=False, init="default")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, b = F.linear(h, torch.cat([self.linear_a.weight, self.linear_b.weight])).split(self.hidden, dim=-1)
        return self.linear_out(F.silu(a) * b) * mask.unsqueeze(-1)


class Chunk9TransformerBlock(nn.Module):
    """Self-attention (4 heads) + SwiGLU (4x) attending independently within each 9-residue chunk."""

    def __init__(self, d: int = MPNN_DIM, num_heads: int = 4, ffn_expansion: int = 4):
        super().__init__()
        if d % num_heads != 0:
            raise ValueError(f"d={d} must be divisible by num_heads={num_heads}")
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm_attn = LayerNorm(d)
        self.qkv = Linear(d, 3 * d, bias=False, init="default")
        self.out_proj = Linear(d, d, bias=False, init="default")
        self.ffn = Chunk9SwiGLU(d, expansion=ffn_expansion)

    def forward(self, x: torch.Tensor, res_mask: torch.Tensor) -> torch.Tensor:
        # x: [N_chunks, 9, d], res_mask: [N_chunks, 9] float in {0, 1}
        N, C, D = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(N, C, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [N, H, C, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, H, C, C]
        key_valid = res_mask[:, None, None, :] > 0.5
        any_valid = key_valid.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~key_valid & any_valid, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * res_mask[:, None, :, None] * res_mask[:, None, None, :]

        out = torch.matmul(attn, v).transpose(1, 2).reshape(N, C, D)
        x = (x + self.out_proj(out)) * res_mask.unsqueeze(-1)
        x = x + self.ffn(x, res_mask)
        return x * res_mask.unsqueeze(-1)


class Chunk9Transformer(nn.Module):
    """2-layer Transformer over 9-residue chunks + masked mean pooling + learned mask chunk embedding."""

    def __init__(
        self,
        in_dim: int = MPNN_DIM,
        d: int = MPNN_DIM,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        chunk_size: int = CHUNK_SIZE,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d = d
        self.chunk_size = chunk_size

        self.in_proj = Linear(in_dim, d, init="default")
        self.intra_chunk_pos = nn.Parameter(torch.zeros(chunk_size, d))
        with torch.no_grad():
            lecun_normal_init_(self.intra_chunk_pos)

        self.layers = nn.ModuleList(
            [Chunk9TransformerBlock(d=d, num_heads=num_heads, ffn_expansion=ffn_expansion) for _ in range(num_layers)]
        )
        self.out_norm = LayerNorm(d)
        self.mask_chunk_emb = nn.Parameter(torch.zeros(d))
        with torch.no_grad():
            nn.init.normal_(self.mask_chunk_emb, mean=0.0, std=0.02)

    def forward(
        self,
        h_V: torch.Tensor,
        res_mask: torch.Tensor,
        target_T: int,
        chunk_valid: torch.Tensor,
        chunk_masked: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return [B, target_T, d] unit embeddings for each 9-residue chunk."""
        B, L, _ = h_V.shape
        C = self.chunk_size
        total_res = target_T * C
        if L < total_res:
            h_pad = F.pad(h_V, (0, 0, 0, total_res - L))
            m_pad = F.pad(res_mask, (0, total_res - L))
        else:
            h_pad = h_V[:, :total_res]
            m_pad = res_mask[:, :total_res]

        x = h_pad.reshape(B * target_T, C, self.in_dim)
        m = m_pad.reshape(B * target_T, C).to(h_V.dtype)

        x = (self.in_proj(x) + self.intra_chunk_pos[None].to(x.dtype)) * m.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, m)
        x = self.out_norm(x) * m.unsqueeze(-1)

        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x.sum(dim=1) / denom).reshape(B, target_T, self.d)

        cv = chunk_valid.to(pooled.dtype).unsqueeze(-1)
        if chunk_masked is None:
            is_masked = torch.zeros_like(cv)
        else:
            is_masked = chunk_masked.to(pooled.dtype).unsqueeze(-1)

        # Keep `pooled` in the graph with weight (1 - is_masked) so all Transformer weights
        # receive valid gradients (0.0 when 100% masked) under DDP.
        mask_vec = self.mask_chunk_emb.to(pooled.dtype).view(1, 1, self.d)
        out = pooled * (1.0 - is_masked) + mask_vec * is_masked + pooled * 0.0
        return out * cv


class ChunkPairEdgeFFN(nn.Module):
    """2-layer FFN (same d as TopoRef) over ProteinMPNN edges, pooled (averaged) onto [B, T, T, d] chunk pairs."""

    def __init__(self, in_dim: int = MPNN_DIM, d: int = MPNN_DIM, pool: str = "mean"):
        super().__init__()
        if pool not in ("mean", "sum"):
            raise ValueError(f"pool must be 'mean' or 'sum', got {pool!r}")
        self.in_dim = in_dim
        self.d = d
        self.pool = pool

        self.norm = LayerNorm(in_dim)
        self.fc1 = Linear(in_dim, d, init="relu")
        self.fc2 = Linear(d, d, init="default")

    def _run_ffn(self, e: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(e))))

    def forward(
        self,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_attend: torch.Tensor,
        target_T: int,
        chunk_valid: torch.Tensor,
        chunk_size: int = CHUNK_SIZE,
    ) -> torch.Tensor:
        """Transform ProteinMPNN edges and pool them into [B, target_T, target_T, d]."""
        B, L, K, _ = h_E.shape
        device, dtype = h_E.device, h_E.dtype
        T = target_T
        pair_valid = (chunk_valid[:, :, None] & chunk_valid[:, None, :]).to(dtype).unsqueeze(-1)

        if L == 0 or K == 0 or not bool((mask_attend > 0.5).any()):
            # Feed dummy 0 tensor through the 2-layer FFN so all parameters participate in autograd.
            dummy = h_E.new_zeros((B, 1, self.in_dim))
            dummy_out = self._run_ffn(dummy) * 0.0
            out = h_E.new_zeros((B, T, T, self.d)) + dummy_out[:, :, None, :]
            return out * pair_valid

        edge_valid = (mask_attend > 0.5).to(dtype)
        c_src = (torch.arange(L, device=device, dtype=torch.long) // chunk_size)[None, :, None].expand(B, L, K)
        c_dst = E_idx.long() // chunk_size
        in_bounds = (c_src < T) & (c_dst < T) & (c_src >= 0) & (c_dst >= 0)
        edge_valid = edge_valid * in_bounds.to(dtype)

        e_trans = self._run_ffn(h_E) * edge_valid.unsqueeze(-1)  # [B, L, K, d]

        c_src_clamped = c_src.clamp(0, max(T - 1, 0))
        c_dst_clamped = c_dst.clamp(0, max(T - 1, 0))
        flat_pair_idx = (c_src_clamped * T + c_dst_clamped).reshape(B, L * K)  # [B, L*K]

        flat_feat = e_trans.reshape(B, L * K, self.d)
        flat_valid = edge_valid.reshape(B, L * K)

        pooled = h_E.new_zeros((B, T * T, self.d))
        pooled.scatter_add_(1, flat_pair_idx.unsqueeze(-1).expand(-1, -1, self.d), flat_feat)

        if self.pool == "mean":
            counts = h_E.new_zeros((B, T * T))
            counts.scatter_add_(1, flat_pair_idx, flat_valid)
            pooled = pooled / counts.unsqueeze(-1).clamp(min=1.0)

        # Keep 0-weight touch to all edge outputs so even if all edges were out of bounds, gradients flow.
        pooled = pooled + e_trans.reshape(B, L * K, self.d).sum(dim=1, keepdim=True) * 0.0
        return pooled.reshape(B, T, T, self.d) * pair_valid


class MPNNChunk9TopologyModule(nn.Module):
    """End-to-end ProteinMPNN + 2-layer Chunk9 Transformer + 2-layer Edge FFN module for ContactMapTriV2."""

    def __init__(
        self,
        d: int = MPNN_DIM,
        num_chunk_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        k_neighbors: int = 48,
        edge_pool: str = "mean",
        mpnn_ckpt_path: Optional[str] = None,
        chunk_size: int = CHUNK_SIZE,
    ):
        super().__init__()
        self.d = d
        self.chunk_size = chunk_size
        self.mpnn = MPNNBackboneEncoder(
            node_features=MPNN_DIM,
            edge_features=MPNN_DIM,
            hidden_dim=MPNN_DIM,
            num_encoder_layers=3,
            k_neighbors=k_neighbors,
            ckpt_path=mpnn_ckpt_path,
        )
        self.chunk_transformer = Chunk9Transformer(
            in_dim=MPNN_DIM,
            d=d,
            num_layers=num_chunk_layers,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            chunk_size=chunk_size,
        )
        self.edge_ffn = ChunkPairEdgeFFN(
            in_dim=MPNN_DIM,
            d=d,
            pool=edge_pool,
        )

    def forward(
        self,
        tmpl_backbone_coords: Optional[torch.Tensor],
        tmpl_backbone_mask: Optional[torch.Tensor],
        target_T: int,
        chunk_valid: torch.Tensor,
        chunk_masked: Optional[torch.Tensor] = None,
        tmpl_residue_idx: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (chunk_emb [B, T, d], edge_pair_emb [B, T, T, d])."""
        B = chunk_valid.shape[0]
        device = chunk_valid.device

        if tmpl_backbone_coords is None or tmpl_backbone_coords.shape[1] == 0:
            h_V = torch.zeros((B, 0, MPNN_DIM), device=device, dtype=dtype)
            res_mask = torch.zeros((B, 0), device=device, dtype=dtype)
            h_E = torch.zeros((B, 0, 0, MPNN_DIM), device=device, dtype=dtype)
            E_idx = torch.zeros((B, 0, 0), device=device, dtype=torch.long)
            mask_attend = torch.zeros((B, 0, 0), device=device, dtype=dtype)
        else:
            if tmpl_backbone_mask is None:
                tmpl_backbone_mask = torch.ones(tmpl_backbone_coords.shape[:2], device=device, dtype=dtype)
            h_V, h_E, E_idx, mask_attend = self.mpnn(
                tmpl_backbone_coords, tmpl_backbone_mask, residue_idx=tmpl_residue_idx
            )
            h_V = h_V.to(dtype)
            res_mask = tmpl_backbone_mask.to(dtype)
            h_E = h_E.to(dtype)
            mask_attend = mask_attend.to(dtype)

        # Zero out edges belonging to masked chunks so masked chunks do not leak via MPNN edges.
        if chunk_masked is not None and h_E.shape[1] > 0 and h_E.shape[2] > 0:
            L_t, K_t = h_E.shape[1], h_E.shape[2]
            c_src = (torch.arange(L_t, device=device, dtype=torch.long) // self.chunk_size)[None, :, None].expand(
                B, L_t, K_t
            )
            c_dst = E_idx.long() // self.chunk_size
            c_src_c = c_src.clamp(0, max(target_T - 1, 0))
            c_dst_c = c_dst.clamp(0, max(target_T - 1, 0))
            src_masked = torch.gather(chunk_masked.bool(), 1, c_src_c.reshape(B, -1)).reshape(B, L_t, K_t)
            dst_masked = torch.gather(chunk_masked.bool(), 1, c_dst_c.reshape(B, -1)).reshape(B, L_t, K_t)
            mask_attend = mask_attend * (~(src_masked | dst_masked)).to(dtype)

        chunk_emb = self.chunk_transformer(
            h_V=h_V,
            res_mask=res_mask,
            target_T=target_T,
            chunk_valid=chunk_valid,
            chunk_masked=chunk_masked,
        )
        edge_pair_emb = self.edge_ffn(
            h_E=h_E,
            E_idx=E_idx,
            mask_attend=mask_attend,
            target_T=target_T,
            chunk_valid=chunk_valid,
            chunk_size=self.chunk_size,
        )
        return chunk_emb, edge_pair_emb
