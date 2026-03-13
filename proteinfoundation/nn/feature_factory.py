# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import gzip
import math
import os
from typing import Dict, List, Literal

import torch
from loguru import logger
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

from proteinfoundation.utils.ff_utils.idx_emb_utils import get_index_embedding, get_time_embedding


# ################################
# # # Some auxiliary functions # #
# ################################


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class

    def assert_defaults_allowed(self, batch: Dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise IOError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(self, dim_feats_out, mode: Literal["seq", "pair"]):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch):
        xt = batch["x_t"]  # [b, n, 3]
        b, n = xt.shape[0], xt.shape[1]
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=xt.device)
        elif self.mode == "pair":
            return torch.zeros((b, n, n, self.dim), device=xt.device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class FoldEmbeddingSeqFeat(Feature):
    """Computes fold class embedding from precomputed cath_code_indices.

    Expects batch["cath_code_indices"] as tensor [b, 3] (sample) or
    [b, max_labels, 3] (average/sum/transformer) with optional cath_code_indices_mask.
    Indices are precomputed in the data pipeline; this module does embedding lookup only.
    """

    def __init__(
        self,
        fold_emb_dim,
        cath_code_dir,
        multilabel_mode="sample",
        fold_nhead=4,
        fold_nlayer=2,
        **kwargs,
    ):
        """
        multilabel_mode (["sample", "average", "sum", "transformer"]): Schemes to handle multiple fold labels
            "sample": one label per sample, indices [b, 3]
            "average": multiple labels, average embeddings over valid labels
            "sum": multiple labels, sum embeddings over valid labels
            "transformer": pad labels, run transformer, average output
        """
        super().__init__(dim=fold_emb_dim * 3)
        self.create_mapping(cath_code_dir)
        # The last class is null embedding
        self.embedding_C = torch.nn.Embedding(self.num_classes_C + 1, fold_emb_dim)  
        self.embedding_A = torch.nn.Embedding(self.num_classes_A + 1, fold_emb_dim)
        self.embedding_T = torch.nn.Embedding(self.num_classes_T + 1, fold_emb_dim)
        self.register_buffer("_device_param", torch.tensor(0), persistent=False)
        assert multilabel_mode in ["sample", "average", "sum", "transformer"]
        self.multilabel_mode = multilabel_mode
        if multilabel_mode == "transformer":
            encoder_layer = torch.nn.TransformerEncoderLayer(
                fold_emb_dim * 3,
                nhead=fold_nhead,
                dim_feedforward=fold_emb_dim * 3,
                batch_first=True,
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, fold_nlayer)

    @property
    def device(self):
        return next(self.buffers()).device

    def create_mapping(self, cath_code_dir):
        """Load cath label vocabulary for embedding sizes."""
        mapping_file = os.path.join(cath_code_dir, "cath_label_mapping.pt")
        if os.path.exists(mapping_file):
            class_mapping = torch.load(mapping_file, weights_only=False)
        else:
            raise IOError(f"{mapping_file} does not exist...")

        self.num_classes_C = len(class_mapping["C"])
        self.num_classes_A = len(class_mapping["A"])
        self.num_classes_T = len(class_mapping["T"])

    def forward(self, batch):
        xt = batch["x_t"]  # [b, n, 3]
        bs = xt.shape[0]
        n = xt.shape[1]
        idx = batch.get("cath_code_indices")
        if idx is None:
            # No CATH: use null embedding indices
            idx = torch.full(
                (bs, 3),
                self.num_classes_C,
                device=xt.device,
                dtype=torch.long,
            )
            idx[:, 1] = self.num_classes_A
            idx[:, 2] = self.num_classes_T
        if idx.dim() == 2:
            # [b, 3] - sample mode
            fold_emb = torch.cat(
                [
                    self.embedding_C(idx[:, 0]),
                    self.embedding_A(idx[:, 1]),
                    self.embedding_T(idx[:, 2]),
                ],
                dim=-1,
            )  # [b, fold_emb_dim * 3]
        else:
            # [b, max_labels, 3] - average/sum/transformer
            mask = batch.get("cath_code_indices_mask")
            if mask is None:
                mask = torch.zeros(idx.shape[:2], dtype=torch.bool, device=idx.device)
            fold_emb = torch.cat(
                [
                    self.embedding_C(idx[:, :, 0]),
                    self.embedding_A(idx[:, :, 1]),
                    self.embedding_T(idx[:, :, 2]),
                ],
                dim=-1,
            )  # [b, max_labels, fold_emb_dim * 3]
            if self.multilabel_mode == "transformer":
                fold_emb = self.transformer(
                    fold_emb, src_key_padding_mask=mask
                )  # [b, max_labels, fold_emb_dim * 3]
            # Aggregate over valid (non-masked) labels
            valid = ~mask  # [b, max_labels]
            if self.multilabel_mode == "transformer":
                fold_emb = (fold_emb * valid[:, :, None].float()).sum(dim=1) / (
                    valid.sum(dim=1, keepdim=True).float().clamp(min=1e-10)
                )
            elif self.multilabel_mode == "average":
                fold_emb = (fold_emb * valid[:, :, None].float()).sum(dim=1) / (
                    valid.sum(dim=1, keepdim=True).float().clamp(min=1e-10)
                )
            elif self.multilabel_mode == "sum":
                fold_emb = (fold_emb * valid[:, :, None].float()).sum(dim=1)
        fold_emb = fold_emb[:, None, :]  # [b, 1, fold_emb_dim * 3]
        return fold_emb.expand(
            (fold_emb.shape[0], n, fold_emb.shape[2])
        )  # [b, n, fold_emb_dim * 3]


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class ResidueTypeEmbeddingSeqFeat(Feature):
    """Computes sequence embedding and returns sequence feature of shape [b, n, seq_emb_dim]."""

    def __init__(self, seq_emb_dim, **kwargs):
        super().__init__(dim=seq_emb_dim)
        self.embedding = torch.nn.Embedding(21, seq_emb_dim)

    def forward(self, batch):
        seq = batch["residue_type"]  # [b, n]
        n = seq.shape[1]
        seq_emb = self.embedding(seq)  # [b, n, seq_emb_dim]
        return seq_emb  # [b, n, seq_emb_dim]


class ResidueTypeEmbeddingPairFeat(Feature):
    """Computes sequence embedding and returns pair feature of shape [b, n, n, seq_emb_dim]."""

    def __init__(self, seq_emb_dim, **kwargs):
        super().__init__(dim=seq_emb_dim)
        self.embedding_1 = torch.nn.Embedding(21, seq_emb_dim)
        self.embedding_2 = torch.nn.Embedding(21, seq_emb_dim)

    def forward(self, batch):
        seq = batch["residue_type"]  # [b, n]
        n = seq.shape[1]
        seq_emb_1 = self.embedding_1(seq)  # [b, n, seq_emb_dim]
        seq_emb_2 = self.embedding_2(seq)  # [b, n, seq_emb_dim]
        return seq_emb_1[:, :, None, :] * seq_emb_2[:, None, :, :]  # [b, n, n, seq_emb_dim]


class ExtLigEmbeddingSeqFeat(Feature):
    """Embeds ext_lig per-residue labels (0=absent, 1=present, 2=unknown) as [b, n, ext_lig_emb_dim]."""

    def __init__(self, ext_lig_emb_dim, **kwargs):
        super().__init__(dim=ext_lig_emb_dim)
        self.embedding = torch.nn.Embedding(3, ext_lig_emb_dim)

    def forward(self, batch):
        if "ext_lig" in batch:
            # nn.Embedding expects long indices; ext_lig may be stored as int8 on disk
            return self.embedding(batch["ext_lig"].to(torch.long))  # [b, n, ext_lig_emb_dim]
        xt = batch["x_t"]  # [b, n, 3]
        # Default to all unknown (index 2) when ext_lig is missing
        unknown = torch.full(
            (xt.shape[0], xt.shape[1]), 2, dtype=torch.long, device=xt.device
        )
        return self.embedding(unknown)


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = (
                torch.arange(1, n + 1, device=xt.device)
                .unsqueeze(0)
                .expand(b, -1)
                .float()
            )  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        # If it has the actual chain breaks
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            chain_breaks = torch.zeros((b, n), device=xt.device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscSeqFeat(Feature):
    """Computes feature from self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        if "x_sc" in batch:
            return batch["x_sc"]  # [b, n, 3]
        else:
            # If we do not provide self-conditioning as input to the nn
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            return torch.zeros(b, n, 3, device=x.device)


class MotifX1SeqFeat(Feature):
    """Computes feature from motif coordinates if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        # Support x_motif (motif_factory) and motif_structure (GenMotifDataset)
        x_motif = batch.get("x_motif", batch.get("motif_structure"))
        if x_motif is not None:
            return x_motif  # [b, n, 3]
        else:
            # If no motif
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            device = x.device
            return torch.zeros(b, n, 3, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes feature from mask of the motif positions if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        # Support motif_mask (flow), fixed_sequence_mask (motif_factory), motif_seq_mask (GenMotifDataset)
        mask = batch.get("motif_mask", batch.get("fixed_sequence_mask", batch.get("motif_seq_mask")))
        if mask is not None:
            return mask.unsqueeze(-1)  # [b, n, 1]
        x = batch["x_t"]
        b, n = x.shape[0], x.shape[1]
        device = x.device
        return torch.zeros(b, n, device=device).unsqueeze(-1)


class MotifStructureMaskFeat(Feature):
    """Computes feature of the pair wise motif mask of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "fixed_structure_mask" in batch:
            mask = batch["fixed_structure_mask"].unsqueeze(-1)  # [b, n, n, 1]
        elif "motif_seq_mask" in batch:
            m = batch["motif_seq_mask"]  # [b, n]
            mask = (m[:, :, None] * m[:, None, :]).unsqueeze(-1)  # [b, n, n, 1]
        else:
            raise ValueError("No fixed_structure_mask or motif_seq_mask in batch")
        return mask


class MotifX1PairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms of motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""
    def __init__(
        self, x_motif_pair_dist_dim, x_motif_pair_dist_min, x_motif_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_motif_pair_dist_dim)
        self.min_dist = x_motif_pair_dist_min
        self.max_dist = x_motif_pair_dist_max
        
    def forward(self, batch):
        # Support x_motif/motif_structure and fixed_structure_mask/motif_seq_mask
        x_motif = batch.get("x_motif", batch.get("motif_structure"))
        if x_motif is None:
            raise ValueError("No x_motif or motif_structure in batch")
        fs_mask = batch.get("fixed_structure_mask")
        if fs_mask is None and "motif_seq_mask" in batch:
            m = batch["motif_seq_mask"]
            fs_mask = m[:, :, None] * m[:, None, :]
        if fs_mask is None:
            raise ValueError("No fixed_structure_mask or motif_seq_mask in batch")
        return bin_pairwise_distances(
            x=x_motif,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        ) * fs_mask.unsqueeze(-1)  # [b, n, n, pair_dist_dim]


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = torch.arange(1, n + 1, device=xt.device, dtype=torch.float32).expand(b, -1)  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
            self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        return bin_pairwise_distances(
            x=batch["x_t"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class XscPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self, x_sc_pair_dist_dim, x_sc_pair_dist_min, x_sc_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max

    def forward(self, batch):
        if "x_sc" in batch:
            return bin_pairwise_distances(
                x=batch["x_sc"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            return torch.zeros(b, n, n, self.dim, device=x.device)


class ContactMapScPairFeat(Feature):
    """Embeds self-conditioned contact map prediction as a pair feature.

    Takes the self-conditioning contact map from batch["contact_map_sc"] and projects it.
    """

    def __init__(
        self,
        contact_map_sc_embed_dim: int = 64,
        **kwargs
    ):
        """Initialize the self-conditioning contact map pair feature.

        Args:
            contact_map_sc_embed_dim: Output dimension for the embedding.
        """
        super().__init__(dim=contact_map_sc_embed_dim)
        self.contact_map_sc_embed_dim = contact_map_sc_embed_dim
        self.linear_embed = torch.nn.Linear(1, contact_map_sc_embed_dim, bias=False)

    def forward(self, batch):
        """Extract and embed the self-conditioning contact map.

        Args:
            batch: Dictionary containing "contact_map_sc" of shape [b, n, n]
                   and "x_t" for shape/device inference.

        Returns:
            Embedded contact map of shape [b, n, n, dim]
        """
        if "contact_map_sc" in batch:
            contact_map_sc = batch["contact_map_sc"]  # [b, n, n]
            contact_map_expanded = contact_map_sc.unsqueeze(-1)  # [b, n, n, 1]
            return self.linear_embed(contact_map_expanded)  # [b, n, n, dim]
        else:
            # If no self-conditioning, return zeros
            x_t = batch["x_t"]  # [b, n, 3]
            b, n = x_t.shape[0], x_t.shape[1]
            return torch.zeros(b, n, n, self.dim, device=x_t.device, dtype=x_t.dtype)


####################################
# # Class that produces features # #
####################################


def _get_feature_creator(f: str, mode: Literal["seq", "pair"], **kwargs) -> Feature:
    """Returns the Feature instance for the requested feature name. Shared by FeatureFactory and IndividualFeatureFactory."""
    if mode == "seq":
        if f == "time_emb":
            return TimeEmbeddingSeqFeat(**kwargs)
        elif f == "res_seq_pdb_idx":
            return IdxEmbeddingSeqFeat(**kwargs)
        elif f == "chain_break_per_res":
            return ChainBreakPerResidueSeqFeat(**kwargs)
        elif f == "fold_emb":
            return FoldEmbeddingSeqFeat(**kwargs)
        elif f == "x_sc":
            return XscSeqFeat(**kwargs)
        elif f == "motif_x1":
            return MotifX1SeqFeat(**kwargs)
        elif f == "motif_sequence_mask":
            return MotifMaskSeqFeat(**kwargs)
        elif f == "residue_type_emb":
            return ResidueTypeEmbeddingSeqFeat(**kwargs)
        elif f == "ext_lig_emb":
            return ExtLigEmbeddingSeqFeat(**kwargs)
        else:
            raise IOError(f"Sequence feature {f} not implemented.")
    elif mode == "pair":
        if f == "xt_pair_dists":
            return XtPairwiseDistancesPairFeat(**kwargs)
        elif f == "x_sc_pair_dists":
            return XscPairwiseDistancesPairFeat(**kwargs)
        elif f == "rel_seq_sep":
            return SequenceSeparationPairFeat(**kwargs)
        elif f == "time_emb":
            return TimeEmbeddingPairFeat(**kwargs)
        elif f == "motif_x1_pair_dists":
            return MotifX1PairwiseDistancesPairFeat(**kwargs)
        elif f == "motif_structure_mask":
            return MotifStructureMaskFeat(**kwargs)
        elif f == "residue_type_emb":
            return ResidueTypeEmbeddingPairFeat(**kwargs)
        elif f == "contact_map_sc":
            return ContactMapScPairFeat(**kwargs)
        else:
            raise IOError(f"Pair feature {f} not implemented.")
    else:
        raise IOError(f"Wrong feature mode: {mode}. Should be 'seq' or 'pair'.")


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        use_residue_type_emb: bool = False,
        use_ext_lig_emb: bool = False,
        feature_embedding_mode: Literal["concat", "individual"] = "concat",
        individual_feat_ln: bool = True,
        **kwargs,
    ):
        """
        Sequence features include:
            - "res_seq_pdb_idx", requires transform ResidueSequencePositionPdbTransform
            - "time_emb"
            - "chain_break_per_res", requires transform ChainBreakPerResidueTransform
            - "fold_emb"
            - "x_sc"

        Pair features include:
            - "xt_pair_dists"
            - "x_sc_pair_dists"
            - "rel_seq_sep"
            - "time_emb"
        """
        super().__init__()
        self.mode = mode
        self.use_residue_type_emb = use_residue_type_emb
        self.use_ext_lig_emb = use_ext_lig_emb and mode == "seq"
        self.feature_embedding_mode = kwargs.pop("feature_embedding_mode", feature_embedding_mode)
        self._individual_feat_ln = kwargs.get("individual_feat_ln", individual_feat_ln)

        if self.feature_embedding_mode == "individual":
            _feat_kwargs = {k: v for k, v in kwargs.items() if k not in ("individual_feat_ln",)}
            self._individual_factory = IndividualFeatureFactory(
                feats=feats,
                dim_feats_out=dim_feats_out,
                use_ln_out=use_ln_out,
                mode=mode,
                use_residue_type_emb=use_residue_type_emb,
                use_ext_lig_emb=self.use_ext_lig_emb,
                individual_feat_ln=self._individual_feat_ln,
                **_feat_kwargs,
            )
            return

        # Disable sinusoidal positional embedding (res_seq_pdb_idx) when idx_emb_dim is 0 or not specified
        if feats is not None and mode == "seq":
            idx_emb_dim = kwargs.get("idx_emb_dim", 0)
            if idx_emb_dim <= 0:
                feats = [f for f in feats if f != "res_seq_pdb_idx"]

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [_get_feature_creator(f, self.mode, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False
        )
        if self.use_residue_type_emb:
            self.residue_type_feat_creator = _get_feature_creator("residue_type_emb", self.mode, **kwargs)
            self.residue_type_out = torch.nn.Linear(
                self.residue_type_feat_creator.get_dim(), dim_feats_out, bias=False
            )
        if self.use_ext_lig_emb:
            self.ext_lig_feat_creator = _get_feature_creator("ext_lig_emb", "seq", **kwargs)
            self.ext_lig_out = torch.nn.Linear(
                self.ext_lig_feat_creator.get_dim(), dim_feats_out, bias=False
            )
    
    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        if self.feature_embedding_mode == "individual":
            return self._individual_factory(batch)
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_out = self.linear_out(features)
        if self.use_residue_type_emb and "residue_type" in batch:
            features_out += self.residue_type_out(
                self.residue_type_feat_creator(batch)
                )
        if self.use_ext_lig_emb:
            features_out = features_out + self.ext_lig_out(
                self.ext_lig_feat_creator(batch)
            )
        features_proc = self.ln_out(
            features_out
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]


class IndividualFeatureFactory(torch.nn.Module):
    """Projects each feature individually to dim_feats_out and sums them.
    Mirrors residue_type_emb pattern: each modality gets its own Linear projection."""

    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        use_residue_type_emb: bool = False,
        use_ext_lig_emb: bool = False,
        individual_feat_ln: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.use_residue_type_emb = use_residue_type_emb
        self.use_ext_lig_emb = use_ext_lig_emb and mode == "seq"
        self.individual_feat_ln = individual_feat_ln
        kwargs = {k: v for k, v in kwargs.items() if k not in ("feature_embedding_mode", "individual_feat_ln")}

        # Disable sinusoidal positional embedding (res_seq_pdb_idx) when idx_emb_dim is 0 or not specified
        if feats is not None and mode == "seq":
            idx_emb_dim = kwargs.get("idx_emb_dim", 0)
            if idx_emb_dim <= 0:
                feats = [f for f in feats if f != "res_seq_pdb_idx"]

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        self.dim_feats_out = dim_feats_out
        if self.ret_zero:
            logger.info("No features requested (IndividualFeatureFactory)")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_names = feats
        self.feat_creators = torch.nn.ModuleList(
            [_get_feature_creator(f, self.mode, **kwargs) for f in feats]
        )
        self.projections = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(creator.get_dim(), dim_feats_out, bias=False)
                for name, creator in zip(feats, self.feat_creators)
            }
        )
        if self.individual_feat_ln:
            self.feat_layer_norms = torch.nn.ModuleDict(
                {
                    name: torch.nn.LayerNorm(dim_feats_out)
                    for name in feats
                }
            )
        else:
            self.feat_layer_norms = None

        if self.use_residue_type_emb:
            self.residue_type_feat_creator = _get_feature_creator(
                "residue_type_emb", self.mode, **kwargs
            )
            self.residue_type_out = torch.nn.Linear(
                self.residue_type_feat_creator.get_dim(), dim_feats_out, bias=False
            )
        if self.use_ext_lig_emb:
            self.ext_lig_feat_creator = _get_feature_creator(
                "ext_lig_emb", "seq", **kwargs
            )
            self.ext_lig_out = torch.nn.Linear(
                self.ext_lig_feat_creator.get_dim(), dim_feats_out, bias=False
            )

        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )

    def _apply_padding_mask(self, feature_tensor, mask):
        if self.mode == "seq":
            return feature_tensor * mask[..., None]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]
            return feature_tensor * mask_pair[..., None]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        if self.ret_zero:
            return self.zero_creator(batch)

        xt = batch["x_t"]
        b, n = xt.shape[0], xt.shape[1]
        if self.mode == "seq":
            output = torch.zeros((b, n, self.dim_feats_out), device=xt.device, dtype=xt.dtype)
        else:
            output = torch.zeros((b, n, n, self.dim_feats_out), device=xt.device, dtype=xt.dtype)

        target_dtype = xt.dtype
        for name, creator in zip(self.feat_names, self.feat_creators):
            raw = creator(batch)
            if raw.dtype != target_dtype:
                raw = raw.to(target_dtype)
            proj = self.projections[name](raw)
            if self.feat_layer_norms is not None:
                proj = self.feat_layer_norms[name](proj)
            output = output + proj

        if self.use_residue_type_emb and "residue_type" in batch:
            residue_feat = self.residue_type_feat_creator(batch)
            if residue_feat.dtype != target_dtype:
                residue_feat = residue_feat.to(target_dtype)
            output = output + self.residue_type_out(residue_feat)

        if self.use_ext_lig_emb:
            ext_lig_feat = self.ext_lig_feat_creator(batch)
            if ext_lig_feat.dtype != target_dtype:
                ext_lig_feat = ext_lig_feat.to(target_dtype)
            output = output + self.ext_lig_out(ext_lig_feat)

        output = self._apply_padding_mask(output, batch["mask"])
        output = self.ln_out(output)
        return self._apply_padding_mask(output, batch["mask"])
