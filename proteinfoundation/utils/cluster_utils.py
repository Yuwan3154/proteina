# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# inspired from https://github.com/a-r-j/graphein/blob/master/graphein/ml/datasets/pdb_data.py

import math
import pathlib
# import random  # Removed to use torch.randint for reproducibility
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Literal, Tuple

import pandas as pd
import torch
import torch_geometric
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from torch.utils.data import Sampler


@rank_zero_only
def log_info(msg):
    logger.info(msg)


class ClusterSampler(Sampler):
    def __init__(
        self,
        dataset: torch_geometric.data.Dataset,
        clusterid_to_seqid_mapping: Dict[str, List[str]],
        sampling_mode: Literal["cluster-random", "cluster-reps"],
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        v2: bool = True,
    ):
        """
        Initializes the ClusterSampler for selecting sequences during training.

        Args:
            dataset (torch_geometric.data.Dataset): The dataset object.
            clusterid_to_seqid_mapping (Dict[str, List[str]]): Dictionary holding cluster names and corresponding sequence IDs.
            sampling_mode (Literal["cluster-random", "cluster-reps"]): The sampling mode to use.
                - "cluster-random": Select a random sequence from each cluster.
                - "cluster-reps": Select the representative sequence from each cluster.
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the indices.
            drop_last (bool, optional): If ``True``, then the sampler will drop the tail of the data to make it
                evenly divisible across the number of replicas. If ``False``, the sampler will add extra indices to
                make the data evenly divisible across the replicas. Default: ``False``.
            seed (int): Base seed for the per-epoch torch.Generator. Combined with the
                Lightning-supplied epoch via ``set_epoch`` to derive deterministic-but-distinct
                shuffles. Default 0.
            v2 (bool): When True (default), use seeded torch.Generator instances for the
                cluster-order shuffle (synchronized across DDP ranks → correct partitioning)
                and per-cluster member draw (rank-distinct → no global RNG contamination).
                Set False to fall back to the legacy global-RNG behaviour for comparison.
        """
        self.dataset = dataset
        self.clusterid_to_seqid_mapping = clusterid_to_seqid_mapping
        self.cluster_names = list(clusterid_to_seqid_mapping.keys())
        self.sampling_mode = sampling_mode
        if dataset.database == "pdb" or dataset.database == "scop":  # PDBDataset
            self.sequence_id_to_idx = {
                fname.split(".")[0]: i for i, fname in enumerate(dataset.file_names)
            }
        elif dataset.database == "pinder":
            self.sequence_id_to_idx = dataset.pinder_id_to_idx
        else:  # FoldCompDataset
            self.sequence_id_to_idx = dataset.protein_to_idx
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.log_clusters = True
        self.num_replicas = None
        self.seed = int(seed)
        self.v2 = bool(v2)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Lightning calls this each train epoch start; used to seed the per-epoch generators."""
        self.epoch = int(epoch)

    def _make_generator(self, *parts: int) -> torch.Generator:
        """Build a fresh torch.Generator seeded by (self.seed, self.epoch, *parts).

        Mixed via simple 64-bit combine to avoid pulling in numpy. Same inputs across
        ranks yields the same generator state — this is the property the cluster-order
        shuffle relies on for correct DDP partitioning.
        """
        s = (self.seed & 0xFFFFFFFF)
        for p in (self.epoch, *parts):
            s = ((s * 0x9E3779B97F4A7C15) ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        # torch.Generator.manual_seed wants a non-negative 64-bit int.
        g = torch.Generator()
        g.manual_seed(s if s != 0 else 1)
        return g

    def __iter__(self):
        """Iterate over clusters in dataset and yield samples depending on sampling_mode."""
        # set logging to true so that first sample in epoche gets logged
        self.log_clusters = True
        # setup distributed/non-distributed backend
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = None
            self.rank = 0
            logger.info(
                f"Distributed sampler is not initialized, assuming single-device setup."
            )

        # Build a rank-synchronized generator for cluster-order shuffling.
        # The same (seed, epoch) input on every rank yields an identical permutation,
        # so `indices[rank::num_replicas]` gives a clean partition.
        if self.v2:
            shuffle_gen = self._make_generator(0)  # 0 = "shuffle stream"
        else:
            shuffle_gen = None

        if self.num_replicas is not None:
            self.num_samples = math.ceil(
                len(self.cluster_names) * 1.0 / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
            # Distributed mode, deterministically shuffle
            if self.v2:
                indices = torch.randperm(len(self.cluster_names), generator=shuffle_gen).tolist()
            else:
                indices = torch.randperm(len(self.cluster_names)).tolist()

            # drop samples to make it evenly divisible
            if self.drop_last:
                indices_to_keep = self.total_size - self.num_replicas
                indices = indices[:indices_to_keep]
            # add extra samples to make it evenly divisible
            else:
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]

            # subsample
            indices = indices[self.rank : self.total_size : self.num_replicas]
            # Per-cluster member generator: rank-distinct so members differ across DDP
            # workers (which is fine — each rank owns a disjoint cluster slice).
            member_gen = self._make_generator(1, self.rank) if self.v2 else None
            if self.sampling_mode == "cluster-reps":
                # Assumes that cluster_names are the IDs of the representative (longest) sequences (true for mmseqs2 clusters)
                for cluster_name_idx in indices:
                    cluster_name = self.cluster_names[cluster_name_idx]
                    yield self.sequence_id_to_idx[cluster_name]
            elif self.sampling_mode == "cluster-random":
                for cluster_name_idx in indices:
                    cluster_name = self.cluster_names[cluster_name_idx]
                    sequences = self.clusterid_to_seqid_mapping[cluster_name]
                    if self.v2:
                        idx = torch.randint(0, len(sequences), (1,), generator=member_gen).item()
                    else:
                        idx = torch.randint(0, len(sequences), (1,)).item()
                    sequence_id = sequences[idx]
                    if self.log_clusters:
                        # log first sampling
                        logger.info(
                            f"First cluster sampling: sampling {sequence_id} from cluster {cluster_name}, rank {self.rank}"
                        )
                        self.log_clusters = False
                    yield self.sequence_id_to_idx[sequence_id]
            else:
                raise ValueError(
                    f"Unknown cluster sampling mode {self.sampling_mode} for ClusterSampler, only 'cluster-random' and 'cluster-reps' supported"
                )
        else:
            # Non-distributed mode. Build a local copy of the cluster ordering per epoch
            # so we don't mutate self.cluster_names (legacy code did, which compounded
            # shuffles across epochs).
            if self.shuffle:
                if self.v2:
                    perm = torch.randperm(len(self.cluster_names), generator=shuffle_gen).tolist()
                else:
                    perm = torch.randperm(len(self.cluster_names)).tolist()
                ordered_cluster_names = [self.cluster_names[i] for i in perm]
            else:
                ordered_cluster_names = list(self.cluster_names)
            member_gen = self._make_generator(1, 0) if self.v2 else None
            if self.sampling_mode == "cluster-reps":
                # Assumes that cluster_names are the IDs of the representative (longest) sequences (true for mmseqs2 clusters)
                for cluster_name in ordered_cluster_names:
                    yield self.sequence_id_to_idx[cluster_name]
            elif self.sampling_mode == "cluster-random":
                for cluster_name in ordered_cluster_names:
                    sequences = self.clusterid_to_seqid_mapping[cluster_name]
                    if self.v2:
                        idx = torch.randint(0, len(sequences), (1,), generator=member_gen).item()
                    else:
                        idx = torch.randint(0, len(sequences), (1,)).item()
                    sequence_id = sequences[idx]
                    if self.log_clusters:
                        # log first sampling
                        logger.info(
                            f"First cluster sampling: sampling {sequence_id} from cluster {cluster_name}"
                        )
                        self.log_clusters = False
                    yield self.sequence_id_to_idx[sequence_id]
            else:
                raise ValueError(
                    f"Unknown cluster sampling mode {self.sampling_mode} for ClusterSampler, only 'cluster-random' and 'cluster-reps' supported"
                )

    def __len__(self):
        if self.num_replicas is not None:
            return self.num_samples
        else:
            return len(self.cluster_names)


class CATBalancedSampler(Sampler):
    """CAT-balanced epoch sampler layered on top of sequence-similarity clusters.

    Each epoch presents exactly one chain per CAT family (CATH 3-level
    Class.Architecture.Topology). Per the design:
      1. Epoch length = number of distinct CAT topologies in the (split) mapping.
      2. For each topology, draw a seq-sim cluster that contains it
         (cat_cluster_draw: "random" | "largest").
      3. From the drawn cluster, draw a chain (member_mode: "cluster-random" |
         "cluster-reps"), mirroring ClusterSampler's member selection.

    Clusters are REUSED (no re-clustering) with multi-membership: a cluster is
    registered under every CAT present among its labeled members. Unlabeled
    members are infilled only in PURE clusters (single CAT among labeled members).
    Clusters with zero labeled members form a separate no-CAT bucket, sampled via
    nocat_bucket_draws extra draws per epoch (per rank under DDP).

    DDP: topology order is shuffled with a rank-synchronized generator and padded
    to an equal per-rank count (mirrors ClusterSampler) so every rank yields the
    same number of samples; per-rank draws use a rank-distinct generator.
    """

    def __init__(
        self,
        dataset: torch_geometric.data.Dataset,
        clusterid_to_seqid_mapping: Dict[str, List[str]],
        chain_to_cat: Dict[str, List[str]],
        cat_cluster_draw: Literal["random", "largest"] = "largest",
        member_mode: Literal["cluster-random", "cluster-reps"] = "cluster-random",
        nocat_bucket: bool = True,
        nocat_bucket_draws: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        v2: bool = True,
    ):
        self.dataset = dataset
        self.clusterid_to_seqid_mapping = clusterid_to_seqid_mapping
        self.cluster_names = list(clusterid_to_seqid_mapping.keys())
        self.chain_to_cat = chain_to_cat
        self.cat_cluster_draw = cat_cluster_draw
        self.member_mode = member_mode
        self.nocat_bucket = bool(nocat_bucket)
        self.nocat_bucket_draws = int(nocat_bucket_draws)
        if dataset.database == "pdb" or dataset.database == "scop":
            self.sequence_id_to_idx = {
                fname.split(".")[0]: i for i, fname in enumerate(dataset.file_names)
            }
        elif dataset.database == "pinder":
            self.sequence_id_to_idx = dataset.pinder_id_to_idx
        else:
            self.sequence_id_to_idx = dataset.protein_to_idx
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.log_first = True
        self.num_replicas = None
        self.rank = 0
        self.seed = int(seed)
        self.v2 = bool(v2)
        self.epoch = 0
        self._build_cat_index()

    def _build_cat_index(self) -> None:
        """Build the CAT -> cluster -> member indices once at construction.

        cluster_cats: cluster -> set of CAT codes (union over labeled members).
        eligible: (cluster, cat) -> member seqids carrying cat (+ infill in pure clusters).
        rep_for_T: (cluster, cat) -> representative seqid for member_mode='cluster-reps'.
        cluster_size_for_T: (cluster, cat) -> len(eligible) for cat_cluster_draw='largest'.
        topology_to_clusters: cat -> [clusters carrying it].
        nocat_clusters: clusters with zero labeled members.
        """
        cluster_cats: Dict[str, set] = {}
        eligible: Dict[Tuple[str, str], List[str]] = {}
        rep_for_T: Dict[Tuple[str, str], str] = {}
        topology_to_clusters: Dict[str, List[str]] = {}
        nocat_clusters: List[str] = []

        for cluster_name, members in self.clusterid_to_seqid_mapping.items():
            # Exact-case first (AFDB v6 stems are case-sensitive), then lowercased
            # (PDB chain_to_cat keys are lowercased); supports both datasets.
            member_cats = {m: (self.chain_to_cat.get(m) or self.chain_to_cat.get(m.lower(), [])) for m in members}
            cats_here = set()
            for cl in member_cats.values():
                cats_here.update(cl)
            cluster_cats[cluster_name] = cats_here
            if not cats_here:
                nocat_clusters.append(cluster_name)
                continue
            pure = len(cats_here) == 1
            unlabeled = [m for m in members if not member_cats[m]] if pure else []
            rep_cats = member_cats.get(cluster_name, [])  # rep seqid == cluster_name (mmseqs rep)
            for cat in cats_here:
                elig = [m for m in members if cat in member_cats[m]]
                if pure:
                    elig = elig + unlabeled  # infill the whole pure cluster to its single CAT
                eligible[(cluster_name, cat)] = elig
                topology_to_clusters.setdefault(cat, []).append(cluster_name)
                if cat in rep_cats:
                    rep_for_T[(cluster_name, cat)] = cluster_name
                else:
                    rep_for_T[(cluster_name, cat)] = elig[0]

        self.cluster_cats = cluster_cats
        self.eligible = eligible
        self.rep_for_T = rep_for_T
        self.cluster_size_for_T = {k: len(v) for k, v in eligible.items()}
        self.topology_to_clusters = topology_to_clusters
        self.nocat_clusters = nocat_clusters
        self._topologies = sorted(topology_to_clusters.keys())
        log_info(
            f"CATBalancedSampler: {len(self._topologies)} CAT topologies, "
            f"{len(self.cluster_names)} clusters ({len(nocat_clusters)} no-CAT), "
            f"cat_cluster_draw={self.cat_cluster_draw}, member_mode={self.member_mode}, "
            f"nocat_bucket={self.nocat_bucket} (draws={self.nocat_bucket_draws})"
        )

    def set_epoch(self, epoch: int) -> None:
        """Lightning calls this each train epoch start; seeds the per-epoch generators."""
        self.epoch = int(epoch)

    def _make_generator(self, *parts: int) -> torch.Generator:
        """Fresh torch.Generator seeded by (seed, epoch, *parts); same inputs -> same state across ranks."""
        s = (self.seed & 0xFFFFFFFF)
        for p in (self.epoch, *parts):
            s = ((s * 0x9E3779B97F4A7C15) ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
        g = torch.Generator()
        g.manual_seed(s if s != 0 else 1)
        return g

    def _draw_for_topology(self, cat: str, gen) -> int:
        clusters = self.topology_to_clusters[cat]
        if self.cat_cluster_draw == "largest":
            # max eligible size; deterministic tie-break by cluster name
            chosen = sorted(clusters, key=lambda c: (-self.cluster_size_for_T[(c, cat)], c))[0]
        elif self.cat_cluster_draw == "random":
            if self.v2:
                j = torch.randint(0, len(clusters), (1,), generator=gen).item()
            else:
                j = torch.randint(0, len(clusters), (1,)).item()
            chosen = clusters[j]
        else:
            raise ValueError(
                f"Unknown cat_cluster_draw {self.cat_cluster_draw}, expected 'random' or 'largest'"
            )
        if self.member_mode == "cluster-reps":
            seqid = self.rep_for_T[(chosen, cat)]
        elif self.member_mode == "cluster-random":
            elig = self.eligible[(chosen, cat)]
            if self.v2:
                k = torch.randint(0, len(elig), (1,), generator=gen).item()
            else:
                k = torch.randint(0, len(elig), (1,)).item()
            seqid = elig[k]
        else:
            raise ValueError(
                f"Unknown member_mode {self.member_mode}, expected 'cluster-random' or 'cluster-reps'"
            )
        if self.log_first:
            logger.info(
                f"First CAT sampling: topology {cat} -> cluster {chosen} -> {seqid}, rank {self.rank}"
            )
            self.log_first = False
        return self.sequence_id_to_idx[seqid]

    def _draw_nocat(self, gen) -> List[int]:
        out: List[int] = []
        if not self.nocat_clusters:
            return out
        for _ in range(self.nocat_bucket_draws):
            if self.v2:
                j = torch.randint(0, len(self.nocat_clusters), (1,), generator=gen).item()
            else:
                j = torch.randint(0, len(self.nocat_clusters), (1,)).item()
            cluster_name = self.nocat_clusters[j]
            if self.member_mode == "cluster-reps":
                seqid = cluster_name
            else:
                members = self.clusterid_to_seqid_mapping[cluster_name]
                if self.v2:
                    k = torch.randint(0, len(members), (1,), generator=gen).item()
                else:
                    k = torch.randint(0, len(members), (1,)).item()
                seqid = members[k]
            out.append(self.sequence_id_to_idx[seqid])
        return out

    def __iter__(self):
        self.log_first = True
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = None
            self.rank = 0
            logger.info("Distributed sampler is not initialized, assuming single-device setup.")

        n = len(self._topologies)
        shuffle_gen = self._make_generator(0) if self.v2 else None
        if self.shuffle:
            if self.v2:
                perm = torch.randperm(n, generator=shuffle_gen).tolist()
            else:
                perm = torch.randperm(n).tolist()
        else:
            perm = list(range(n))

        if self.num_replicas is not None:
            self.num_samples = math.ceil(n * 1.0 / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
            if self.drop_last:
                perm = perm[: self.total_size - self.num_replicas]
            else:
                padding_size = self.total_size - len(perm)
                if padding_size <= len(perm):
                    perm += perm[:padding_size]
                else:
                    perm += (perm * math.ceil(padding_size / len(perm)))[:padding_size]
            perm = perm[self.rank : self.total_size : self.num_replicas]
            draw_gen = self._make_generator(1, self.rank) if self.v2 else None
        else:
            draw_gen = self._make_generator(1, 0) if self.v2 else None

        indices = [self._draw_for_topology(self._topologies[i], draw_gen) for i in perm]
        if self.nocat_bucket:
            indices.extend(self._draw_nocat(draw_gen))
        return iter(indices)

    def __len__(self):
        if self.num_replicas is not None:
            base = self.num_samples
        else:
            base = len(self._topologies)
        if self.nocat_bucket and self.nocat_clusters:
            base += self.nocat_bucket_draws
        return base


def split_dataframe(
    df: pd.DataFrame,
    splits: List[str],
    ratios: List[float],
    leftover_split: int = 0,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame into multiple parts based on specified split ratios.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        splits (List[str]): Names of the resulting splits.
        ratios (List[float]): Ratios to split df into. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover rows to.
            Defaults to 0.
        seed (int): Random seed for shuffling. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to
            DataFrame splits.

    Raises:
        AssertionError: If len(splits) != len(ratios) or sum(ratios) != 1.
    """
    assert len(splits) == len(ratios), "Number of splits must equal number of ratios"
    assert sum(ratios) == 1, "Split ratios must sum to 1"

    # Calculate size of each split
    split_sizes = [int(len(df) * ratio) for ratio in ratios]

    # Assign leftover rows to specified split
    split_sizes[leftover_split] += len(df) - sum(split_sizes)

    # Shuffle DataFrame rows
    df = df.sample(frac=1, random_state=seed)

    # Split DataFrame into parts
    split_dfs = {}
    start = 0
    for split, size in zip(splits, split_sizes):
        split_dfs[split] = df.iloc[start : start + size]
        start += size

    return split_dfs


def merge_dataframe_splits(
    df1: pd.DataFrame, df2: pd.DataFrame, list_columns: List[str]
) -> pd.DataFrame:
    """
    Merge two DataFrame splits on all columns except 'split'.

    Args:
        df1 (pd.DataFrame): First DataFrame split to merge.
        df2 (pd.DataFrame): Second DataFrame split to merge.
        list_columns (List[str]): Columns containing lists to convert to tuples.

    Returns:
        pd.DataFrame: Merged DataFrame containing rows in both splits.
    """
    # Convert list columns to tuples for merging
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(tuple)

    # Merge the two DataFrames
    merge_cols = [c for c in df1.columns if c != "split"]
    merged_df = pd.merge(df1, df2, on=merge_cols, how="inner")

    # Convert tuple columns back to lists
    for df in [df1, df2]:
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(list)

    return merged_df


def cluster_sequences(
    fasta_input_filepath: str,
    cluster_output_filepath: str = None,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    overwrite: bool = False,
    silence_mmseqs_output: bool = True,
    efficient_linclust: bool = False,
) -> None:
    """
    Cluster protein sequences in a DataFrame using MMseqs2.

    Args:
        fasta_input_file (str): Fasta File path containing protein sequences.
        cluster_output_filepath (str): Path to write clustering results. If None, defaults to
            "cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta".
        min_seq_id (float): Minimum sequence identity for clustering. Defaults to 0.3.
        coverage (float): Minimum coverage for clustering. Defaults to 0.8.
        overwrite (bool): Whether to overwrite existing cluster file. Defaults to False.
        silence_mmseqs_output (bool): Whether to silence MMseqs2 output. Defaults to True.
        efficient_linclust (bool): Whether to use efficient linclust for clustering for large datasets. Defaults to False.
    """
    if cluster_output_filepath is None:
        cluster_output_filepath = f"cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"

    cluster_fasta_path = pathlib.Path(cluster_output_filepath)
    cluster_tsv_path = cluster_fasta_path.with_suffix(".tsv")

    if not cluster_fasta_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_fasta_path.exists() and overwrite:
            cluster_fasta_path.unlink()

    if not cluster_tsv_path.exists() or overwrite:
        # Remove existing file if overwriting
        if cluster_tsv_path.exists() and overwrite:
            cluster_tsv_path.unlink()

        # Run MMseqs2 clustering
        if shutil.which("mmseqs") is None:
            logger.error(
                "MMseqs2 not found. Please install it: conda install -c conda-forge -c bioconda mmseqs2"
            )

        # Run mmseqs from data_dir so tmp/ and output files are created there (avoids CWD/tmp
        # permission issues on HPC/DDP where CWD may be unwritable)
        work_dir = cluster_fasta_path.parent
        fasta_abs = pathlib.Path(fasta_input_filepath).resolve()

        if efficient_linclust:
            cmd_parts = ["mmseqs", "easy-linclust", str(fasta_abs), "pdb_cluster", "tmp", "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "1"]
        else:
            cmd_parts = ["mmseqs", "easy-cluster", str(fasta_abs), "pdb_cluster", "tmp", "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "1"]
        run_kw = {"cwd": str(work_dir), "stderr": subprocess.PIPE}
        if silence_mmseqs_output:
            run_kw["stdout"] = subprocess.DEVNULL
        result = subprocess.run(cmd_parts, **run_kw)
        if result.returncode != 0:
            stderr = result.stderr.decode() if getattr(result, "stderr", None) else ""
            raise RuntimeError(
                f"mmseqs clustering failed (exit {result.returncode}). "
                f"Ensure tmp and output can be written in {work_dir}. stderr: {stderr}"
            )
        # Move output files (created in work_dir) to final paths
        rep_seq_src = work_dir / "pdb_cluster_rep_seq.fasta"
        cluster_tsv_src = work_dir / "pdb_cluster_cluster.tsv"
        if not rep_seq_src.exists():
            raise RuntimeError(
                f"mmseqs did not create pdb_cluster_rep_seq.fasta in {work_dir}. "
                "Check mmseqs stderr for errors (run with silence_mmseqs_output=False)."
            )
        shutil.move(str(rep_seq_src), cluster_fasta_path)
        shutil.move(str(cluster_tsv_src), cluster_tsv_path)


def split_sequence_clusters(
    df, splits, ratios, leftover_split=0, seed=42
) -> Dict[str, pd.DataFrame]:
    """
    Split clustered sequences into train/val/test sets.

    Args:
        df (pd.DataFrame): DataFrame with clustered sequences.
        splits (List[str]): Names of splits, e.g. ["train", "val", "test"].
        ratios (List[float]): Ratios for each split. Must sum to 1.0.
        leftover_split (int): Index of split to assign leftover sequences.
            Defaults to 0.
        seed (int): Random seed. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping split names to DataFrames that contain randomly-split representative sequences.
    """
    # Split clusters into subsets
    cluster_splits = split_dataframe(df, splits, ratios, leftover_split, seed)
    # Get representative sequences for each split
    split_dfs = {}
    for split, cluster_df in cluster_splits.items():
        rep_seqs = cluster_df.representative_sequences()
        split_dfs[split] = rep_seqs

    return split_dfs


def expand_cluster_splits(
    cluster_rep_splits: Dict[str, pd.DataFrame],
    clusterid_to_seqid_mapping: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Expand the cluster representative splits to full cluster splits based on the provided cluster dictionary.

    Args:
        cluster_rep_splits: A dictionary containing DataFrames for each split (e.g., 'train', 'val', 'test').
            Each DataFrame should have an 'id' column representing the cluster representative IDs.
        clusterid_to_seqid_mapping: A dictionary mapping cluster representative IDs to their corresponding cluster member IDs.

    Returns:
        A new dictionary of DataFrames with expanded 'id' columns based on the cluster dictionary.
        The 'id' column in the original DataFrames is replaced with the corresponding cluster member IDs.
        If df_sequences is provided, the additional columns from df_sequences are added to the resulting DataFrames.

    """
    full_cluster_splits = {}
    split_clusterid_to_seqid_mapping = {}

    for split_name, split_df in cluster_rep_splits.items():
        # Create a dictionary to store the cluster members for the current split
        split_cluster_members = {}

        for rep_id in split_df["id"]:
            if rep_id in clusterid_to_seqid_mapping:
                split_cluster_members[rep_id] = clusterid_to_seqid_mapping[rep_id]
            else:
                logger.warning(
                    f"ID {rep_id} is a representative in the splits, but not in the cluster_dicts"
                )

        # Create a DataFrame with the cluster representative IDs and their corresponding cluster member IDs for the current split
        split_cluster_members_df = pd.DataFrame(
            [
                (rep_id, member_id)
                for rep_id, member_ids in split_cluster_members.items()
                for member_id in member_ids
            ],
            columns=["cluster_id", "id"],
        )
        # Split the 'id' column into 'pdb' and 'chain' columns
        if len(split_cluster_members_df) > 0:
            split_cluster_members_df[["pdb", "chain"]] = split_cluster_members_df[
                "id"
            ].str.split("_", n=1, expand=True)
        # Add the expanded DataFrame to the dictionary
        full_cluster_splits[split_name] = split_cluster_members_df
        # Add the split-specific cluster_dict to the dictionary
        split_clusterid_to_seqid_mapping[split_name] = split_cluster_members
    return full_cluster_splits, split_clusterid_to_seqid_mapping


def read_cluster_tsv(cluster_tsv_filepath: pathlib.Path) -> Dict[str, List[str]]:
    """
    Read the cluster TSV file that is output from mmseqs2 and construct a dictionary mapping cluster representatives to sequence IDs.

    Args:
        cluster_tsv_filepath (pathlib.Path): The path to the cluster TSV file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping cluster representatives to lists of sequence IDs.
    """
    cluster_dict = {}
    with open(cluster_tsv_filepath, "r") as file:
        for line in file:
            cluster_name, sequence_name = line.strip().split("\t")
            cluster_dict.setdefault(cluster_name, []).append(sequence_name)
    return cluster_dict


def setup_clustering_file_paths(
    data_dir: str,
    file_identifier: str,
    split_sequence_similarity: float,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """
    Set up file paths for the fasta file, cluster file, and cluster TSV file.

    Args:
        data_dir (str): The directory where the files will be stored.
        file_identifier (str): The identifier used to name the files.
        split_sequence_similarity (float): The sequence similarity threshold for splitting.

    Returns:
        Tuple[pathlib.Path, pathlib.Path, pathlib.Path]: A tuple containing the file paths for
            the input fasta file, cluster file, and cluster TSV file.
    """
    input_fasta_filepath = pathlib.Path(data_dir) / f"seq_{file_identifier}.fasta"
    cluster_filepath = (
        pathlib.Path(data_dir)
        / f"cluster_seqid_{split_sequence_similarity}_{file_identifier}.fasta"
    )
    cluster_tsv_filepath = cluster_filepath.with_suffix(".tsv")
    return input_fasta_filepath, cluster_filepath, cluster_tsv_filepath


def df_to_fasta(df: pd.DataFrame, output_file: str) -> None:
    """
    Convert a pandas DataFrame to a FASTA file.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'sequence' columns.
        output_file (str): Path to the output FASTA file.

    Returns:
        None
    """
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")


def fasta_to_df(fasta_input_file: str) -> pd.DataFrame:
    """
    Convert a FASTA file to a pandas DataFrame.

    Args:
        fasta_input_file (str): Path to the input FASTA file.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and 'sequence' columns.
    """
    data = []
    with open(fasta_input_file, "r") as file:
        sequence_id = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if sequence_id is not None:
                    data.append([sequence_id, "".join(sequence)])
                sequence_id = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if sequence_id is not None:
            data.append([sequence_id, "".join(sequence)])

        df = pd.DataFrame(data, columns=["id", "sequence"])
    return df
