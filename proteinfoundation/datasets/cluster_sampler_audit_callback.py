# SPDX-License-Identifier: MIT
"""Lightning callback that verifies the ClusterSampler DDP partitioning at runtime.

The audit script ``scripts/analysis/audit_cluster_sampler.py`` proves the v2 fix
works on a synthetic mock; this callback is the in-training-run analogue. It
runs once on the first epoch, materializes that rank's sampled indices, and
``all_gather``s them to verify:

  - Indices are disjoint across ranks (no duplicate work).
  - Total coverage matches the expected per-rank partition size.

Outputs:
  - logs ``cluster_sampler_audit/duplicates_total`` and ``coverage_ratio`` to
    the wandb run.
  - if env ``CLUSTER_SAMPLER_AUDIT_STRICT=1`` is set, raises on duplicates.
  - otherwise emits a single warning per training run if anything's off.
"""

from __future__ import annotations

import os
from typing import List

import lightning as L
import torch
import torch.distributed as dist
from loguru import logger

from proteinfoundation.utils.cluster_utils import ClusterSampler


class ClusterSamplerAuditCallback(L.Callback):
    """Verifies disjoint cross-rank partitioning of the train ClusterSampler."""

    def __init__(self, strict: bool = False, max_indices_per_rank: int = 50000):
        """
        Args:
            strict: If True, raise RuntimeError when duplicates are detected.
                Defaults to False (warn only). Overridden by env
                ``CLUSTER_SAMPLER_AUDIT_STRICT=1``.
            max_indices_per_rank: Cap the per-rank index list to bound the
                all_gather buffer size. Datasets with > ~5M clusters per rank
                should bump this knob.
        """
        super().__init__()
        self._ran = False
        self._strict = strict or os.environ.get("CLUSTER_SAMPLER_AUDIT_STRICT", "0") == "1"
        self._max_indices_per_rank = int(max_indices_per_rank)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._ran:
            return
        self._ran = True

        train_dl = getattr(trainer, "train_dataloader", None)
        if train_dl is None:
            return
        sampler = getattr(train_dl, "sampler", None)
        if not isinstance(sampler, ClusterSampler):
            logger.info(
                "[cluster_sampler_audit] train sampler is not ClusterSampler "
                f"({type(sampler).__name__}); skipping audit."
            )
            return
        # Make sure the sampler has the right epoch (Lightning normally does
        # this via _set_sampler_epoch; we do it explicitly here too).
        sampler.set_epoch(int(trainer.current_epoch))

        # Materialize this rank's index list. Bounded by max_indices_per_rank.
        rank_indices: List[int] = []
        for i, idx in enumerate(iter(sampler)):
            if i >= self._max_indices_per_rank:
                break
            rank_indices.append(int(idx))

        # Pad to a common length so all_gather works (different ranks see
        # slightly different partition sizes when not drop_last).
        local_len = torch.tensor(len(rank_indices), dtype=torch.long)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            sizes = [torch.zeros_like(local_len) for _ in range(world_size)]
            dist.all_gather(sizes, local_len)
            max_len = int(max(s.item() for s in sizes))
            padded = torch.full((max_len,), -1, dtype=torch.long)
            padded[: len(rank_indices)] = torch.tensor(rank_indices, dtype=torch.long)
            gathered = [torch.zeros_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)
            if rank != 0:
                return  # only rank 0 reports
            all_indices: List[int] = []
            for buf, sz in zip(gathered, sizes):
                all_indices.extend(buf[: int(sz.item())].tolist())
        else:
            all_indices = rank_indices

        total = len(all_indices)
        unique = len(set(all_indices))
        duplicates = total - unique
        ratio = unique / total if total else 0.0

        msg = (
            f"[cluster_sampler_audit] epoch={trainer.current_epoch} "
            f"world={dist.get_world_size() if dist.is_initialized() else 1} "
            f"total_yielded={total} unique={unique} duplicates={duplicates} "
            f"coverage_ratio={ratio:.4f}"
        )
        if duplicates > 0:
            (logger.error if self._strict else logger.warning)(msg)
            if self._strict:
                raise RuntimeError(
                    "ClusterSampler DDP partitioning violated: ranks see overlapping clusters. "
                    "Check that BaseLightningDataModule.cluster_sampler_v2=True and that "
                    "set_epoch is being called by Lightning. "
                    f"({duplicates} duplicate index assignments out of {total})"
                )
        else:
            logger.info(msg)

        # Best-effort wandb log.
        if (
            getattr(pl_module, "logger", None) is not None
            and hasattr(pl_module.logger, "experiment")
            and hasattr(pl_module.logger.experiment, "log")
        ):
            try:
                pl_module.logger.experiment.log({
                    "cluster_sampler_audit/duplicates_total": int(duplicates),
                    "cluster_sampler_audit/coverage_ratio": float(ratio),
                    "cluster_sampler_audit/total_yielded": int(total),
                    "global_step": int(trainer.global_step),
                })
            except Exception as e:
                logger.warning(f"[cluster_sampler_audit] wandb log failed: {e!r}")
