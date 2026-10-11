"""Epoch batch plan (user 2026-10-10): one example per sequence cluster per epoch, as ClusterSampler draws it, but
all of the epoch's members are drawn at epoch start and grouped into batches of similar length, and the plan plus
the number of batches already consumed travel in the checkpoint, so a mid-epoch resume continues the SAME epoch.

Plan for epoch e (identical on every rank, from (seed, e) only):
  1. cluster order = randperm(n_clusters) from generator (seed, e, 0); one member per cluster from (seed, e, 1),
     drawn in that order;
  2. each member goes to the first length edge >= its length; within an edge, consecutive members (already in random
     cluster order) form batches of batch_size; the remainders of all edges are concatenated in edge order and
     chunked the same way, and a last short batch is completed by repeating the epoch's first members (as
     ClusterSampler / DistributedSampler pad);
  3. batch order = randperm from (seed, e, 2); the batch list is padded to a multiple of the world size by repeating
     its first batches; rank r gets batches[r::world].
"""

import bisect
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Sampler


def _generator(seed: int, *parts: int) -> torch.Generator:
    """Same 64-bit mix as ClusterSampler._make_generator, so (seed, epoch, stream) seeds are comparable."""
    s = seed & 0xFFFFFFFF
    for p in parts:
        s = ((s * 0x9E3779B97F4A7C15) ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    g = torch.Generator()
    g.manual_seed(s if s != 0 else 1)
    return g


def build_epoch_plan(cluster_members: List[List[int]], lengths: List[int], batch_size: int, len_edges: List[int],
                     seed: int, epoch: int, world: int) -> List[List[List[int]]]:
    """Return plan[rank] = list of batches (lists of dataset indices). lengths[i] is dataset index i's length."""
    order = torch.randperm(len(cluster_members), generator=_generator(seed, epoch, 0)).tolist()
    g_member = _generator(seed, epoch, 1)
    members = []
    for c in order:
        m = cluster_members[c]
        members.append(m[torch.randint(0, len(m), (1,), generator=g_member).item()])
    by_edge: Dict[int, List[int]] = {}
    for i in members:
        k = bisect.bisect_left(len_edges, lengths[i])
        by_edge.setdefault(k, []).append(i)
    batches, rest = [], []
    for k in sorted(by_edge):
        xs = by_edge[k]
        full = len(xs) // batch_size * batch_size
        batches += [xs[j:j + batch_size] for j in range(0, full, batch_size)]
        rest += xs[full:]
    batches += [rest[j:j + batch_size] for j in range(0, len(rest), batch_size)]
    if batches and len(batches[-1]) < batch_size:
        batches[-1] = batches[-1] + members[:batch_size - len(batches[-1])]
    perm = torch.randperm(len(batches), generator=_generator(seed, epoch, 2)).tolist()
    batches = [batches[j] for j in perm]
    pad = -len(batches) % world
    batches += batches[:pad]
    return [batches[r::world] for r in range(world)]


class EpochBatchPlanSampler(Sampler[List[int]]):
    """batch_sampler over a per-epoch plan. It advances its OWN epoch: every iterator that is STARTED (first next())
    after the first one starts the next epoch. Lightning 2.5 never calls set_epoch on a batch sampler (only on dataloader.sampler), and it builds exactly
    one iterator per epoch -- also when it re-enters a finished epoch after an epoch-end checkpoint, where that one
    iterator serves the next epoch. The datamodule asserts sampler.epoch == trainer.current_epoch at every save.

    State for checkpoints: state_dict() -> {epoch, plan, consumed}; plan is the full all-rank plan of the current
    epoch (so a resume never depends on rebuilding it), consumed is the number of THIS rank's batches already trained
    in that epoch (from the trainer's progress at save time).

    Without a saved plan (a fresh run, or a checkpoint written before this sampler existed), the first started iterator
    asks position_fn() for (epoch, batches already trained in it) -- the trainer's restored counters -- so the plan's
    epoch and position agree with Lightning's from the first save on."""

    def __init__(self, cluster_members: List[List[int]], lengths: List[int], batch_size: int, len_edges: List[int],
                 seed: int, rank: int, world: int, restore: Optional[Dict] = None,
                 position_fn: Optional[Callable[[], Tuple[int, int]]] = None):
        self.cluster_members, self.lengths = cluster_members, lengths
        self.batch_size, self.len_edges = int(batch_size), sorted(int(x) for x in len_edges)
        self.seed, self.rank, self.world = int(seed), int(rank), int(world)
        self.n_batches = -(-len(cluster_members) // self.batch_size)  # all ranks, before world padding
        assert self.n_batches >= self.world, (self.n_batches, self.world)
        self.n_per_rank = -(-self.n_batches // self.world)  # the same every epoch: one member per cluster
        self.epoch, self.plan, self.skip, self._started = 0, None, 0, False
        self.position_fn = position_fn if restore is None else None
        if restore is not None:
            assert len(restore["plan"]) == self.world, (len(restore["plan"]), self.world)
            self._set_position(int(restore["epoch"]), int(restore["consumed"]), restore["plan"])

    def _set_position(self, epoch: int, consumed: int, plan=None):
        self.epoch, self.plan, self.skip = epoch, plan, consumed
        if self.skip >= self.n_per_rank:
            # checkpoint at an epoch's last batch: Lightning re-enters that finished epoch and builds ONE iterator,
            # before the next epoch starts, so that iterator must be the next epoch's
            self.epoch, self.plan, self.skip = epoch + 1, None, 0

    def _ensure_plan(self):
        if self.plan is None:
            self.plan = build_epoch_plan(self.cluster_members, self.lengths, self.batch_size, self.len_edges,
                                         self.seed, self.epoch, self.world)
            assert len(self.plan[self.rank]) == self.n_per_rank, (len(self.plan[self.rank]), self.n_per_rank)

    def __iter__(self):
        # Must stay a generator: torch's DataLoader calls iter(batch_sampler) twice per loader iterator and starts only
        # the second, so the epoch must advance on the first next(), never in iter() itself.
        if self._started:
            self.epoch, self.plan, self.skip = self.epoch + 1, None, 0
        elif self.position_fn is not None:
            self._set_position(*self.position_fn())
        self._started = True
        self._ensure_plan()
        skip, self.skip = self.skip, 0
        yield from self.plan[self.rank][skip:]

    def __len__(self) -> int:
        return self.n_per_rank

    def state_dict(self, consumed: int) -> Dict:
        self._ensure_plan()
        return {"epoch": self.epoch, "plan": self.plan, "consumed": int(consumed)}
