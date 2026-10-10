"""Gate for proteinfoundation/datasets/epoch_batch_plan.py (pure torch, CPU).

1. Every cluster contributes exactly one member per epoch (plus the documented repeats that complete the last batch
   and pad the batch count to the world size, all drawn from that epoch's own members).
2. Batches are length-grouped: outside the remainder batches, all members of a batch share a length edge.
3. Identical plan on every rank; the rank shards are disjoint in batches and equal in length; plans differ across
   epochs and repeat for the same (seed, epoch).
4. The member draw is uniform within a cluster over epochs (chi-square-free check: every member is drawn).
5. Sampler: every pass advances the epoch (no set_epoch); len is constant; a restore at consumed = k yields exactly the
   plan's batches k..end, then the next epoch; a restore at the end starts the NEXT epoch; a world-size mismatch refuses.
Run: python scratchpad/test_epoch_batch_plan.py
"""

import bisect
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "epoch_batch_plan", os.path.join(HERE, "..", "proteinfoundation", "datasets", "epoch_batch_plan.py"))
ebp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ebp)

EDGES = [64, 128, 192, 256, 320, 384, 448, 512]
N_CLUSTERS = 997  # prime, so neither the batch size nor the world size divides it
CLUSTERS, LENGTHS, OWNER = [], [], {}
for c in range(N_CLUSTERS):
    ms = list(range(len(LENGTHS), len(LENGTHS) + 1 + c % 4))
    for i in ms:
        LENGTHS.append(20 + (37 * i) % 493)
        OWNER[i] = c
    CLUSTERS.append(ms)


def flat(plan):
    return [i for rank in plan for b in rank for i in b]


def test_coverage_and_grouping():
    for bs, world in ((1, 1), (3, 2), (4, 4)):
        plan = ebp.build_epoch_plan(CLUSTERS, LENGTHS, bs, EDGES, 0, 5, world)
        allb = [b for rank in plan for b in rank]
        assert all(len(b) == bs for b in allb)
        clusters = [OWNER[i] for i in flat(plan)]
        uniq = set(clusters)
        assert uniq == set(range(N_CLUSTERS)), len(uniq)
        n_batches = -(-N_CLUSTERS // bs)
        n_total = n_batches + (-n_batches % world)
        assert len(allb) == n_total, (len(allb), n_total)
        extra = len(clusters) - N_CLUSTERS
        assert extra < bs + world * bs, extra
        mixed = sum(len({bisect.bisect_left(EDGES, LENGTHS[i]) for i in b}) > 1 for b in allb)
        assert mixed <= len(EDGES), mixed
        print(f"PASS bs={bs} world={world}: all {N_CLUSTERS} clusters once, {extra} padding repeats, "
              f"{len(allb)} batches, {mixed} mixed-length (remainder) batches <= {len(EDGES)} edges")


def test_ranks_and_determinism():
    p1 = ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 7, 2, 4)
    p2 = ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 7, 2, 4)
    p3 = ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 7, 3, 4)
    assert p1 == p2 and p1 != p3
    assert len({len(r) for r in p1}) == 1
    ids = [tuple(b) for r in p1 for b in r]
    n_unique_batches = len(set(ids))
    print(f"PASS same (seed, epoch) -> same plan, next epoch differs; 4 equal shards of {len(p1[0])} batches "
          f"({len(ids) - n_unique_batches} padded batch repeats)")


def test_member_draw_covers_members():
    seen = set()
    for e in range(40):
        seen |= set(flat(ebp.build_epoch_plan(CLUSTERS, LENGTHS, 1, EDGES, 0, e, 1)))
    assert seen == set(range(len(LENGTHS))), len(LENGTHS) - len(seen)
    print(f"PASS over 40 epochs every one of the {len(LENGTHS)} members was drawn")


def test_sampler_state():
    s = ebp.EpochBatchPlanSampler(CLUSTERS, LENGTHS, 3, EDGES, 0, 1, 2)
    e0, e1 = list(s), list(s)
    assert s.epoch == 1 and e0 != e1 and len(e0) == len(e1) == len(s)
    assert e1 == ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 0, 1, 2)[1]
    st = s.state_dict(consumed=10)
    r = ebp.EpochBatchPlanSampler(CLUSTERS, LENGTHS, 3, EDGES, 0, 1, 2, restore=st)
    assert list(r) == e1[10:] and r.epoch == 1
    assert list(r) == ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 0, 2, 2)[1] and r.epoch == 2
    end = ebp.EpochBatchPlanSampler(CLUSTERS, LENGTHS, 3, EDGES, 0, 1, 2, restore=s.state_dict(consumed=len(e1)))
    assert end.epoch == 2 and list(end) == ebp.build_epoch_plan(CLUSTERS, LENGTHS, 3, EDGES, 0, 2, 2)[1]
    try:
        ebp.EpochBatchPlanSampler(CLUSTERS, LENGTHS, 3, EDGES, 0, 0, 4, restore=st)
        raise SystemExit("FAIL: world-size mismatch restored")
    except AssertionError:
        pass
    print(f"PASS each pass advances the epoch (len {len(s)} constant); restore at k yields batches k..end, then the "
          "next epoch; restore at the end starts the next epoch; world-size mismatch refused")


if __name__ == "__main__":
    test_coverage_and_grouping()
    test_ranks_and_determinism()
    test_member_draw_covers_members()
    test_sampler_state()
    print("ALL PASS")
