"""ClusterSampler shard flag (2026-09-27, c2c 2-GPU with unsharded validation).

Under a real 2-process gloo group:
  A. shard=False on each rank == the single-process sequence (same order AND members) at the same epoch.
  B. shard=True (default) on each rank == the pre-change ClusterSampler (git HEAD~ copy passed as argv[1]).
  C. shard=True ranks are disjoint in cluster slots and together cover every cluster.
Usage: python scratchpad/test_cluster_sampler_shard.py /path/to/cluster_utils_old.py
"""

import importlib.util
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from proteinfoundation.utils.cluster_utils import ClusterSampler

N_CLUSTERS, SEED, EPOCHS = 101, 7, (0, 3, 25159)   # odd count exercises the padding path


class FakeDS:
    database = "pdb"

    def __init__(self, names):
        self.file_names = [f"{n}.pt" for n in names]


def mapping():
    g = torch.Generator().manual_seed(0)
    m, names = {}, []
    for c in range(N_CLUSTERS):
        k = int(torch.randint(1, 6, (1,), generator=g))
        members = [f"c{c:03d}m{j}" for j in range(k)]
        m[members[0]] = members
        names += members
    return m, FakeDS(names)


def run(cls, epoch, **kw):
    m, ds = mapping()
    s = cls(dataset=ds, clusterid_to_seqid_mapping=m, sampling_mode="cluster-random", seed=SEED, **kw)
    s.set_epoch(epoch)
    return list(iter(s)), len(s)


def old_cls(path):
    spec = importlib.util.spec_from_file_location("cluster_utils_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ClusterSampler


def worker(rank, world, init, old_path, out):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    res = {}
    for e in EPOCHS:
        res[("noshard", e)] = run(ClusterSampler, e, shard=False)
        res[("shard", e)] = run(ClusterSampler, e)
        res[("old", e)] = run(old_cls(old_path), e)
    torch.save(res, f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def main():
    old_path = sys.argv[1]
    single = {e: run(ClusterSampler, e, shard=False) for e in EPOCHS}
    single_default = {e: run(ClusterSampler, e) for e in EPOCHS}
    with tempfile.TemporaryDirectory() as d:
        init = f"file://{d}/pg"
        mp.spawn(worker, args=(2, init, old_path, d), nprocs=2, join=True)
        r = [torch.load(f"{d}/rank{i}.pt") for i in range(2)]
    n_pass = 0
    for e in EPOCHS:
        seq1, len1 = single[e]
        assert single_default[e] == single[e], f"epoch {e}: 1-process default != shard=False"
        assert len(seq1) == N_CLUSTERS == len1, (len(seq1), len1)
        for i in range(2):
            assert r[i][("noshard", e)] == (seq1, len1), f"A epoch {e} rank {i}: unsharded != single-process"
            assert r[i][("shard", e)] == r[i][("old", e)], f"B epoch {e} rank {i}: shard=True != pre-change"
            n_pass += 2
        s0, s1 = r[0][("shard", e)][0], r[1][("shard", e)][0]
        assert len(s0) == len(s1) == (N_CLUSTERS + 1) // 2, (len(s0), len(s1))
        m, ds = mapping()
        idx2cl = {i: fn.split("m")[0] for i, fn in enumerate(ds.file_names)}
        cl0, cl1 = [idx2cl[i] for i in s0], [idx2cl[i] for i in s1]
        assert len(set(cl0) | set(cl1)) == N_CLUSTERS, "C: shards do not cover every cluster"
        assert len(set(cl0) & set(cl1)) <= 1, "C: overlap beyond the one padded slot"
        n_pass += 1
        print(f"epoch {e}: A+B pass on 2 ranks, C pass ({len(s0)}+{len(s1)} slots, "
              f"{len(set(cl0) | set(cl1))}/{N_CLUSTERS} clusters)", flush=True)
    print(f"ALL PASS ({n_pass} checks over {len(EPOCHS)} epochs, {N_CLUSTERS} clusters)")


if __name__ == "__main__":
    main()
