"""Real-data check of c2c unsharded validation (2026-09-27): the val dataloader train_c2c.py builds must give
every DDP rank the single-process val sequence. Builds the datamodule from the real dataset config exactly as
train_c2c.py does (hydra compose + val_shard_across_ranks=False), under a 2-process gloo group, and compares the
first N_VAL chain ids (limit_val_batches=64 at batch_size 1) per rank against a single-process build.
Also checks the sharded (default) build still partitions, i.e. the flag is what changes the behaviour.
Usage: python scratchpad/test_c2c_val_unsharded.py DATASET_NAME
"""

import os
import sys
import tempfile

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from proteinfoundation.utils.cluster_utils import ClusterSampler

N_VAL, EPOCHS = 64, (0, 7, 40)


def val_ids(dataset, unsharded):
    ds_dir = "../configs/datasets_config/pdb"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=dataset)
    OmegaConf.set_struct(cfg, False)
    cfg.datamodule.batch_size = 1
    cfg.datamodule.num_workers = 0
    if unsharded:
        cfg.datamodule.val_shard_across_ranks = False
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup("fit")
    dl = dm.val_dataloader()
    s = dl.sampler
    assert isinstance(s, ClusterSampler), type(s)
    assert s.shard is (not unsharded), (s.shard, unsharded)
    out = {}
    for e in EPOCHS:
        s.set_epoch(e)
        idx = [i for _, i in zip(range(N_VAL), iter(s))]
        out[e] = [dm.val_ds.file_names[i].split(".")[0] for i in idx]
    return out


def worker(rank, world, init, dataset, d):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    torch.save({"unsharded": val_ids(dataset, True), "sharded": val_ids(dataset, False)}, f"{d}/rank{rank}.pt")
    dist.destroy_process_group()


def main():
    dataset = sys.argv[1]
    single = val_ids(dataset, True)
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(worker, args=(2, f"file://{d}/pg", dataset, d), nprocs=2, join=True)
        r = [torch.load(f"{d}/rank{i}.pt") for i in range(2)]
    for e in EPOCHS:
        assert len(single[e]) == N_VAL, len(single[e])
        for i in range(2):
            assert r[i]["unsharded"][e] == single[e], f"epoch {e} rank {i}: unsharded val ids != single-process"
        assert r[0]["sharded"][e] != r[1]["sharded"][e], f"epoch {e}: sharded ranks identical (flag inert?)"
        print(f"epoch {e}: 2 ranks x {N_VAL} unsharded val ids == single-process; first {single[e][:3]}; "
              f"sharded ranks differ (first {r[0]['sharded'][e][0]} vs {r[1]['sharded'][e][0]})", flush=True)
    assert single[EPOCHS[0]] != single[EPOCHS[1]], "val set did not rotate with the epoch"
    print(f"ALL PASS ({len(EPOCHS)} epochs x 2 ranks x {N_VAL} ids, dataset {dataset})")


if __name__ == "__main__":
    main()
