"""How long does the 48M run's FIRST training batch actually take?

Job 21827811 died with `DataLoader timed out after 300 seconds` on the first batch, GPUs idle
throughout. The pack file is healthy (542 GB, 301 MB/s sequential), so the suspect is cold-cache
random access over NFS -- but "slow" and "hung" look identical from a timeout.

`base_data.py:317` sets `dl_timeout = 300 if num_workers > 0 else 0`, so **num_workers=0 has no
timeout at all**. Timing that case gives the true first-batch latency with no deadline, which is
what separates the two explanations:

  - num_workers=0 returns in well under 300 s  -> the workers/IPC are the problem, not the data
  - num_workers=0 also takes minutes           -> genuine cold-cache cost; the hardcoded 300 s is
                                                  simply too tight for a cold start on this pack

No GPU is required to build the datamodule and pull batches.
"""

import os
import sys
import time

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN = "dssp_contact_48M_udlm_pb_v2_stage1_catbalanced_domaincrop_combined"
WORKER_SETTINGS = [int(x) for x in os.environ.get("WORKERS", "0,1,4").split(",")]
N_BATCHES = int(os.environ.get("N_BATCHES", "3"))


def main():
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=f"training_{RUN}")
    OmegaConf.set_struct(cfg_exp, False)
    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    base_workers = cfg_data.datamodule.get("num_workers", "?")
    print(f"[cfg] dataset={cfg_exp.dataset} configured num_workers={base_workers} "
          f"prefetch={cfg_data.datamodule.get('prefetch_factor','?')} "
          f"batch_size={cfg_data.datamodule.get('batch_size','?')}", flush=True)
    print(f"[cfg] PACK_PATH={os.environ.get('PACK_PATH','(unset)')}", flush=True)

    # Set up ONCE (it costs ~224 s) and then vary only the dataloader knobs.
    t0 = time.time()
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.setup("fit")
    print(f"\n[setup] {time.time() - t0:.1f} s (done once)", flush=True)

    for nw in WORKER_SETTINGS:
        print(f"\n=== num_workers={nw} "
              f"({'NO timeout' if nw == 0 else 'timeout 300 s'}) ===", flush=True)
        dm.num_workers = nw
        # PyTorch rejects prefetch_factor when num_workers == 0.
        dm.prefetch_factor = None if nw == 0 else 2

        t0 = time.time()
        dl = dm.train_dataloader()
        it = iter(dl)
        print(f"  iter()           {time.time() - t0:8.1f} s", flush=True)

        for i in range(N_BATCHES):
            t0 = time.time()
            batch = next(it)
            dt = time.time() - t0
            n = len(batch["protein_id"]) if "protein_id" in batch else "?"
            print(f"  batch {i}          {dt:8.1f} s   (n={n})"
                  f"{'   <-- FIRST BATCH' if i == 0 else ''}", flush=True)
        del it, dl

    return 0


if __name__ == "__main__":
    sys.exit(main())
