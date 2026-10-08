"""Gate for 660fbdb (user 2026-10-07): with load_cath_mapping cached, a DataLoader collate that calls it every batch
opens cath_label_mapping.pt at most once per worker (io_audit, positive control checked). Usage: python ... WORKDIR"""

import glob
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

W = sys.argv[1]
os.environ["IO_AUDIT_DIR"] = W
from proteinfoundation.datasets.cath_utils import load_cath_mapping
from proteinfoundation.utils import io_audit

CATH = "/orcd/pool/006/chenxiou/proteina/data/cath_shared"


class D(Dataset):
    def __len__(self): return 64
    def __getitem__(self, i): return torch.tensor(i)


def collate(xs):
    load_cath_mapping(CATH)  # the per-batch call that used to hit NFS every time
    return torch.stack(xs)


n = sum(1 for _ in DataLoader(D(), batch_size=2, num_workers=2, collate_fn=collate,
                              worker_init_fn=lambda w: io_audit.install(f"w{w}")))
files = sorted(glob.glob(os.path.join(W, "w*.tsv")))
rows = {f: open(f).read().splitlines() for f in files}
assert n == 32 and len(files) == 2 and all(r and r[0].endswith(os.devnull) for r in rows.values()), (n, files)
opens = {os.path.basename(f): sum("cath_label_mapping.pt" in x for x in r) for f, r in rows.items()}
assert all(v <= 1 for v in opens.values()), opens
print(f"PASS cath cache: {n} batches over 2 workers, cath_label_mapping.pt opens per worker {opens} (control row present)")
