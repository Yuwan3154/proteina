"""[3] Count DataLoader handoff socket connects by location (audit hook, inherited by forked workers)."""
import os, sys, torch
from torch.utils.data import DataLoader, Dataset
LOG = sys.argv[1]
def hook(ev, args):
    if ev == "socket.connect" and isinstance(args[1], str):
        with open(f"{LOG}/{os.getpid()}.ev", "a") as fh: fh.write(args[1] + "\n")
sys.addaudithook(hook)
class D(Dataset):
    def __len__(self): return 40
    def __getitem__(self, i): return {"x": torch.randn(64, 64), "i": torch.tensor(i)}
n = sum(1 for _ in DataLoader(D(), batch_size=2, num_workers=2, persistent_workers=True, timeout=300))
ev = [l.strip() for f in os.listdir(LOG) if f.endswith(".ev") for l in open(os.path.join(LOG, f))]
tmp, scr = sum(e.startswith(os.environ["TMPDIR"]) for e in ev), sum(e.startswith("/orcd/scratch/") for e in ev)
assert n == 20 and scr == 0 and tmp == 2 * n, (n, scr, tmp)
print(f"[3] PASS sockets: {n} batches, {tmp} connects under node-local TMPDIR, {scr} under scratch")
