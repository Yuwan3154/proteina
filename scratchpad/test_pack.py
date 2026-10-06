"""Gate for the tri pack (user 2026-10-06): a tiny pack loads graphs identical to disk; a missing stem raises at init.
Usage: python scratchpad/test_pack.py WORKDIR"""

import os
import subprocess
import sys

import torch

from proteinfoundation.datasets.pdb_data import PDBDataset

W = sys.argv[1]
D = "/orcd/pool/006/chenxiou/proteina/data/pdb_train"
stems = ["1dmx_A", "1csb_B", "5f3x_A", "1g5r_A", "102l_A"]
lst = os.path.join(W, "stems.txt")
open(lst, "w").write("\n".join(stems) + "\n")
pack = os.path.join(W, "t.pack")
assert os.system(f"{sys.executable} scratchpad/build_pack.py {lst} {D} {pack}") == 0
codes, chains = [s.split("_")[0] for s in stems], [s.split("_")[1] for s in stems]
disk = PDBDataset(pdb_codes=codes, chains=chains, data_dir=D, file_names=stems, num_workers=0)
packed = PDBDataset(pdb_codes=codes, chains=chains, data_dir=D, file_names=stems, num_workers=0, pack_path=pack)
for i in range(len(stems)):
    a, b = disk[i], packed[i]
    assert a.id == b.id and torch.equal(a.coords, b.coords) and torch.equal(a.coord_mask, b.coord_mask), stems[i]
print(f"[1] PASS {len(stems)} packed graphs identical to disk (id, coords, coord_mask)")
code = ("from proteinfoundation.datasets.pdb_data import PDBDataset; "
        f"PDBDataset(pdb_codes={codes + ['9zzz']!r}, chains={chains + ['A']!r}, data_dir={D!r}, "
        f"file_names={stems + ['9zzz_A']!r}, num_workers=0, pack_path={pack!r})")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
assert r.returncode != 0 and "ValueError" in r.stderr and "lacks 1 of 6 stems" in r.stderr, r.stderr[-500:]
print("[2] PASS a stem absent from the pack raises at init")
