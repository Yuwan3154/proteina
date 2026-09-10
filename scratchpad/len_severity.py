"""How large are the template/native length mismatches, among the rows that DO differ?

84.8% match exactly; the tail reaches +-246. Whether the mismatched 15.2% are off by 3 residues or
by 200 decides whether this is noise or a data defect, so this reports the distribution of the
RELATIVE mismatch |diff| / native_length over the non-matching rows only.
"""

import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--max-chains", type=int, default=4000)
a = ap.parse_args()

idx = torch.load(a.index, map_location="cpu", weights_only=False)
runs_flat, runs_off = idx["runs_flat"], idx["runs_offset"]
mem_off, mem_flat = idx["members_offset"], idx["members_flat"]
native_rows = torch.nonzero(idx["row_is_native"]).flatten()


def total(r):
    b = runs_flat[runs_off[r]:runs_off[r + 1]]
    return 0 if b.numel() == 0 else int(b[:, 1].long().sum())


rel, chains_bad, chains_seen = [], 0, 0
for c in range(min(int(mem_off.numel()) - 1, a.max_chains)):
    rows = mem_flat[mem_off[c]:mem_off[c + 1]].tolist()
    if not rows:
        continue
    n_tot = total(int(native_rows[c]))
    if n_tot == 0:
        continue
    chains_seen += 1
    bad = False
    for r in rows:
        d = total(r) - n_tot
        if d:
            rel.append(abs(d) / n_tot)
            bad = True
    chains_bad += bad

assert chains_seen >= 200, f"VACUOUS: {chains_seen} chains"
t = torch.tensor(rel, dtype=torch.float32)
print(f"chains with >=1 template: {chains_seen};  chains with ANY mismatched template: "
      f"{chains_bad} ({100.0 * chains_bad / chains_seen:.1f}%)")
print(f"mismatched rows: {t.numel()}")
print("\nrelative mismatch |diff| / native_length, over MISMATCHED rows only:")
for q in (0.05, 0.25, 0.5, 0.75, 0.95):
    print(f"  p{int(q * 100):<3} {float(torch.quantile(t, q)):.3f}")
print(f"  mean {t.mean():.3f}")
print(f"\nrows off by MORE THAN 10% of the native length: "
      f"{int((t > 0.10).sum())} ({100.0 * float((t > 0.10).float().mean()):.1f}% of mismatched)")
