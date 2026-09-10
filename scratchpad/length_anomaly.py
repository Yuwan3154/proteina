"""D19: why do 15.7% of template rows cover MORE residues than their native?

Measured earlier (job 22485013, 485 chains / 15,646 template rows): template run-sum vs native
run-sum was identical for 81.6%, template LONGER for 15.7% (max +233), shorter for 2.7%.

A synthetic template is a partial-diffusion variant of the chain's OWN native, so both should have
the same residue count. Two candidate explanations, and they are distinguishable:

  (H1) The NATIVE loses residues to the DSSP gap, not the template. `dssp_to_runs` now emits
       DSSP_GAP runs for unresolved backbone, and a native from a crystal structure has unresolved
       residues while a GENERATED template has a complete backbone everywhere. Then
       native_run_sum == native_L (gaps included, since the fix) but the native's NON-GAP coverage
       is smaller -- and if any code compares non-gap sums, the template looks longer.
  (H2) The template genuinely has more residues than proteina's processed chain, i.e. the two come
       from different residue sets (openfold's chain vs proteina's).

H1 predicts: total run-sum (gaps included) matches, and the difference is entirely in GAP residues.
H2 predicts: total run-sum differs, gaps or no gaps.

This reports both totals per row, so the two hypotheses separate cleanly.
"""

import argparse

import torch

from proteinfoundation.datasets.sse_topology import DSSP_GAP

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--max-chains", type=int, default=3000)
a = ap.parse_args()

idx = torch.load(a.index, map_location="cpu", weights_only=False)
runs_flat, runs_off = idx["runs_flat"], idx["runs_offset"]
mem_flat, mem_off = idx["members_flat"], idx["members_offset"]
native_rows = torch.nonzero(idx["row_is_native"]).flatten()


def sums(r):
    """(total residues covered, residues in GAP runs) for one row."""
    block = runs_flat[runs_off[r]:runs_off[r + 1]]
    if block.numel() == 0:
        return 0, 0
    types, lens = block[:, 0], block[:, 1].long()
    return int(lens.sum()), int(lens[types == DSSP_GAP].sum())


n_chains = int(mem_off.numel()) - 1
tot_diff, gap_diff, n = [], [], 0
nat_gap, tpl_gap = [], []
for c in range(min(n_chains, a.max_chains)):
    rows = mem_flat[mem_off[c]:mem_off[c + 1]].tolist()
    if not rows:
        continue
    n_tot, n_gap = sums(int(native_rows[c]))
    if n_tot == 0:
        continue
    for r in rows:
        t_tot, t_gap = sums(r)
        if t_tot == 0:
            continue
        tot_diff.append(t_tot - n_tot)
        gap_diff.append(t_gap - n_gap)
        nat_gap.append(n_gap)
        tpl_gap.append(t_gap)
        n += 1

assert n >= 200, f"VACUOUS: only {n} template rows compared"
d = torch.tensor(tot_diff, dtype=torch.float32)
gd = torch.tensor(gap_diff, dtype=torch.float32)
ng = torch.tensor(nat_gap, dtype=torch.float32)
tg = torch.tensor(tpl_gap, dtype=torch.float32)

print(f"template rows compared: {n}\n")
print("TOTAL residues covered (gaps INCLUDED), template minus native:")
print(f"  identical {int((d == 0).sum())} ({100.0 * float((d == 0).float().mean()):.1f}%)   "
      f"longer {int((d > 0).sum())} ({100.0 * float((d > 0).float().mean()):.1f}%)   "
      f"shorter {int((d < 0).sum())} ({100.0 * float((d < 0).float().mean()):.1f}%)")
print(f"  mean {d.mean():+.2f}  min {int(d.min())}  max {int(d.max())}")

print("\nGAP residues (unresolved backbone / sub-min_len runs):")
print(f"  native  : mean {ng.mean():.2f}  nonzero in {100.0 * float((ng > 0).float().mean()):.1f}% of rows")
print(f"  template: mean {tg.mean():.2f}  nonzero in {100.0 * float((tg > 0).float().mean()):.1f}% of rows")
print(f"  template minus native gap residues: mean {gd.mean():+.2f}")

same_tot = int((d == 0).sum())
print("\nVERDICT:")
if same_tot / n > 0.95:
    print("  H1 -- totals agree; the earlier 15.7% came from comparing NON-GAP coverage,")
    print("        i.e. the native's unresolved residues, not a different residue set.")
else:
    print(f"  H2 not excluded -- {100.0 * (1 - same_tot / n):.1f}% of template rows still differ in")
    print("        TOTAL residue coverage, so template and native do not share a residue set.")
