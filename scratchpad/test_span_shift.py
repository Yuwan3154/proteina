"""RED regression test: element -> residue spans must be in the SOURCE chain's coordinates.

`dssp_to_runs` drops residues labelled -1 (incomplete backbone) -- they "contribute nothing" -- and
`runs_to_spans` then accumulates only the kept runs. Every consumer of those spans indexes a tensor
on the FULL residue axis (`elem = full((L,), -1); elem[s:t] = a` in sse_contact_reference,
sse_structural_pair_features and the synthetic-index builder), so every element after a dropped
residue is shifted LEFT by the number of residues dropped before it.

Measured incidence (job 22484779, 400 eligible chains, per STRUCTURE): 9 of 400 = 2.2% of chains
have sum(run lengths) != residue count -- median 1 residue dropped, mean 36.9, max 323. Every
element after the first gap is mislabelled. (An earlier 14.70% figure compared a template row's
runs against the NATIVE's residue count -- two different structures -- and is retracted.)

These were RED when written; the central fix (DSSP_GAP runs) makes them green.
"""

import torch

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_STRAND,
    dssp_to_runs,
    runs_to_spans,
)



def elem_of_residue(dssp, min_len=1):
    """Element index per residue on the full residue axis -- verbatim what the consumers build.

    Mirrors sse_contact_reference / sse_structural_pair_features / the index builder: `keep` is the
    helix+strand runs, and `elem[s:t] = a` numbers by POSITION IN keep, so gap runs occupy residues
    but never take an element index.
    """
    runs = dssp_to_runs(dssp, min_len=min_len)
    spans = runs_to_spans(runs)
    keep = [i for i, (t, _) in enumerate(runs) if t in (DSSP_HELIX, DSSP_STRAND)]
    elem = torch.full((dssp.numel(),), -1, dtype=torch.long)
    for a, ia in enumerate(keep):
        s, t = spans[ia]
        elem[s:t] = a
    return elem


def test_unresolved_residue_does_not_shift_later_elements():
    dssp = torch.tensor([DSSP_HELIX] * 4 + [-1] + [DSSP_STRAND] * 4)
    expected = torch.tensor([0, 0, 0, 0, -1, 1, 1, 1, 1])
    got = elem_of_residue(dssp)
    assert torch.equal(got, expected), f"shifted: {got.tolist()} != {expected.tolist()}"


def test_leading_gap_does_not_shift_the_first_element():
    dssp = torch.tensor([-1, -1] + [DSSP_HELIX] * 3)
    expected = torch.tensor([-1, -1, 0, 0, 0])
    got = elem_of_residue(dssp)
    assert torch.equal(got, expected), f"shifted: {got.tolist()} != {expected.tolist()}"


def test_no_gap_is_unchanged():
    """The fix must be a no-op on chains with a complete backbone -- the 85.3% majority."""
    dssp = torch.tensor([DSSP_HELIX] * 3 + [DSSP_STRAND] * 2)
    expected = torch.tensor([0, 0, 0, 1, 1])
    assert torch.equal(elem_of_residue(dssp), expected)


def test_min_len_filtered_run_is_a_gap_not_a_deletion():
    """A run dropped by min_len must leave a HOLE, not pull the rest of the chain left."""
    dssp = torch.tensor([DSSP_HELIX] * 3 + [DSSP_STRAND] + [DSSP_HELIX] * 3)
    expected = torch.tensor([0, 0, 0, -1, 1, 1, 1])
    got = elem_of_residue(dssp, min_len=2)
    assert torch.equal(got, expected), f"shifted: {got.tolist()} != {expected.tolist()}"


if __name__ == "__main__":
    n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                n_fail += 1
                print(f"FAIL {name}: {e}")
    print(f"\n{n_fail} of 4 failing")
