"""Pin the SSE vocabulary (loops excluded) and the reduced pair-feature set.

User 2026-09-10: drop the min/mean CA-CA channels from the topology reference ("a feature that must
be derived from real coordinates is one a synthetic-reference generator has to reproduce"), and
confirm the SSE vocabulary excludes loops.
"""

import torch

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_LOOP,
    DSSP_STRAND,
    N_PAIR_FEATURES,
    N_SPECIAL_TOKENS,
    PAIR_FEATURE_MODES,
    PAIR_FEATURE_NAMES,
    SSEAlphabet,
    STRUCTURAL_PAIR_FEATURES,
    assemble_pair_features,
    sse_structural_pair_features,
)

HE = (DSSP_HELIX, DSSP_STRAND)


def test_helix_strand_vocab_is_44():
    a = SSEAlphabet(types=HE)
    assert a.slots_per_type == 21, a.slots_per_type
    assert a.vocab_size == N_SPECIAL_TOKENS + 2 * 21 == 44, a.vocab_size


def test_default_three_type_vocab_is_65_unchanged():
    assert SSEAlphabet().vocab_size == 65


def test_no_loop_token_exists_in_the_he_alphabet():
    a = SSEAlphabet(types=HE)
    try:
        a.token(DSSP_LOOP, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("a loop was tokenised by the helix+strand alphabet")


def test_every_he_token_is_in_range_and_above_the_special_tokens():
    a = SSEAlphabet(types=HE)
    seen = set()
    for t in HE:
        for n in range(a.min_len, 60):
            tok = a.token(t, n)
            assert N_SPECIAL_TOKENS <= tok < a.vocab_size, (t, n, tok)
            seen.add(tok)
    assert len(seen) == 2 * a.slots_per_type, f"{len(seen)} distinct tokens, expected {2 * 21}"


def test_ca_distance_channels_are_gone():
    assert "min_ca_dist" not in PAIR_FEATURE_NAMES
    assert "mean_ca_dist" not in PAIR_FEATURE_NAMES
    assert STRUCTURAL_PAIR_FEATURES == ("contact_frac", "orientation_cos")
    assert N_PAIR_FEATURES == 8, N_PAIR_FEATURES
    for mode, names in PAIR_FEATURE_MODES.items():
        assert all(n in PAIR_FEATURE_NAMES for n in names), mode


def test_structural_features_have_two_channels_and_stay_symmetric():
    L, T = 12, 2
    runs = [(DSSP_HELIX, 6), (DSSP_STRAND, 6)]
    keep = [0, 1]
    cm = torch.zeros(L, L)
    cm[2, 8] = cm[8, 2] = 1.0
    coords = torch.randn(L, 5, 3)
    cmask = torch.ones(L, 5)
    out = sse_structural_pair_features(cm, coords, cmask, runs, keep)
    assert out.shape == (T, T, 2), out.shape
    assert torch.allclose(out, out.transpose(0, 1)), "structural features must be symmetric"


def test_assembled_width_matches_the_name_list():
    L = 12
    runs = [(DSSP_HELIX, 6), (DSSP_STRAND, 6)]
    keep = [0, 1]
    cm = torch.zeros(L, L)
    coords, cmask = torch.randn(L, 5, 3), torch.ones(L, 5)
    contact = torch.zeros(2, 2)
    structural = sse_structural_pair_features(cm, coords, cmask, runs, keep)
    feat = assemble_pair_features(contact, structural, runs, keep)
    assert feat.shape[-1] == N_PAIR_FEATURES == len(PAIR_FEATURE_NAMES), feat.shape


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
    print(f"\n{n_fail} failing")
