"""Gate for the T8 loop-element changes in sse_topology.py / topology_reference.py (user 2026-10-09).

1. SSEAlphabet backward compatibility: with no single-token type, every token of the old 44-id (helix+strand) and
   65-id (loop+helix+strand) alphabets is unchanged; with loops single-token the vocab is 45 and decode round-trips.
2. Element axes on synthetic geometry with known answers: an ideal alpha helix's axis is its helix axis; two
   antiparallel strands give cosine -1, parallel +1; a curved loop gets cosine 0 everywhere ("zero") or the unit
   end-to-end vector ("end_to_end"); a 1-CA loop gets 0 in both modes; loops without a loop_axis raise.
3. element_lengths follows the runs (so it tracks length augmentation).
4. TopologyReferenceTransform.assemble_reference with loops as elements: loop tokens on the T axis, positions, the
   standardised length feature; the index guard rejects a transform whose element_types differ from the index's.
Run: python scratchpad/test_sse_loop_elements.py
"""

import contextlib
import math
import os
import tempfile

import torch

from proteinfoundation.datasets.sse_topology import (
    CA_ATOM_INDEX,
    DSSP_HELIX,
    DSSP_LOOP,
    DSSP_STRAND,
    N_PAIR_FEATURES,
    SSEAlphabet,
    element_lengths,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

L_, H_, E_ = DSSP_LOOP, DSSP_HELIX, DSSP_STRAND


def raises(exc, fn):
    """True iff fn() raises exc (no try/except: the suppress block exits before `done` is set)."""
    done = False
    with contextlib.suppress(exc):
        fn()
        done = True
    return not done


def old_token(types, slots, t, slot):
    return 2 + types.index(t) * slots + slot


def test_alphabet():
    for types, vocab in (((H_, E_), 44), ((L_, H_, E_), 65)):
        a = SSEAlphabet(types=types)
        assert a.vocab_size == vocab, (types, a.vocab_size)
        for t in types:
            for n in range(1, 60):
                assert a.token(t, n) == old_token(types, a.slots_per_type, t, a._slot(n)), (t, n)
    a = SSEAlphabet(types=(L_, H_, E_), single_token_types=(L_,))
    assert a.vocab_size == 45
    assert {a.token(L_, n) for n in range(1, 200)} == {2}
    toks = {a.token(t, n) for t in (H_, E_) for n in range(1, 60)}
    assert min(toks) == 3 and max(toks) == 44 and len(toks) == 42
    for tok in range(2, 45):
        t, rng = a.decode(tok)
        assert (t == L_ and rng == "any") if tok == 2 else t in (H_, E_)
    print("PASS alphabet: old 44/65 layouts unchanged token-for-token; loop single-token vocab 45, decode round-trips")


def helix_ca(n, start, axis_dir):
    """Ideal alpha helix: radius 2.3 A, rise 1.5 A, 100 deg/residue, along axis_dir from start."""
    d = torch.tensor(axis_dir, dtype=torch.float32)
    d = d / d.norm()
    u = torch.linalg.svd(torch.eye(3) - d[:, None] * d[None])[0][:, 0]
    w = torch.linalg.cross(d, u)
    k = torch.arange(n, dtype=torch.float32)
    th = k * math.radians(100.0)
    return torch.tensor(start) + 2.3 * (torch.cos(th)[:, None] * u + torch.sin(th)[:, None] * w) + 1.5 * k[:, None] * d


def strand_ca(n, start, direction):
    d = torch.tensor(direction, dtype=torch.float32)
    d = d / d.norm()
    k = torch.arange(n, dtype=torch.float32)
    return torch.tensor(start) + 3.3 * k[:, None] * d  # straight (no pleat), so the axis is known exactly


def arc_ca(n, start, radius=6.0):
    """A curved loop: CAs on a circle of radius 6 A in the xy plane, 3.8 A apart (8 CAs span ~254 deg)."""
    step = 3.8 / radius
    th = torch.arange(n, dtype=torch.float32) * step
    return torch.tensor(start) + radius * torch.stack([torch.cos(th), torch.sin(th), torch.zeros(n)], -1)


def build(segments):
    """segments: list of (type, CA [n,3]) -> runs, coords [L, 37, 3] with CA filled, coord_mask."""
    ca = torch.cat([x for _, x in segments])
    L = ca.shape[0]
    coords = torch.zeros(L, 37, 3)
    coords[:, CA_ATOM_INDEX] = ca
    mask = torch.zeros(L, 37)
    mask[:, CA_ATOM_INDEX] = 1
    runs = [(t, x.shape[0]) for t, x in segments]
    return runs, coords, mask


def test_axes():
    loop = arc_ca(8, [0.0, 30.0, 0.0])
    # helix of 18 x 100 deg = 5 whole turns, so its principal axis is not tilted by a partial turn
    segs = [(H_, helix_ca(18, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])), (L_, loop),
            (E_, strand_ca(6, [20.0, 0.0, 0.0], [1.0, 0.0, 0.0])), (L_, arc_ca(1, [0.0, 50.0, 0.0])),
            (E_, strand_ca(6, [36.5, 4.8, 0.0], [-1.0, 0.0, 0.0])), (E_, strand_ca(6, [20.0, 9.6, 0.0], [1.0, 0.0, 0.0]))]
    runs, coords, mask = build(segs)
    L = coords.shape[0]
    cm = torch.zeros(L, L)
    keep_all = list(range(len(runs)))
    assert raises(ValueError, lambda: sse_structural_pair_features(cm, coords, mask, runs, keep_all, loop_axis=None)), \
        "loops without loop_axis must raise ValueError"
    zero = sse_structural_pair_features(cm, coords, mask, runs, keep_all, loop_axis="zero")[..., 1]
    e2e = sse_structural_pair_features(cm, coords, mask, runs, keep_all, loop_axis="end_to_end")[..., 1]
    # helix (0) axis is z: perpendicular to the x-direction strands
    assert abs(zero[0, 2]) < 0.05 and abs(zero[0, 4]) < 0.05, zero[0]
    assert zero[2, 4] < -0.99 and zero[2, 5] > 0.99 and zero[4, 5] < -0.99, (zero[2, 4], zero[2, 5], zero[4, 5])
    assert zero[1].abs().max() == 0 and zero[3].abs().max() == 0, "zero mode: loops have no axis"
    v = (loop[-1] - loop[0]) / (loop[-1] - loop[0]).norm()
    assert abs(float(e2e[1, 2]) - float(v[0])) < 1e-4 and abs(float(e2e[1, 0]) - float(v[2])) < 5e-3, (e2e[1], v)  # helix axis tilt ~2e-3
    assert e2e[3].abs().max() == 0, "a 1-CA loop has no end-to-end vector"
    assert torch.equal(zero[[0, 2, 4, 5]][:, [0, 2, 4, 5]], e2e[[0, 2, 4, 5]][:, [0, 2, 4, 5]]), "H/E pairs must not depend on the loop mode"
    print(f"PASS axes: helix vs strands |cos| < 0.05; strands anti -1 / par +1; loop zero -> 0; end_to_end -> unit "
          f"end-to-end vector (cos to strand {float(e2e[1, 2]):.3f} = v_x {float(v[0]):.3f}); 1-CA loop 0; raises without a mode")
    assert element_lengths(runs, [0, 1, 3]).tolist() == [18.0, 8.0, 1.0]
    print("PASS element_lengths follows the runs")


def test_transform_with_loops():
    with tempfile.TemporaryDirectory() as d:
        idx = {"ids": ["q_A"], "element_types": (L_, H_, E_), "elem_feature_mean": torch.tensor(6.0),
               "elem_feature_std": torch.tensor(2.0)}
        p = os.path.join(d, "idx.pt")
        torch.save(idx, p)
        tf = TopologyReferenceTransform(index_path=p, sse_types=(L_, H_, E_), single_token_types=(L_,),
                                        element_types=(L_, H_, E_), elem_features=True, max_topology_he_len=96)
        runs = [(L_, 3), (H_, 12), (L_, 4), (E_, 6), (L_, 2)]
        T = len(runs)
        out = tf.assemble_reference(runs, torch.zeros(T, T), torch.zeros(T, T, 2), length=27)
        toks = out["topology_he_tokens"].tolist()
        assert toks[0] == toks[2] == toks[4] == 2 and toks[1] == tf.alphabet.token(H_, 12) and len(toks) == T, toks
        assert torch.allclose(out["topology_he_pos_raw"], torch.tensor([1.5, 9.0, 17.0, 22.0, 26.0]))
        want = (torch.tensor([3.0, 12.0, 4.0, 6.0, 2.0]) - 6.0) / 2.0
        assert torch.allclose(out["topology_he_elem_feat"][:, 0], want), out["topology_he_elem_feat"]
        assert out["topology_he_feat"].shape == (T, T, N_PAIR_FEATURES)
        print(f"PASS transform: loops are elements (tokens {toks}), midpoints, standardised lengths {want.tolist()}")

        tf_old = TopologyReferenceTransform(index_path=p, sse_types=(H_, E_))
        assert raises(ValueError, tf_old._ensure_loaded)
        print("PASS index guard: a helix/strand transform refuses an index built with loop elements")


if __name__ == "__main__":
    test_alphabet()
    test_axes()
    test_transform_with_loops()
    print("ALL PASS")
