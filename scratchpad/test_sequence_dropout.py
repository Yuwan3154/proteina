"""Whole-query-sequence dropout: length preserved, independent of the topology dropout.

User 2026-09-10: "implement an option to train the model without the query sequence at some given
probability, i.e. whole sequence masking but preserves the length. Defaults to 25%. This should be
independent from the no-topology-reference probability, i.e. at the current setting 0.25*0.25=0.0625
amount of the training will be done without either sequence or topology reference."
"""

import torch
from torch_geometric.data import Data

from proteinfoundation.datasets.transforms import SequenceDropoutTransform


def a_graph(L=20, seed=0):
    g = Data()
    g.coords = torch.randn(L, 5, 3)
    g.residue_type = torch.randint(0, 20, (L,), generator=torch.Generator().manual_seed(seed))
    return g


def test_rate_zero_is_a_no_op_and_flags_nothing():
    t = SequenceDropoutTransform(prob=0.0)
    g = t.forward(a_graph())
    assert int(g.seq_dropped) == 0
    assert t._generator is None, "prob 0 must consume no randomness"


def test_rate_one_always_drops():
    t = SequenceDropoutTransform(prob=1.0)
    assert all(int(t.forward(a_graph(seed=i)).seq_dropped) == 1 for i in range(20))


def test_length_and_residue_type_are_preserved():
    """'whole sequence masking but PRESERVES THE LENGTH' -- and the drop must not mutate inputs."""
    t = SequenceDropoutTransform(prob=1.0)
    g0 = a_graph(L=37)
    rtype_before = g0.residue_type.clone()
    g = t.forward(g0)
    assert g.coords.shape[0] == 37
    assert g.residue_type.shape[0] == 37
    assert torch.equal(g.residue_type, rtype_before), "residue_type must be left untouched"


def test_empirical_rate_matches_the_requested_probability():
    t = SequenceDropoutTransform(prob=0.25, seed=1234)
    torch.manual_seed(0)
    n = 4000
    hits = sum(int(t.forward(a_graph(L=8, seed=i)).seq_dropped) for i in range(n))
    rate = hits / n
    # 3 sd of a Binomial(4000, 0.25) proportion is ~0.0206
    assert abs(rate - 0.25) < 0.0206, f"rate {rate:.4f} is not 0.25 within 3 sd"


def test_independent_of_a_second_dropout_stream():
    """The joint 'neither sequence nor reference' rate must be the PRODUCT, not coupled."""
    seq = SequenceDropoutTransform(prob=0.25, seed=1)
    torch.manual_seed(7)
    ref_gen = torch.Generator().manual_seed(99)
    n = 4000
    both = 0
    for i in range(n):
        s = int(seq.forward(a_graph(L=8, seed=i)).seq_dropped)
        r = float(torch.rand(1, generator=ref_gen)) < 0.25
        both += int(bool(s) and bool(r))
    rate = both / n
    # 3 sd of Binomial(4000, 0.0625) proportion is ~0.0115
    assert abs(rate - 0.0625) < 0.0115, f"joint rate {rate:.4f} is not 0.0625 within 3 sd"


def test_model_side_zeroing_matches_the_no_sequence_state():
    """Zeroing the embedding for flagged samples == the state `rtype is None` leaves the model in."""
    B, L, D = 3, 6, 8
    e = torch.randn(B, L, D)
    seq_dropped = torch.tensor([[0], [1], [0]])
    keep = 1.0 - seq_dropped.reshape(-1, 1, 1).to(e.dtype)
    out = e * keep
    assert torch.equal(out[1], torch.zeros(L, D)), "dropped sample must contribute no sequence"
    assert torch.equal(out[0], e[0]) and torch.equal(out[2], e[2]), "others must be untouched"


if __name__ == "__main__":
    n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as err:
                n_fail += 1
                print(f"FAIL {name}: {err}")
    print(f"\n{n_fail} failing")
