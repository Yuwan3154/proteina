"""Regression: a per-GRAPH flag must not be padded onto the residue axis.

tri_cb8synth 22501212 OOM'd at the step-0 sanity check trying to allocate 67.85 GiB. Cause:
`seq_dropped` is stored as shape [1], PaddingTransform pads every tensor with dim() >= 1 to
max_size along dim 0, so it became [384]. The model's `reshape(-1, 1, 1)` then produced [384, 1, 1]
and broadcast the sequence embedding from [B, L, dim] to [384, L, dim], inflating the pair grid
[B, N, N, dim] by a factor of B*384.

Unit tests over the transform alone had NO chance of catching this: the transform is correct in
isolation, and the bug only appears once PaddingTransform runs after it -- test the term THROUGH its
consumer.
"""

import torch
from torch_geometric.data import Data

from proteinfoundation.datasets.transforms import PaddingTransform, SequenceDropoutTransform

MAX = 384


def a_graph(L=120):
    g = Data()
    g.coords = torch.randn(L, 37, 3)
    g.coord_mask = torch.ones(L, 37)
    g.residue_type = torch.zeros(L, dtype=torch.long)
    g.mask = torch.ones(L)
    return g


def test_seq_dropped_survives_padding_as_one_value():
    g = SequenceDropoutTransform(prob=1.0).forward(a_graph())
    assert g.seq_dropped.numel() == 1, g.seq_dropped.shape
    g = PaddingTransform(max_size=MAX).forward(g)
    assert g.seq_dropped.numel() == 1, (
        f"seq_dropped was padded to {tuple(g.seq_dropped.shape)}; a per-graph flag must stay scalar"
    )
    assert int(g.seq_dropped) == 1, "the flag's VALUE must survive too"


def test_residue_tensors_are_still_padded():
    """The fix must not turn the padder off for real per-residue tensors."""
    g = PaddingTransform(max_size=MAX).forward(SequenceDropoutTransform(prob=0.0).forward(a_graph(120)))
    assert g.coords.shape[0] == MAX, g.coords.shape
    assert g.mask.shape[0] == MAX, g.mask.shape


def test_model_side_broadcast_cannot_expand_the_batch():
    """`reshape(B, -1)[:, :1, None]` keeps e's shape even if the flag arrives over-long."""
    B, L, D = 2, 16, 8
    e = torch.randn(B, L, D)
    for flag in (torch.tensor([[0], [1]]), torch.zeros(B, MAX, dtype=torch.long)):
        keep = 1.0 - flag.reshape(B, -1)[:, :1, None].to(e.dtype)
        out = e * keep
        assert out.shape == (B, L, D), f"{tuple(out.shape)} from flag {tuple(flag.shape)}"


def test_the_old_reshape_would_have_blown_up():
    """Pin the actual failure, so the regression is unambiguous."""
    B, L, D = 1, 16, 8
    e = torch.randn(B, L, D)
    padded_flag = torch.zeros(B, MAX, dtype=torch.long)
    bad = (1.0 - padded_flag.reshape(-1, 1, 1).to(e.dtype)) * e
    assert bad.shape == (MAX, L, D), tuple(bad.shape)   # the bug: batch axis became 384


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
