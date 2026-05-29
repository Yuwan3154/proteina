"""Follow-up unit tests for the SC-compile fix.

Q1 — Intermittent SC invocation under compile
-----------------------------------------------
In the trainer, ``use_sc = random.random() > 0.5 and self.cfg_exp.training.self_cond``
is a per-MINIBATCH boolean. About half the minibatches don't take the SC route
at all — `_set_self_cond` just initializes `batch[sc_key]=zeros` and returns
without calling the SC model. This test confirms the compile object survives
gaps of non-SC minibatches and never triggers extra recompiles.

Q2 — Is the nn_sc deepcopy still required for compile-cache isolation?
----------------------------------------------------------------------
Historically, `nn_sc = copy.deepcopy(nn)` existed in part to keep grad-mode
specialization off of the training `_forward_impl`. With the new
`_forward_impl_sc` wrapper, the SC cache is keyed on the wrapper's own
``__code__`` object — independent of the training cache — even when SC calls
the SAME instance as training. This test runs two configurations side-by-side:
    Case A: SC via a deepcopy (current default)
    Case B: SC via the same instance as training (no deepcopy)
Both should produce exactly 2 unique compiled graphs and identical SC outputs.

If both Q1 and Q2 pass, the deepcopy is no longer required FOR COMPILE-CACHE
ISOLATION. It may still be useful for EMA-style delayed SC targets
(`training.self_cond_copy_update_every` controls that lag), which is a
learning-stability decision separate from this test.

Run:
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \
        scripts/test_sc_compile_q1_q2.py
"""
import os
os.environ.pop("TORCH_LOGS", None)

import copy
import random
import sys

import torch
from torch import nn

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


class TinyModel(ProteinTransformerAF3):
    """Subclass that bypasses ProteinTransformerAF3.__init__; only the
    forward() dispatcher is under test here, not the network builders."""

    def __init__(self, use_torch_compile, use_torch_compile_sc):
        nn.Module.__init__(self)
        self.use_torch_compile = use_torch_compile
        self.use_torch_compile_sc = use_torch_compile_sc
        self.w = nn.Parameter(torch.randn(8, 8))

    def _forward_impl(self, batch_nn):
        return torch.relu(batch_nn["x"] @ self.w) * 2.0


def mb(seed, B=2, N=8, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return {"x": torch.randn(B, N, 8, generator=g, device=device)}


def unique_graphs():
    from torch._dynamo.utils import counters
    return counters["stats"].get("unique_graphs", 0)


def reset_dynamo():
    import torch._dynamo as dynamo
    from torch._dynamo.utils import counters
    dynamo.reset()
    counters.clear()


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------
def test_q1_intermittent_sc(device):
    print("\n=== Q1: intermittent SC invocation under compile ===")
    torch.manual_seed(0)
    m = TinyModel(use_torch_compile=True, use_torch_compile_sc=True).to(device)
    reset_dynamo()

    random.seed(7)
    pattern = [random.random() > 0.5 for _ in range(20)]
    n_sc = sum(pattern)
    print(f"  pattern: {''.join('S' if x else '.' for x in pattern)}  "
          f"({n_sc} SC steps, {20 - n_sc} non-SC)")

    for step, take_sc in enumerate(pattern):
        _ = m(mb(step, device=device))                       # train (grad=True)
        if take_sc:
            with torch.no_grad():
                _ = m(mb(step + 100, device=device))         # SC (grad=False)
    n1 = unique_graphs()
    assert n1 == 2, f"expected 2 graphs (train + SC), got {n1}"
    print(f"  [PASS] (a)+(b): {n1} graphs, intermittent SC = no extra compiles")

    # After a 5-minibatch non-SC gap, the cached SC compile is still reusable.
    for _ in range(5):
        _ = m(mb(999, device=device))                        # train only
    with torch.no_grad():
        _ = m(mb(999, device=device))                        # SC
    assert unique_graphs() == n1, "SC compile was re-triggered after gap"
    print(f"  [PASS] (c): SC compile survives gap of non-SC steps")


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------
def test_q2_deepcopy_optional(device):
    print("\n=== Q2: is the nn_sc deepcopy still required? ===")

    # Case A — SC via deepcopy (current default)
    torch.manual_seed(0)
    nn_a = TinyModel(use_torch_compile=True, use_torch_compile_sc=False).to(device)
    nn_sc_a = copy.deepcopy(nn_a)
    for p in nn_sc_a.parameters():
        p.requires_grad = False
    nn_sc_a.use_torch_compile = False
    nn_sc_a.use_torch_compile_sc = True

    reset_dynamo()
    random.seed(7)
    pattern = [random.random() > 0.5 for _ in range(20)]
    sc_outputs_a = []
    for step, take_sc in enumerate(pattern):
        _ = nn_a(mb(step, device=device))
        if take_sc:
            with torch.no_grad():
                sc_outputs_a.append(nn_sc_a(mb(step + 100, device=device)).clone())
    n_a = unique_graphs()
    print(f"  Case A (with deepcopy):    unique_graphs = {n_a}")
    assert n_a == 2

    # Case B — no deepcopy; SC through self.nn
    torch.manual_seed(0)
    nn_b = TinyModel(use_torch_compile=True, use_torch_compile_sc=True).to(device)
    nn_b.w.data.copy_(nn_a.w.data)

    reset_dynamo()
    random.seed(7)
    pattern = [random.random() > 0.5 for _ in range(20)]
    sc_outputs_b = []
    for step, take_sc in enumerate(pattern):
        _ = nn_b(mb(step, device=device))
        if take_sc:
            with torch.no_grad():
                sc_outputs_b.append(nn_b(mb(step + 100, device=device)).clone())
    n_b = unique_graphs()
    print(f"  Case B (NO deepcopy):       unique_graphs = {n_b}")
    assert n_b == 2, (
        f"Case B expected 2 graphs, got {n_b}. Cache isolation failed when "
        f"SC shares self.nn with training."
    )

    # Case C — Math parity between A and B
    assert len(sc_outputs_a) == len(sc_outputs_b)
    for a, b in zip(sc_outputs_a, sc_outputs_b):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    print(f"  [PASS] A and B both = {n_a} graphs; all {len(sc_outputs_a)} SC "
          "outputs identical")
    print("  → The deepcopy is OPTIONAL for compile-cache isolation.")
    print("  → It may still be useful for EMA-style delayed SC targets;")
    print("    `training.self_cond_copy_update_every` controls that lag.")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 2
    device = "cuda"
    test_q1_intermittent_sc(device)
    test_q2_deepcopy_optional(device)
    print("\nAll Q1+Q2 tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
