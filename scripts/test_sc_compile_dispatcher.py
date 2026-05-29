"""Mechanism test for ``ProteinTransformerAF3.forward()`` SC-compile branch.

Fast unit-style test (~5 s) that proves the four invariants the SC-compile fix
depends on — without booting the full Proteina/Lightning stack:

1. ``use_torch_compile_sc=True`` + ``torch.no_grad()`` dispatches to
   ``_forward_impl_sc``, which goes through ``torch.compile()``.
2. ``use_torch_compile_sc=False`` + ``torch.no_grad()`` falls back to eager
   ``_forward_impl`` (the historical safe default — unchanged behavior).
3. Training-path compile (``use_torch_compile=True`` + ``grad_enabled``) and
   SC-path compile (``use_torch_compile_sc=True`` + ``no_grad``) populate
   **separate** Dynamo cache entries — measured via
   ``dynamo.counters['stats']['unique_graphs']``. This is the actual
   isolation property — if Dynamo collapsed the two into one cache entry,
   the cache would specialize on grad_mode and start thrashing.
4. Repeated alternation between training and SC paths does NOT cause runaway
   recompilation: the unique-graph counter stays at exactly 2 after warmup.

We monkey-patch ``_forward_impl`` to a trivial deterministic op so the test
runs in milliseconds without the network's feature builders / data deps. The
dispatcher logic in ``forward()`` is what's actually under test.

Run:
    /home/ubuntu/miniforge3/envs/cue_openfold/bin/python \\
        scripts/test_sc_compile_dispatcher.py

Companion: ``scripts/test_sc_compile.py`` runs the same test end-to-end through
``proteinfoundation.train`` on a tiny overfit dataset for additional confidence
(slow, ~15 min — two compile cycles).
"""

import os
os.environ.pop("TORCH_LOGS", None)

import sys

import torch
from torch import nn

from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3


class TinyForwardModel(ProteinTransformerAF3):
    """Subclass that bypasses ProteinTransformerAF3.__init__ — we don't need
    the full kwargs grid for a forward-dispatch test. ``_forward_impl`` is
    replaced with a trivial op; ``_forward_impl_sc`` is inherited unchanged."""

    def __init__(self, use_torch_compile, use_torch_compile_sc):
        nn.Module.__init__(self)
        self.use_torch_compile = use_torch_compile
        self.use_torch_compile_sc = use_torch_compile_sc
        self.w = nn.Parameter(torch.randn(8, 8))

    def _forward_impl(self, batch_nn):
        return torch.relu(batch_nn["x"] @ self.w) * 2.0


def make_batch(B=2, N=8, device="cuda"):
    return {"x": torch.randn(B, N, 8, device=device)}


def reset_dynamo():
    import torch._dynamo as dynamo
    dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()


def unique_graphs():
    from torch._dynamo.utils import counters
    return counters["stats"].get("unique_graphs", 0)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required for this test (Inductor needs a real backend).")
        return 2

    device = "cuda"
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Test 1: dispatcher routing
    # ------------------------------------------------------------------
    m_eager = TinyForwardModel(use_torch_compile=False, use_torch_compile_sc=False).to(device)
    batch = make_batch(device=device)
    with torch.no_grad():
        y = m_eager(batch)
    assert y.shape == (2, 8, 8)
    assert not hasattr(m_eager, "_forward_compiled_sc"), \
        "use_torch_compile_sc=False should NOT create a compile object"
    print("[PASS] use_torch_compile_sc=False → eager path, no compile created")

    m_sc = TinyForwardModel(use_torch_compile=False, use_torch_compile_sc=True).to(device)
    m_sc.w.data.copy_(m_eager.w.data)  # same weights for output parity
    reset_dynamo()
    with torch.no_grad():
        y_sc = m_sc(batch)
    assert hasattr(m_sc, "_forward_compiled_sc"), \
        "compile object should be cached on instance"
    n1 = unique_graphs()
    print(f"[PASS] use_torch_compile_sc=True → compiled (unique_graphs={n1})")

    torch.testing.assert_close(y, y_sc, rtol=1e-5, atol=1e-5)
    print("[PASS] compiled SC output == eager SC output")

    # ------------------------------------------------------------------
    # Test 2: separate Dynamo caches for train vs SC
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    m_both = TinyForwardModel(use_torch_compile=True, use_torch_compile_sc=True).to(device)
    reset_dynamo()

    y_train = m_both(make_batch(device=device))  # grad on
    assert y_train.requires_grad
    n_train = unique_graphs()
    assert n_train >= 1, f"expected >=1 unique graph after train compile, got {n_train}"
    print(f"[PASS] training forward compiled (unique_graphs={n_train})")

    with torch.no_grad():
        _ = m_both(make_batch(device=device))
    n_after_sc = unique_graphs()
    assert n_after_sc > n_train, (
        f"SC compile did not create a new graph: train={n_train}, sc={n_after_sc}. "
        "Both caches collapsed — grad-mode thrash would return."
    )
    print(f"[PASS] SC and train have SEPARATE caches "
          f"(train graphs={n_train}, total after SC={n_after_sc})")

    # ------------------------------------------------------------------
    # Test 3: no recompile thrash under alternation
    # ------------------------------------------------------------------
    baseline = unique_graphs()
    for _ in range(10):
        _ = m_both(make_batch(device=device))                  # train (grad=True)
        with torch.no_grad():
            _ = m_both(make_batch(device=device))              # SC    (grad=False)
    after = unique_graphs()
    assert after == baseline, (
        f"Recompilation triggered during alternation: {baseline} → {after}"
    )
    print(f"[PASS] 10× train/SC alternation = 0 new compiles (graphs stable at {after})")

    # ------------------------------------------------------------------
    # Test 4: _forward_impl_sc has a DISTINCT __code__ object
    # ------------------------------------------------------------------
    # The Python-level invariant the whole fix depends on. If a future edit
    # ever turns _forward_impl_sc into a property/alias of _forward_impl,
    # the two Dynamo caches silently re-merge and the fix becomes a no-op.
    assert (
        ProteinTransformerAF3._forward_impl_sc.__code__
        is not ProteinTransformerAF3._forward_impl.__code__
    ), "_forward_impl_sc must have a distinct code object from _forward_impl"
    print("[PASS] _forward_impl_sc.__code__ is distinct from _forward_impl.__code__")

    print("\nAll dispatcher tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
