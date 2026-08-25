"""Unit tests for proteinfoundation.utils.sampletrace. Stdlib + torch/numpy only, no pytest.

Run: python tests/test_sampletrace.py   (exits non-zero on any failure)

The property that matters is sc_is_prev_pred: it must be True only when the tensor handed to
the network is genuinely the PREVIOUS step's prediction. A test that merely checks the key is
present would pass even if self-conditioning fed stale or zeroed data.
"""

import os
import shutil
import sys
import tempfile
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.utils import sampletrace

_FAILURES = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def case(fn):
    name = fn.__name__
    d = tempfile.mkdtemp(prefix="sampletrace_")
    prev = os.environ.get("SAMPLETRACE")
    prev_every = os.environ.get("SAMPLETRACE_EVERY")
    sampletrace.reset()
    os.environ.pop("SAMPLETRACE", None)
    os.environ.pop("SAMPLETRACE_EVERY", None)
    try:
        fn(d)
        print(f"  PASS  {name}")
    except Exception as e:
        _FAILURES.append((name, traceback.format_exc()))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    finally:
        sampletrace.reset()
        for k, v in (("SAMPLETRACE", prev), ("SAMPLETRACE_EVERY", prev_every)):
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        shutil.rmtree(d, ignore_errors=True)


def t_disabled_records_nothing(d):
    sampletrace.record_args(dt=0.02)
    sampletrace.record_step(0, 0.0, {}, torch.zeros(2, 2), torch.zeros(2, 2))
    check(sampletrace.summary()["n_steps"] == 0, "must record nothing when disabled")
    check(sampletrace.dump(os.path.join(d, "x")) is None, "dump must no-op when disabled")


def t_args_recorded(d):
    os.environ["SAMPLETRACE"] = "1"
    sampletrace.record_args(dt=0.02, nsteps=50, sampling_mode="sc", self_cond=True)
    a = sampletrace.summary()["args"]
    check(a["dt"] == 0.02 and a["nsteps"] == 50.0, f"bad args {a}")
    check(a["sampling_mode"] == "sc", "non-numeric args must pass through unchanged")


def t_sc_absent_first_step(d):
    """Step 0 legitimately has no previous prediction, so no sc key."""
    os.environ["SAMPLETRACE"] = "1"
    pred = torch.ones(3, 3)
    sampletrace.record_step(0, 0.0, {}, torch.zeros(3, 3), pred)
    s = sampletrace.summary()["steps"][0]
    check(s["sc_present"] is False, "sc must be absent at step 0")
    check(s["sc_norm"] == 0.0, "sc_norm must be 0 when absent")
    check(s["sc_is_prev_pred"] is False, "cannot be prev-pred at step 0")


def t_sc_detected_as_prev_pred(d):
    """The real property: sc at step N must BE the prediction from step N-1."""
    os.environ["SAMPLETRACE"] = "1"
    pred0 = torch.rand(4, 4)
    sampletrace.record_step(0, 0.0, {}, torch.zeros(4, 4), pred0)
    pred1 = torch.rand(4, 4)
    sampletrace.record_step(1, 0.1, {"contact_map_sc": pred0.clone()}, torch.zeros(4, 4), pred1)
    s = sampletrace.summary()["steps"][1]
    check(s["sc_present"] is True, "sc must be present at step 1")
    check(s["sc_is_prev_pred"] is True, "sc SHOULD equal the previous prediction")
    check(s["sc_norm"] > 0, "a real prediction must have nonzero norm")


def t_stale_sc_is_caught(d):
    """A wrong-but-present sc (e.g. zeros or a stale tensor) must NOT report as prev-pred."""
    os.environ["SAMPLETRACE"] = "1"
    pred0 = torch.rand(4, 4) + 1.0
    sampletrace.record_step(0, 0.0, {}, torch.zeros(4, 4), pred0)
    sampletrace.record_step(1, 0.1, {"contact_map_sc": torch.zeros(4, 4)}, torch.zeros(4, 4), torch.rand(4, 4))
    s = sampletrace.summary()["steps"][1]
    check(s["sc_present"] is True, "key is present")
    check(s["sc_is_prev_pred"] is False, "zeros must NOT be reported as the previous prediction")
    check(s["sc_norm"] == 0.0, "zeros must report norm 0")


def t_snapshots_gated_by_every(d):
    os.environ["SAMPLETRACE"] = "1"
    os.environ["SAMPLETRACE_EVERY"] = "2"
    for i in range(5):
        sampletrace.record_step(i, i / 5, {}, torch.zeros(2, 2), torch.rand(2, 2))
    p = sampletrace.dump(os.path.join(d, "tr"))
    check(p is not None, "dump should write when enabled")
    import numpy as np
    z = np.load(os.path.join(d, "tr_maps.npz"))
    check(list(z["steps"]) == [0, 2, 4], f"expected steps 0,2,4 got {list(z['steps'])}")


def t_topology_presence_recorded(d):
    """An arm sampling UNCONDITIONED must be distinguishable from one holding a real reference.
    The MASK fallback is a single valid element, so presence alone is not enough -- the count is."""
    os.environ["SAMPLETRACE"] = "1"
    real = torch.tensor([[5, 9, 12, -1]])
    sampletrace.record_step(0, 0.0, {"topology_he_tokens": real}, torch.zeros(2, 2), torch.rand(2, 2))
    sampletrace.record_step(1, 0.1, {"topology_he_tokens": torch.tensor([[1]])}, torch.zeros(2, 2), torch.rand(2, 2))
    sampletrace.record_step(2, 0.2, {}, torch.zeros(2, 2), torch.rand(2, 2))
    st = sampletrace.summary()["steps"]
    check(st[0]["topo_present"] and st[0]["topo_n_valid"] == 3, "real reference: 3 valid elements")
    check(st[1]["topo_present"] and st[1]["topo_n_valid"] == 1, "MASK fallback: 1 valid element")
    check(not st[2]["topo_present"] and st[2]["topo_n_valid"] == 0, "absent: no topology at all")


def t_reset_clears(d):
    os.environ["SAMPLETRACE"] = "1"
    sampletrace.record_step(0, 0.0, {}, torch.zeros(2, 2), torch.rand(2, 2))
    sampletrace.reset()
    check(sampletrace.summary()["n_steps"] == 0, "reset must clear steps")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    print(f"running {len(tests)} sampletrace tests")
    for fn in tests:
        case(fn)
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED:")
        for n, tb in _FAILURES:
            print(f"--- {n} ---\n{tb}")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")
