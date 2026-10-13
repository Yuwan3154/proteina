"""select_pack_rows (user 2026-10-13 "(b)" cap 64, evenly over the TM range in TM value): in-range only, at most cap,
distinct, deterministic, uniform in value, all kept when <= cap, None for a chain absent from the pack, repeated draws
refused. Run (PYTHONPATH=repo, nsrun): python scratchpad/test_select_pack_rows.py"""

import numpy as np

from proteinfoundation.utils.precompute_synthetic_topology_index import select_pack_rows


class FakePack:
    def __init__(self, tms, draws):
        self.col = {"tm_native": np.asarray(tms, dtype=np.float32), "draw": np.asarray(draws)}
        self.chain_row = {"a_A": 0}

    def __contains__(self, c):
        return c in self.chain_row

    def rows(self, c):
        return range(len(self.col["draw"]))


rng = np.random.default_rng(0)
tms = np.concatenate([rng.uniform(0.4, 0.5, 20), rng.beta(8, 2, 140) * 0.4 + 0.5])  # dense near 0.85, as real packs are
tp = FakePack(tms, np.arange(len(tms)))
sel = select_pack_rows(tp, "a_A", 0.5, 0.9, 64)
v = np.sort(tp.col["tm_native"][sel])
assert len(sel) == 64 == len(set(sel)) and v.min() >= 0.5 and v.max() <= 0.9
assert sel == select_pack_rows(tp, "a_A", 0.5, 0.9, 64)
gaps = np.diff(v)
print(f"PASS 64 distinct in-range rows, deterministic; TM span {v.min():.3f}-{v.max():.3f}, median gap {np.median(gaps):.4f} "
      f"(uniform target 0.4/63 = {0.4 / 63:.4f}), max gap {gaps.max():.4f}")
small = FakePack([0.55, 0.6, 0.95], [3, 1, 2])
assert select_pack_rows(small, "a_A", 0.5, 0.9, 64) == [0, 1]
assert select_pack_rows(small, "b_B", 0.5, 0.9, 64) is None
dup = FakePack([0.6, 0.7], [5, 5])
try:
    select_pack_rows(dup, "a_A", 0.5, 0.9, 64)
    raise SystemExit("FAIL: repeated draws accepted")
except AssertionError:
    pass
print("PASS <= cap keeps all in-range rows; absent chain -> None; repeated draws refused")
print("ALL PASS")
