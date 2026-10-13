"""T8 clash gate (user 2026-10-13: non-bonded CA-CA < 3.4 A, "<=4", before the cap) and the USalign residue-count guard
(ProteinEBMalign's HETATM label shift: USalign reads ATOM records only). Run (PYTHONPATH=repo) on a node with USalign:
python scratchpad/test_clash_gate_and_usalign_guard.py [USALIGN]"""

import os
import subprocess
import sys
import tempfile

import numpy as np

import proteinfoundation.utils.precompute_synthetic_topology_index as B


def refused(pq, pr, n_q, n_r, expect):
    """usalign_pairs must RAISE (child process: exit code + message), never return shifted pairs."""
    code = (f"import proteinfoundation.utils.precompute_synthetic_topology_index as B; B._USALIGN = {B._USALIGN!r}; "
            f"B.usalign_pairs({pq!r}, {pr!r}, {n_q}, {n_r})")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and expect in out.stderr, (n_q, n_r, out.returncode, out.stderr[-300:])


# --- ca_clash_pairs: residue-index separation, not written order; adjacent steps never count
ca = np.stack([np.arange(10) * 3.8, np.zeros(10), np.zeros(10)], 1).astype(np.float32)
present = np.ones(10, bool)
assert B.ca_clash_pairs(ca, present, 3.4) == 0
ca2 = ca.copy()
ca2[6] = ca2[1] + [0, 3.0, 0]                     # |i-j| = 5 at 3.0 A -> 1 clash
ca2[3] = ca2[2] + [1.0, 0, 0]                     # adjacent step 1.0 A -> not a clash
n = B.ca_clash_pairs(ca2, present, 3.4)
assert n == 1, n
gap = present.copy()
gap[4] = False
ca3 = ca.copy()
ca3[5] = ca3[3] + [0, 3.0, 0]                     # 3 and 5 adjacent in written order, |i-j| = 2 -> counts
assert B.ca_clash_pairs(ca3, gap, 3.4) == 1
print("PASS ca_clash_pairs")


# --- select_pack_rows: keep applied BEFORE the cap; cap None = all in range
class FakePack:
    def __init__(self, tms):
        self.col = {"tm_native": np.asarray(tms, dtype=np.float32), "draw": np.arange(len(tms))}

    def __contains__(self, c):
        return c == "a_A"

    def rows(self, c):
        return range(len(self.col["draw"]))


tp = FakePack(np.random.default_rng(0).uniform(0.45, 0.95, 200))
inr = B.select_pack_rows(tp, "a_A", 0.5, 0.9, None)
bad = set(inr[::3])
sel = B.select_pack_rows(tp, "a_A", 0.5, 0.9, 64, keep=lambda r: r not in bad)
assert len(sel) == 64 and not (set(sel) & bad) and set(sel) <= set(inr), len(sel)
assert len(inr) == int(((tp.col["tm_native"] >= 0.5) & (tp.col["tm_native"] <= 0.9)).sum())
print(f"PASS keep-before-cap: {len(inr)} in range, {len(bad)} gated, 64 selected from the survivors")

# --- usalign_pairs guard (needs the binary)
B._USALIGN = sys.argv[1] if len(sys.argv) > 1 else B._USALIGN
if not os.path.exists(B._USALIGN):
    raise SystemExit(f"no USalign at {B._USALIGN}")
t = np.arange(40) * 100 / 180 * np.pi
helix = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(40)], 1).astype(np.float32)
pres = np.ones(40, bool)
pres[[7, 20]] = False                              # residues without CA are skipped in the file and mapped back
with tempfile.TemporaryDirectory() as td:
    pq, pr = os.path.join(td, "q.pdb"), os.path.join(td, "r.pdb")
    qmap = B.write_ca_pdb(helix, pres, pq)
    rmap = B.write_ca_pdb(helix[:35] + 0.3, np.ones(35, bool), pr)
    pairs, tm = B.usalign_pairs(pq, pr, len(qmap), len(rmap))
    assert tm == tm and 0 < tm <= 1, tm
    assert len(qmap) == 38 and all(qmap[a] < 40 and rmap[b] < 35 for a, b in pairs)
    self_pairs, tm_self = B.usalign_pairs(pq, pq, len(qmap), len(qmap))
    assert tm_self > 0.999 and all(a == b for a, b in self_pairs) and len(self_pairs) == len(qmap)
    print(f"PASS usalign_pairs: TM {tm:.4f} parsed (Structure_1 label), self-alignment identity on {len(qmap)} residues")
    for wrong in ((len(qmap) + 1, len(rmap)), (len(qmap), len(rmap) - 1)):
        refused(pq, pr, *wrong, "RuntimeError: residues_read")
    # the ProteinEBMalign failure: a HETATM residue in the file is invisible to USalign -> must be refused, not shifted
    lines = open(pq).read().splitlines()
    lines[10] = "HETATM" + lines[10][6:]
    open(pq, "w").write("\n".join(lines) + "\n")
    refused(pq, pr, len(qmap), len(rmap), "RuntimeError: residues_read_37_")
    print("PASS guard: wrong counts and a HETATM residue (read 37 of 38) are refused")
print("ALL PASS")
