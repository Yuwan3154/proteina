"""Deck example panels, step 1 (user 2026-10-08): superpose template and both outputs onto the native frame.

For each picked query: USalign -TMscore 5 -m (index correspondence, normalised by the native -- the call that scored the
outputs) gives the rotation of structure 1 onto the native; it is applied to every atom and the superposed copy is
written next to the native. The native itself is written unchanged (it defines the frame). Also re-checks that each
regenerated output's TM matches the recorded TM of the T7 row (same seed => same structure).
Usage: python t7_superpose_examples.py PICKS.json TEMPLATE_PDB_DIR OUTPUTS_PDB_ROOTS(comma) OUT_DIR
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_c2c_structures import usalign_tm

USALIGN = "/home/chenxiou/.local/bin/USalign"
picks, tdir, roots, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
os.makedirs(out, exist_ok=True)


def superpose(mobile, native, dst):
    mfile = dst + ".mat"
    subprocess.run([USALIGN, mobile, native, "-TMscore", "5", "-m", mfile], capture_output=True, text=True, check=True)
    rows = [l.split() for l in open(mfile) if re.match(r"^\s*[012]\s", l)][:3]
    t = np.array([float(r[1]) for r in rows]); U = np.array([[float(v) for v in r[2:5]] for r in rows])
    lines = []
    for l in open(mobile):
        if l.startswith("ATOM"):
            x = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]); y = t + U @ x
            l = f"{l[:30]}{y[0]:8.3f}{y[1]:8.3f}{y[2]:8.3f}{l[54:]}"
        lines.append(l)
    open(dst, "w").write("".join(lines)); os.remove(mfile)


P = json.load(open(picks))
for cat, ents in P.items():
    for e in ents:
        st = e["stem"]
        nat = os.path.join(tdir, f"{st}_native.pdb")
        tpl = os.path.join(tdir, f"{st}_template.pdb")
        shutil.copy(nat, os.path.join(out, f"{st}_native.pdb"))
        superpose(tpl, nat, os.path.join(out, f"{st}_template_sup.pdb"))
        for d in ("CB8", "CF"):
            lab = os.path.basename(e[d]["maps_dir"])
            hits = [h for r in roots for h in glob.glob(f"{r}/ex_{lab}/*/{st}/{st}_s{e[d]['sample_index']:02d}_seed00_gen.pdb")]
            assert len(hits) == 1, (st, d, lab, hits)
            g = hits[0]
            # regenerated output must reproduce the recorded TM (vs ITS OWN run's native file)
            tm_regen = usalign_tm(g, os.path.join(os.path.dirname(g), f"{st}_native.pdb"), USALIGN)
            ok = abs(tm_regen - e[d]["tm"]) < 0.01
            # the shared frame (exported native) must be the SAME residues, in order, as the run's own native
            ca = lambda f: np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(f) if l.startswith("ATOM") and l[12:16].strip() == "CA"])
            a1, a2 = ca(nat), ca(os.path.join(os.path.dirname(g), f"{st}_native.pdb"))
            # stageB writes its native after the data pipeline centres it, so compare frame-free: CA-CA distance matrices
            dm = lambda x: np.linalg.norm(x[:, None] - x[None], axis=-1)
            assert a1.shape == a2.shape and np.abs(dm(a1) - dm(a2)).max() < 1e-2, f"{st}: exported native differs from the run's native {a1.shape} vs {a2.shape}"
            superpose(g, nat, os.path.join(out, f"{st}_{d}_sup.pdb"))
            print(f"{cat} {st} {d} sample {e[d]['sample_index']}: recorded TM {e[d]['tm']:.3f} regenerated {tm_regen:.3f} {'OK' if ok else 'MISMATCH'}")
print("[done]", out)
