"""Per-group summary of t8_rosetta_rescue.py / t8_tmol_rescue.py result TSVs. Usage: python scratchpad/t8_rescue_summary.py TSV [TSV ...]"""

import sys

import numpy as np

for path in sys.argv[1:]:
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    h, rows = rows[0], rows[1:]
    col = {k: i for i, k in enumerate(h)}
    print(f"== {path.split('/')[-1]} ({len(rows)} rows)")
    for g in ("gated", "kept_1_k", "kept_0"):
        for proto in sorted({r[col["protocol"]] for r in rows}):
            s = [r for r in rows if r[col["group"]] == g and r[col["protocol"]] == proto]
            if not s:
                continue
            f = lambda k: np.array([float(r[col[k]]) for r in s])
            cb, ca = f("clash_before"), f("clash_after")
            dtm = f("tm_after") - f("tm_before")
            sec = f("sec") if "sec" in col else f("sec_batch") / f("batch")
            print(f"  {g:9s} {proto:15s} n={len(s):2d} | clashes median {np.median(cb):4.0f} -> {np.median(ca):3.0f}, "
                  f"<=4 after {int((ca <= 4).sum()):2d}/{len(s)}, 0 after {int((ca == 0).sum()):2d}/{len(s)} | "
                  f"CA RMSD med {np.median(f('rmsd_ca')):.2f} max {f('rmsd_ca').max():.2f} A | max step med "
                  f"{np.median(f('step_before')):.2f} -> {np.median(f('step_after')):.2f} (max {f('step_before').max():.1f} -> "
                  f"{f('step_after').max():.1f}) | dTM med {np.median(dtm):+.3f} min {dtm.min():+.3f} | E/res med "
                  f"{np.median(f('e_before')):.1f} -> {np.median(f('e_after')):.1f} | s/tpl {np.median(sec):.1f}"
                  + (f" | DSSP same {np.median(f('dssp_same')):.3f}" if "dssp_same" in col else ""))
