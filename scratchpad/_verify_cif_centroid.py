import glob
import math
import os
import sys


def parse(path):
    chains = {}
    with open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                idx = {c.split(".")[1]: k for k, c in enumerate(cols)}
                while j < len(lines) and not lines[j].startswith("#") and lines[j].strip():
                    f = lines[j].split()
                    if len(f) >= len(cols):
                        if f[idx["label_atom_id"]] == "CA" and f[idx["group_PDB"]] == "ATOM":
                            ch = f[idx["label_asym_id"]]
                            xyz = (
                                float(f[idx["Cartn_x"]]),
                                float(f[idx["Cartn_y"]]),
                                float(f[idx["Cartn_z"]]),
                            )
                            chains.setdefault(ch, []).append(xyz)
                    j += 1
                return chains
            i = j
        else:
            i += 1
    return chains


paths = []
for a in sys.argv[1:]:
    paths.extend(sorted(glob.glob(os.path.join(a, "*.cif")))[:80] if os.path.isdir(a) else [a])

rows = []
for p in paths:
    for ch, xs in parse(p).items():
        n = len(xs)
        if n < 50:
            continue
        cx = sum(x[0] for x in xs) / n
        cy = sum(x[1] for x in xs) / n
        cz = sum(x[2] for x in xs) / n
        c = math.sqrt(cx * cx + cy * cy + cz * cz)
        rg = math.sqrt(
            sum((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2 for x in xs) / n
        )
        rows.append((p.split("/")[-1], ch, n, c, rg))

rows.sort(key=lambda r: -r[3])
for r in rows[:40]:
    print(
        f"{r[0]:<28s} chain {r[1]:<4s} n={r[2]:<5d} |centroid|={r[3]:8.2f}  "
        f"Rg={r[4]:6.2f}  ratio={r[3] / r[4]:6.2f}"
    )
if rows:
    cs = sorted(r[3] for r in rows)
    print(
        f"\nN={len(cs)} chains  median |centroid| = {cs[len(cs) // 2]:.2f} A   "
        f"min {cs[0]:.2f}  max {cs[-1]:.2f}"
    )
