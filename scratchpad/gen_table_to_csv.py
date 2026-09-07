"""Convert the pre-fix-A generation probe's printed table into the ID-keyed CSV.

The probe at commit 0f95761 prints positional labels (gen000, gen001, ...) and no chain ids -- the
CSV output was only added later. Joining its results to anything therefore needs the loader-order
map, which val_order_ids.py emits.

⛔ Both files must come from the SAME loader order. That is not an assumption to make quietly: the
map prints a length per chain and the probe prints a length per row, so this script REQUIRES them to
agree and aborts if they do not. A silent off-by-one here would attach every chain's score to the
wrong homology value and invert the conclusion.
"""

import argparse
import csv
import re
import sys

ROW = re.compile(
    r"^\s*(gen\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+|n/a)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(MIRROR)?\s*$"
)
MAPROW = re.compile(r"^\s*(\d+)\s+(gen\d+)\s+(\S+)\s+(\d+)\s*$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_log", required=True, help="stdout of the pre-fix-A gen probe")
    ap.add_argument("--order_log", required=True, help="stdout of val_order_ids.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    order = {}
    for line in open(args.order_log):
        m = MAPROW.match(line)
        if m:
            order[m.group(2)] = (m.group(3), int(m.group(4)))

    rows, bad = [], []
    for line in open(args.gen_log):
        m = ROW.match(line)
        if not m:
            continue
        label, L = m.group(1), int(m.group(2))
        if label not in order:
            bad.append(f"{label}: absent from the order map")
            continue
        cid, L_map = order[label]
        if L_map != L:
            bad.append(f"{label}: length {L} in probe vs {L_map} in map -- loader orders DIFFER")
            continue
        tm = "" if m.group(5) == "n/a" else m.group(5)
        rows.append(dict(label=label, chain_id=cid, length=L, ca_rmsd=m.group(3),
                         mirror_rmsd=m.group(4), tm=tm, dist_mae=m.group(6),
                         rg_ratio=m.group(7), mirrored=int(bool(m.group(9))),
                         chirality=m.group(8)))

    if bad:
        print("ABORT -- the two runs are not aligned:", file=sys.stderr)
        for b in bad[:10]:
            print("  " + b, file=sys.stderr)
        return 1
    if not rows:
        print("ABORT -- no probe rows parsed; check the table format", file=sys.stderr)
        return 1

    cols = ["label", "chain_id", "length", "ca_rmsd", "mirror_rmsd", "tm", "dist_mae",
            "rg_ratio", "mirrored", "chirality"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    n_mir = sum(r["mirrored"] for r in rows)
    print(f"wrote {args.out}: {len(rows)} chains, {n_mir} mirrored "
          f"({100.0*n_mir/len(rows):.1f}%), all lengths cross-checked against the order map")
    return 0


if __name__ == "__main__":
    sys.exit(main())
