"""Compute query-vs-topology-reference TM for the non-self validation pairs, and bin by it.

⭐ WHY. The non-self validation arm currently reports ONE number (~0.55 mean vs a 0.65 self
ceiling, a gap of ~0.10). The D36m TEST-set experiment saw a far larger gap (0.4976 -> 0.1798,
~0.32) and established r(refTM, precision) = +0.694 with a median date-clean refTM of 0.50. Those
two facts are only compatible if the validation set's retrieved templates are BETTER than the test
set's -- validation chains are drawn from training clusters. If so, "non-self" is not one task but
a family indexed by reference quality, and a single mean hides the thing that actually predicts
test-time behaviour.

This computes refTM per (query, reference) pair so the arm can be binned at 0.2 TM. It needs no
GPU: it writes CA-only PDBs from the processed .pt coordinates and calls USalign.

⛔ -TMscore 0 is correct HERE. Query and reference are DIFFERENT chains of different length and
sequence -- the cross-protein case. (-TMscore 5 is for same-sequence pairs, e.g. a generated
structure against its own native.)
⛔ Read the TM normalised by the TRUE NATIVE -- the QUERY, whose fold is the thing being realised --
never max() of the two normalisations, which inflates by ~1.5-2x.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

CA_LINE = "ATOM  %5d  CA  ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n"


def write_ca_pdb(path, ca):
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(ca):
            fh.write(CA_LINE % (i + 1, i + 1, float(x), float(y), float(z)))
        fh.write("END\n")


def load_ca(pt_path):
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    coords = d["coords"] if isinstance(d, dict) else getattr(d, "coords")
    coords = torch.as_tensor(coords).float()
    return (coords[:, 1, :] if coords.dim() == 3 else coords).numpy()


def usalign_tm(query_pdb, ref_pdb, usalign):
    """TM of the REFERENCE against the QUERY, normalised by the QUERY (Structure_1 here).

    USalign prints one line per normalisation. We call it as (reference, query) and read the
    normalisation by Structure_2 = the query, so the score answers "how much of the QUERY's fold
    does this template cover", which is the quantity that should predict realisability.
    """
    try:
        out = subprocess.run([usalign, ref_pdb, query_pdb, "-TMscore", "0"],
                             capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "TM-score=" in line and "Structure_2" in line:
            m = re.search(r"TM-score=\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None


def sharded(root, stem):
    """processed/<shard>/<stem>.pt -- the layout pdb_data uses."""
    for cand in (os.path.join(root, "processed", stem[1:3], f"{stem}.pt"),
                 os.path.join(root, "processed", f"{stem}.pt")):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_list", required=True)
    ap.add_argument("--index",
                    default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt")
    ap.add_argument("--data_root", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train")
    ap.add_argument("--tmp", default="/orcd/scratch/orcd/011/chenxiou/.tmp/reftm")
    ap.add_argument("--usalign", default="USalign")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

    t = TopologyReferenceTransform(index_path=args.index, drop_prob=0.0, seed=args.seed)
    t._ensure_loaded()
    stems = [l.strip() for l in open(args.chain_list) if l.strip()]

    rows, skipped = [], []
    print(f"{'query':>10} {'reference':>10} {'Lq':>5} {'Lr':>5} {'refTM':>7}")
    for s in stems:
        got = t.nonself_reference(s, 128, seed=args.seed)
        if got is None:
            skipped.append((s, "no non-self mate"))
            continue
        ref_stem = got[1]
        qp, rp = sharded(args.data_root, s), sharded(args.data_root, ref_stem)
        if qp is None or rp is None:
            skipped.append((s, f"missing .pt ({'query' if qp is None else 'ref'})"))
            continue
        ca_q, ca_r = load_ca(qp), load_ca(rp)
        fq = os.path.join(args.tmp, f"{s}.pdb")
        fr = os.path.join(args.tmp, f"{ref_stem}.pdb")
        write_ca_pdb(fq, ca_q)
        write_ca_pdb(fr, ca_r)
        tm = usalign_tm(fq, fr, args.usalign)
        if tm is None:
            skipped.append((s, "USalign gave no parseable TM"))
            continue
        rows.append((s, ref_stem, len(ca_q), len(ca_r), tm))
        print(f"{s:>10} {ref_stem:>10} {len(ca_q):>5} {len(ca_r):>5} {tm:>7.3f}", flush=True)

    if not rows:
        print("\nno pairs scored -- nothing to bin.")
        return 2
    tms = np.array([r[4] for r in rows])
    print(f"\nscored {len(rows)} pairs; skipped {len(skipped)}")
    for s, why in skipped:
        print(f"  skipped {s}: {why}")
    print(f"\nrefTM  min {tms.min():.3f}  median {np.median(tms):.3f}  max {tms.max():.3f}")
    print("\n0.2-TM bins (counts -- the set is small, so COUNTS lead, not means):")
    edges = np.arange(0.0, 1.01, 0.2)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tms >= lo) & (tms < hi if hi < 1.0 else tms <= 1.0)
        names = [r[0] for r, k in zip(rows, sel) if k]
        print(f"  [{lo:.1f}, {hi:.1f})  n={int(sel.sum()):>3}  {names[:6]}")
    print("\n⭐ Compare this distribution against the TEST set's median date-clean refTM of 0.50.")
    print("   If validation's median is materially higher, the small validation self-vs-nonself gap")
    print("   (~0.10) and the large test gap (~0.32) are the SAME phenomenon at different reference")
    print("   quality -- and refTM, not 'non-self', is the variable that predicts test behaviour.")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("query,reference,len_query,len_ref,refTM\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f}\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
