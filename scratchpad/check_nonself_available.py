"""Which of the fixed validation chains actually HAVE a non-self topology reference?

⛔ Why this matters. `_build_self_reference_topology` returns None for the WHOLE BATCH when any one
chain lacks a different-sequence cluster-mate, so the batch samples UNCONDITIONED. In the first
non-self sweep point that happened 8 times, and the resulting 0.2922 was the unconditioned floor
being mistaken for non-self performance -- a number that looks entirely plausible next to the
known test-set figure of 0.1798.

The fix is to evaluate the non-self arm on the SUBSET that has a real template, with n stated,
rather than silently averaging conditioned and unconditioned chains. This writes that subset.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_list",
                    default="/orcd/scratch/orcd/011/chenxiou/valset_analysis/val_fixed32_max256.txt")
    ap.add_argument("--index",
                    default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stems = [l.strip() for l in open(args.chain_list) if l.strip()]
    t = TopologyReferenceTransform(index_path=args.index, drop_prob=0.0, seed=args.seed)
    t._ensure_loaded()

    have, missing, absent = [], [], []
    for s in stems:
        row = t._id_to_row.get(s)
        if row is None:
            absent.append(s)
            continue
        got = t.nonself_reference(s, 128, seed=args.seed)
        (have if got is not None else missing).append(s)

    print(f"fixed chain list: {args.chain_list}")
    print(f"  total                     {len(stems)}")
    print(f"  absent from topology index{len(absent):>6}   {absent}")
    print(f"  NO different-seq mate     {len(missing):>6}   {missing}")
    print(f"  usable for the nonself arm{len(have):>6}")
    if have:
        # Show a few (query -> reference) pairs so the arm is visibly conditioning on real templates.
        print("\n  sample (query -> reference):")
        for s in have[:6]:
            got = t.nonself_reference(s, 128, seed=args.seed)
            print(f"    {s} -> {got[1]}")
    if args.out and have:
        with open(args.out, "w") as fh:
            fh.write("\n".join(have) + "\n")
        print(f"\nwrote {len(have)} chains to {args.out}")
    print("\n⚠️ The nonself arm must be run on THIS subset with n stated. Averaging it together with")
    print("   unconditioned chains produces a number that looks like non-self performance and is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
