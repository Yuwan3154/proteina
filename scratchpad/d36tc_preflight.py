"""D36tc pre-flight (run BEFORE tri sampling): no sample may fall back to unconditioned silently.

_build_self_reference_topology records the ref id before its `ref is None` check, so an absent index row would
still show the intended ref_id in samples.jsonl. This gate makes that state impossible up front:
  - every fixed stem's .pt carries a batch protein_id (PDBDataset rule: protein_id, else id, else file stem)
    spelled EXACTLY as the stem;
  - every such protein_id is an index id (self arm) and, with --map, a map key whose value is an index id.
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True, help="fixed_chain_list (first token per line)")
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--map", default=None, help="topology_reference_map JSON (template arm)")
    args = ap.parse_args()

    stems = sorted({ln.split()[0] for ln in open(args.chains) if ln.strip()})
    ids = set(str(s) for s in torch.load(args.index, map_location="cpu", weights_only=False, mmap=True)["ids"])
    bad = []
    for st in stems:
        p = os.path.join(args.processed_dir, f"{st}.pt")
        if not os.path.exists(p):
            bad.append(f"{st}: no {p}")
            continue
        g = torch.load(p, map_location="cpu", weights_only=False)
        pid = getattr(g, "protein_id", None)
        if pid is None:
            pid = getattr(g, "id", None)
        pid = st if pid is None else str(pid)
        if pid != st:
            bad.append(f"{st}: batch protein_id would be {pid!r}")
    if args.map is None:
        bad += [f"{st}: not an index id" for st in stems if st not in ids]
    else:
        m = json.load(open(args.map))
        bad += [f"{st}: no map entry" for st in stems if st not in m]
        bad += [f"{st}: map value {m[st]!r} not an index id" for st in stems if st in m and m[st] not in ids]
    print(f"[preflight] {len(stems)} stems, index {len(ids)} ids, map={args.map}: {len(bad)} problems")
    for b in bad:
        print(f"  FAIL {b}")
    assert not bad, "pre-flight failed"
    print("D36TC_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
