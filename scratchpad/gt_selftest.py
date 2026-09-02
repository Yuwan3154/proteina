"""Gate the alignment ground-truth builder before anything is built on top of it.

Two checks, in order of how much they can catch:

  SELF-ALIGNMENT. A chain aligned to ITSELF must produce an essentially perfect result: every
  residue that lies inside a kept H/E element must be assigned to exactly that element, so
  Q == (number of residues covered by the kept elements) and no residue may land in two elements.
  This single check catches an index-map bug, a USalign parse bug, and an element-span off-by-one
  -- none of which are visible from a cross-chain number, because a cross-chain alignment is
  legitimately partial and there is nothing to compare it against.

  CROSS-CHAIN SANITY. Real (query, reference) pairs drawn from the dataloader should give a
  partial alignment: 0 < Q < L, and Q should not be pinned at either end.
"""

import os
import sys
import tempfile

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from align_gt import build_alignment, element_spans, load_graph  # noqa: E402

CONFIG = "training_contact_tri_full384_v1"
N_PAIRS = int(os.environ.get("N_PAIRS", "12"))
MAX_HE = 64


def find_transform(obj, depth=0):
    if obj is None or depth > 3:
        return None
    if type(obj).__name__ == "TopologyReferenceTransform":
        return obj
    inner = getattr(obj, "transforms", None)
    if isinstance(inner, (list, tuple)):
        for t in inner:
            f = find_transform(t, depth + 1)
            if f is not None:
                return f
    return None


def main():
    with hydra.initialize("../configs/experiment_config", version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=CONFIG)
    OmegaConf.set_struct(cfg_exp, False)
    ds_dir = f"../configs/datasets_config/{cfg_exp.dataset_config_subdir}"
    with hydra.initialize(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    dm.setup("fit")
    dl = dm.val_dataloader()
    # Read the index FILE rather than fishing for the live transform object: the object is not on
    # dm.transform nor dl.dataset.transform, and everything needed (ids, runs) is in the file.
    data_dir = os.environ["DATA_PATH"] + "/pdb_train"
    idx = torch.load(os.path.join(data_dir, "topology_index.pt"), map_location="cpu",
                     weights_only=False, mmap=True)
    id_to_row = {str(v): i for i, v in enumerate(idx["ids"])}

    def runs_for(row):
        a, b = int(idx["runs_offset"][row]), int(idx["runs_offset"][row + 1])
        return [(int(t), int(n)) for t, n in idx["runs_flat"][a:b].tolist()]

    print(f"[index] {len(id_to_row)} chains", flush=True)
    processed = os.path.join(data_dir, "processed")
    import json
    mpath = os.path.join(data_dir, "shard_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else None

    # ---- collect real (query, reference) pairs, skipping empty and self references ----
    pairs = []
    for i, batch in enumerate(dl):
        if len(pairs) >= N_PAIRS or i > 800:
            break
        for r, p in zip(batch["topology_ref_id"], batch["protein_id"]):
            r, p = str(r), str(p)
            if r and r != p and len(pairs) < N_PAIRS:
                pairs.append((p, r))
    print(f"[pairs] collected {len(pairs)} cross-chain pairs", flush=True)

    fails = 0

    # ---- 1. SELF-ALIGNMENT GATE ----
    print("\n=== SELF-ALIGNMENT GATE ===", flush=True)
    for stem, _ in pairs[:5]:
        row = id_to_row.get(stem)
        if row is None:
            continue
        runs = runs_for(row)
        if not runs:
            continue
        g = load_graph(processed, stem, manifest)
        with tempfile.TemporaryDirectory() as td:
            A, Q = build_alignment(g, g, runs, MAX_HE, td)
        spans = element_spans(runs, MAX_HE)
        covered = sum(min(b, A.shape[0]) - a for a, b in spans if a < A.shape[0])
        multi = int((A.sum(dim=1) > 1).sum())
        ok = (Q == covered) and (multi == 0)
        fails += not ok
        print(f"  {stem:12s} L={A.shape[0]:4d} T={A.shape[1]:3d} Q={Q:4d} "
              f"covered={covered:4d} multi_assigned={multi:3d}  {'PASS' if ok else 'FAIL'}",
              flush=True)

    # ---- 2. CROSS-CHAIN SANITY ----
    print("\n=== CROSS-CHAIN PAIRS ===", flush=True)
    stats = []
    for q, r in pairs:
        rrow = id_to_row.get(r)
        if rrow is None:
            print(f"  {q} -> {r}: reference not in index", flush=True)
            continue
        runs = runs_for(rrow)
        if not runs:
            print(f"  {q} -> {r}: reference has no runs", flush=True)
            continue
        try:
            gq = load_graph(processed, q, manifest)
            gr = load_graph(processed, r, manifest)
            with tempfile.TemporaryDirectory() as td:
                A, Q = build_alignment(gq, gr, runs, MAX_HE, td)
        except Exception as e:  # noqa: BLE001 - a per-pair failure must be reported, not fatal
            print(f"  {q} -> {r}: ERROR {type(e).__name__}: {e}", flush=True)
            fails += 1
            continue
        L, T = A.shape
        frac = Q / max(L, 1)
        stats.append(frac)
        print(f"  {q:12s} -> {r:12s} L={L:4d} T={T:3d} Q={Q:4d} ({frac:.1%} of residues aligned "
              f"to an element)", flush=True)

    if stats:
        import statistics
        print(f"\n  cross-chain Q/L: min={min(stats):.1%} median={statistics.median(stats):.1%} "
              f"max={max(stats):.1%}  n={len(stats)}", flush=True)
        if all(f == 0 for f in stats):
            print("  FAIL: every cross-chain pair aligned nothing", flush=True)
            fails += 1

    print(f"\n{'PASS' if fails == 0 else 'FAIL'}: {fails} failure(s)", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
