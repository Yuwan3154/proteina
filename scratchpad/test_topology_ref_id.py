"""Gate for topology_ref_id recording and the require_nonself validation arm.

Two things that would silently ruin the stratification:

 1. ⛔ `topology_ref_id` must be set on EVERY exit path of forward(). dense_padded_collate
    INTERSECTS keys across the samples of a batch, so a key present on only some samples is
    dropped from the batch entirely -- the stratification would come back empty, with no error
    anywhere. This walks every branch and asserts the attribute exists.
 2. require_nonself must NEVER return the query's own topology. If it leaks even occasionally the
    arm stops measuring template threading and starts measuring self-copying, which is the exact
    confound the arm exists to remove.
"""

import os
import sys

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.topology_reference import (
    MASK_REF_ID,
    TopologyReferenceTransform,
)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def main():
    idx_path = os.environ.get("TOPO_INDEX",
                              "/orcd/pool/006/chenxiou/proteina/data/pdb_train/topology_index.pt")
    if not os.path.exists(idx_path):
        print(f"topology index not found at {idx_path}; set TOPO_INDEX. Skipping.")
        return 2

    def make(**kw):
        kw.setdefault("drop_prob", 0.0)          # caller may override; do not pass it twice
        t = TopologyReferenceTransform(index_path=idx_path, seed=0, **kw)
        t._ensure_loaded()
        return t

    t = make()
    ids = list(t._index["ids"])
    print(f"  index: {len(ids)} chains")

    # Pick chains that DO have a non-self mate and chains that do NOT, so both branches are walked.
    with_mate, without_mate = [], []
    for stem in ids[:4000]:
        row = t._id_to_row[stem]
        (with_mate if t._pick_template(row) != row else without_mate).append(stem)
        if len(with_mate) >= 40 and len(without_mate) >= 10:
            break
    print(f"  sampled {len(with_mate)} with a non-self mate, {len(without_mate)} without")
    check("found chains of BOTH kinds (both branches are reachable)",
          len(with_mate) > 0 and len(without_mate) > 0)

    def run(tr, stem, L=64):
        g = Data()
        g.coords = torch.zeros(L, 3)
        g.protein_id = stem
        return tr(g)

    # ---- 1. the attribute exists on every branch ----
    missing = []
    for stem in (with_mate[:20] + without_mate[:10]):
        if not hasattr(run(t, stem), "topology_ref_id"):
            missing.append(stem)
    check("topology_ref_id set on the retrieval and self-fallback branches", not missing,
          str(missing[:3]))

    t_drop = make(drop_prob=1.0)
    g = run(t_drop, with_mate[0])
    check("topology_ref_id set on the DROP branch", hasattr(g, "topology_ref_id"))
    check("...and equals the MASK sentinel", getattr(g, "topology_ref_id", None) == MASK_REF_ID,
          str(getattr(g, "topology_ref_id", None)))

    g = run(t, "chain_that_does_not_exist_xyz")
    check("topology_ref_id set when the chain is absent from the index",
          hasattr(g, "topology_ref_id") and g.topology_ref_id == MASK_REF_ID)

    # ---- 2. self_fallback=True really does hand back the query itself ----
    if without_mate:
        g = run(t, without_mate[0])
        check("self_fallback returns the query's OWN id (this is the ceiling case)",
              g.topology_ref_id == without_mate[0],
              f"{without_mate[0]} -> {g.topology_ref_id}")

    # ---- 3. require_nonself never leaks a self reference ----
    tn = make(require_nonself=True)
    leaks, masked, real = [], 0, 0
    for stem in (with_mate[:40] + without_mate[:10]):
        r = run(tn, stem).topology_ref_id
        if r == MASK_REF_ID:
            masked += 1
        elif r == stem:
            leaks.append(stem)
        else:
            real += 1
    check("require_nonself NEVER returns the query's own topology", not leaks, str(leaks[:3]))
    check("require_nonself still yields real cross-chain references", real > 0, f"n={real}")
    check("require_nonself masks the no-mate chains instead", masked >= len(without_mate[:10]),
          f"masked={masked}")

    # ---- 4. repeated draws stay non-self (the pick is random per call) ----
    reps = [run(tn, with_mate[0]).topology_ref_id for _ in range(25)]
    check("25 repeated draws are all non-self",
          all(r != with_mate[0] for r in reps),
          f"{len(set(reps))} distinct references drawn")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
