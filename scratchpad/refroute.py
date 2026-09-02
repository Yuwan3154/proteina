"""Why is the self-reference rate ~20% instead of the recorded 6.66%? Answer it exactly.

Sampling gave the RATE; this gives the MECHANISM, and over the whole population rather than a
sample. `_pick_template` and `forward` between them have exactly two routes to a self-reference:

  (a) no different-sequence cluster-mate exists  -> `_pick_template` returns `row`
      (either a singleton cluster, or every mate shares the query's seq_hash)
  (b) a mate existed and was picked, but it has NO usable DSSP runs, so `forward` does
      `if not self._runs_for(t_row): t_row = row`

Route (b) is the one `measure_template_pool.py` never counted, and it is not observable from the
pool statistics at all. Because the picker draws UNIFORMLY from the candidate set, the exact
probability of route (b) for a given query is (# candidates with no runs) / (# candidates), so the
population rate can be computed in closed form instead of sampled.

Everything is read straight from topology_index.pt -- no dataloader, no model, no transform object.
"""

import os
import sys

import torch

DATA = os.environ["DATA_PATH"] + "/pdb_train"
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = all chains


def main():
    idx = torch.load(os.path.join(DATA, "topology_index.pt"), map_location="cpu",
                     weights_only=False, mmap=True)
    runs_offset = idx["runs_offset"]
    cluster_of = idx["cluster_of"]
    members_offset = idx["members_offset"]
    members_flat = idx["members_flat"]
    seq_hash = idx["seq_hash"]
    n = len(idx["ids"])
    print(f"[index] {n} chains", flush=True)

    # A row has "no usable runs" iff its runs slice is empty -- the same test forward() applies.
    has_runs = (runs_offset[1:] - runs_offset[:-1]) > 0
    print(f"[runs] rows with NO runs: {int((~has_runs).sum())} "
          f"({float((~has_runs).float().mean()):.3%} of all chains)", flush=True)

    rows = range(n if LIMIT == 0 else min(n, LIMIT))
    p_a = 0.0   # expected fraction taking route (a)
    p_b = 0.0   # expected fraction taking route (b)
    n_singleton = n_allsame = 0
    counted = 0
    for row in rows:
        counted += 1
        cl = int(cluster_of[row])
        lo, hi = int(members_offset[cl]), int(members_offset[cl + 1])
        members = members_flat[lo:hi]
        if members.numel() <= 1:
            p_a += 1.0
            n_singleton += 1
            continue
        own = seq_hash[row]
        cand = members[seq_hash[members.long()] != own]
        if cand.numel() == 0:
            p_a += 1.0
            n_allsame += 1
            continue
        # Uniform pick, so P(route b) is exactly the candidate fraction lacking runs.
        bad = int((~has_runs[cand.long()]).sum())
        p_b += bad / cand.numel()
        if counted % 50000 == 0:
            print(f"  ...{counted}", flush=True)

    a = p_a / counted
    b = p_b / counted
    print(f"\npopulation over {counted} chains (exact, not sampled):")
    print(f"  route (a) no different-seq mate     : {a:.3%}"
          f"   [singleton {n_singleton}, all-same-hash {n_allsame}]")
    print(f"  route (b) picked mate had NO runs   : {b:.3%}   <- never counted before")
    print(f"  TOTAL expected self-reference rate  : {a + b:.3%}")
    print(f"\n  recorded expectation (measure_template_pool.py): 6.66% train")
    print(f"  measured through the dataloader (n=2985 train)  : 20.5%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
