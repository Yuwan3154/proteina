"""Closed-form says 1.9% self-reference; the dataloader measures 20.5%. Which is wrong?

Both cannot be right, and the answer decides whether the training finding stands. This calls the
transform's OWN `_pick_template` on real rows, many draws each, and compares three things on the
same rows:

  observed   -- what `_pick_template` actually returns
  closed     -- my analytic prediction for that row (0 if a different-seq candidate exists,
                else 1), i.e. the calculation that produced 1.9%
  forward    -- what `forward()` would end up with, including the `not _runs_for(t_row)` revert

If observed matches closed, the dataloader measurement is measuring something else and the 20.5%
needs re-explaining. If observed is ~20%, my closed form is wrong and I need to find out how.
"""

import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform  # noqa: E402

DATA = os.environ["DATA_PATH"] + "/pdb_train"
N_ROWS = int(os.environ.get("N_ROWS", "3000"))
DRAWS = int(os.environ.get("DRAWS", "5"))


def main():
    tr = TopologyReferenceTransform(
        index_path=os.path.join(DATA, "topology_index.pt"),
        max_topology_len=128,
        max_topology_he_len=64,
        sigma_frac=0.15,
        mutate_prob=0.3,
        drop_prob=0.25,
        self_fallback=True,
        min_len=1,
    )
    tr._ensure_loaded()
    idx = tr._index
    n = len(idx["ids"])
    print(f"[index] {n} chains", flush=True)

    runs_offset = idx["runs_offset"]
    has_runs = (runs_offset[1:] - runs_offset[:-1]) > 0

    g = torch.Generator().manual_seed(0)
    rows = torch.randint(n, (N_ROWS,), generator=g).tolist()

    obs_self = obs_total = 0
    closed_self = 0
    fwd_self = 0
    why = Counter()
    for row in rows:
        # closed-form prediction for this row
        cl = int(idx["cluster_of"][row])
        lo, hi = int(idx["members_offset"][cl]), int(idx["members_offset"][cl + 1])
        members = idx["members_flat"][lo:hi]
        if members.numel() <= 1:
            closed_pred = 1
            why["singleton"] += 1
        else:
            own = idx["seq_hash"][row]
            cand = members[idx["seq_hash"][members.long()] != own]
            closed_pred = 1 if cand.numel() == 0 else 0
            if cand.numel() == 0:
                why["all-same-hash"] += 1
        closed_self += closed_pred

        for _ in range(DRAWS):
            t_row = tr._pick_template(row)
            obs_total += 1
            if t_row == row:
                obs_self += 1
            t_eff = t_row if bool(has_runs[t_row]) else row
            if t_eff == row:
                fwd_self += 1

    print(f"\nrows={N_ROWS} draws/row={DRAWS} total_draws={obs_total}")
    print(f"  closed-form  self-rate : {closed_self / N_ROWS:.3%}   {dict(why)}")
    print(f"  OBSERVED _pick_template: {obs_self / obs_total:.3%}")
    print(f"  after forward() revert : {fwd_self / obs_total:.3%}")
    print(f"\n  dataloader measured    : 20.5% (train, n=2985)")
    verdict = ("closed form AGREES with the picker -- the 20.5% comes from somewhere else"
               if abs(obs_self / obs_total - closed_self / N_ROWS) < 0.02
               else "closed form DISAGREES with the picker -- the analytic calc is wrong")
    print(f"  => {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
