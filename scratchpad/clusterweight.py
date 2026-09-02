"""Reconcile 1.9% (per chain) with ~20% (per sample). Hypothesis: cluster-random sampling.

Facts to reconcile:
  - Over all 306,749 CHAINS, only 1.9% have no different-sequence cluster-mate.
  - Through the dataloader, 19-20% of sampled chains have none -- and attribution says 100% of the
    observed self-references are exactly that (expected) route, so the picker is not at fault.

`sampling_mode: cluster-random` draws ONE chain PER CLUSTER per epoch. A chain in a 500-member
cluster is therefore seen 1/500 as often as a chain that is alone in its cluster -- and a lone
chain is precisely a chain with no different-sequence mate. So the per-SAMPLE rate should be the
per-CLUSTER rate, not the per-chain rate.

If the per-cluster rate lands near 20%, the discrepancy is fully explained and the ~20% figure is
the training-relevant one (it is what the sampler actually feeds the model).
"""

import os
import sys

import torch

DATA = os.environ["DATA_PATH"] + "/pdb_train"


def main():
    idx = torch.load(os.path.join(DATA, "topology_index.pt"), map_location="cpu",
                     weights_only=False, mmap=True)
    members_offset = idx["members_offset"]
    members_flat = idx["members_flat"]
    seq_hash = idx["seq_hash"]
    n_chains = len(idx["ids"])
    n_clusters = len(members_offset) - 1
    print(f"[index] {n_chains} chains in {n_clusters} clusters "
          f"(mean {n_chains / max(n_clusters,1):.1f} chains/cluster)", flush=True)

    # A cluster is "self-only" if EVERY member would find no different-sequence mate, which for a
    # cluster means all its members share one sequence hash (a singleton trivially qualifies).
    self_only_clusters = 0
    singleton = 0
    chains_in_self_only = 0
    for cl in range(n_clusters):
        lo, hi = int(members_offset[cl]), int(members_offset[cl + 1])
        m = members_flat[lo:hi]
        if m.numel() == 0:
            continue
        if m.numel() == 1:
            singleton += 1
            self_only_clusters += 1
            chains_in_self_only += 1
            continue
        h = seq_hash[m.long()]
        if int((h != h[0]).sum()) == 0:
            self_only_clusters += 1
            chains_in_self_only += int(m.numel())
        if cl % 50000 == 0 and cl:
            print(f"  ...{cl}", flush=True)

    per_cluster = self_only_clusters / max(n_clusters, 1)
    per_chain = chains_in_self_only / max(n_chains, 1)
    print(f"\n  clusters where NO member has a different-seq mate: {self_only_clusters}"
          f"  (singletons {singleton})")
    print(f"  PER-CLUSTER rate (= per-SAMPLE under cluster-random): {per_cluster:.2%}")
    print(f"  PER-CHAIN   rate                                    : {per_chain:.2%}")
    print(f"\n  measured through the dataloader: 19.3% (val) / 20.5% (train)")
    ok = abs(per_cluster - 0.20) < 0.07
    print(f"  => cluster-weighting {'EXPLAINS' if ok else 'does NOT explain'} the discrepancy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
