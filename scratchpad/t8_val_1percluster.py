"""Val chain list for the T8 alignment-bias study (user 2026-10-09: "run the current ConFind tri model on the validation set";
answer: the val split, one chain per cluster). Max384 val split, max384 25% clusters; a chain qualifies when it has a
template in the training TM range in the ConFind synthetic index (the nonself arm needs one). Per cluster: the
representative if it qualifies, else the alphabetically first qualifying chain (the rule of stageA_val_1percluster.txt).
Usage: python t8_val_1percluster.py VAL_IDS CLUSTER_TSV ELIGIBLE_IDS OUT [N_CHUNKS]
"""

import collections
import os
import sys


def main(val_ids, cluster_tsv, eligible, out, n_chunks="1"):
    val = {l.strip() for l in open(val_ids) if l.strip()}
    elig = {l.strip() for l in open(eligible) if l.strip()}
    members = collections.defaultdict(list)
    for l in open(cluster_tsv):
        rep, m = l.rstrip("\n").split("\t")
        if m in val:
            members[rep].append(m)
    covered = {m for ms in members.values() for m in ms}
    assert covered == val, f"{len(val - covered)} val chains in no cluster"
    pick = []
    for rep, ms in sorted(members.items()):
        ok = sorted(m for m in ms if m in elig)
        if ok:
            pick.append(rep if rep in ok else ok[0])
    print(f"val chains {len(val)}, val clusters {len(members)}, clusters with a qualifying chain {len(pick)}")
    with open(out, "w") as f:
        f.write("".join(f"{c}\n" for c in pick))
    k = int(n_chunks)
    for i in range(k):
        with open(f"{os.path.splitext(out)[0]}_chunk{i:02d}.txt", "w") as f:
            f.write("".join(f"{c}\n" for c in pick[i::k]))
    print(f"-> {out} (+ {k} chunks)")


if __name__ == "__main__":
    assert len(sys.argv) in (5, 6), __doc__
    main(*sys.argv[1:])
