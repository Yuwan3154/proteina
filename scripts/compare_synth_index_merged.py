"""Merged v5 (proline-mask, A6000) synthetic index vs v4: every chain-level difference must be explained.

Allowed: a v4 chain missing from v5 because its native was skipped as `not_in_pack` (A6000 natives come from the training
pack) or `dssp_all_ignore` (CA-only chains; v4 used labels computed on fill coordinates) -- read from the v5 skip logs.
Required for every other chain: same native presence, same template-row count and the same row TMs; the v5 eligible list
must be the v4 list minus explained chains.

  python scripts/compare_synth_index_merged.py V5.pt V5_ELIGIBLE V5_PARTS_DIR V4.pt V4_ELIGIBLE
"""

import collections
import glob
import json
import sys

import torch


def groups(path):
    idx = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    ids, nat = idx["ids"], idx["row_is_native"].tolist()
    mo, mf, tm = idx["members_offset"].tolist(), idx["members_flat"].tolist(), idx["row_tm"].float().tolist()
    natives = [i for i, n in enumerate(nat) if n]
    assert len(natives) == len(mo) - 1, (len(natives), len(mo))
    out = {str(ids[r]): sorted(round(tm[m], 3) for m in mf[mo[g]:mo[g + 1]]) for g, r in enumerate(natives)}
    return out, idx.get("dssp_def")


def main():
    v5p, v5e, parts, v4p, v4e = sys.argv[1:6]
    v5, d5 = groups(v5p)
    v4, d4 = groups(v4p)
    skip = {}
    for f in glob.glob(f"{parts}/skips_*.tsv"):
        for line in open(f):
            s = line.rstrip("\n").split("\t")
            if len(s) == 3 and s[1] == "native":
                skip[s[0]] = s[2]
    gone = set(v4) - set(v5)
    why = collections.Counter(skip.get(s, "UNEXPLAINED") for s in gone)
    new = set(v5) - set(v4)
    rows_diff = [s for s in set(v4) & set(v5) if v4[s] != v5[s]]
    e4, e5 = set(open(v4e).read().split()), set(open(v5e).read().split())
    rec = {"v5_dssp_def": d5, "v4_dssp_def": d4, "v4_chains": len(v4), "v5_chains": len(v5), "gone_by_reason": dict(why),
           "new_in_v5": len(new), "common_with_different_template_rows": len(rows_diff), "examples": sorted(rows_diff)[:10],
           "v4_eligible": len(e4), "v5_eligible": len(e5), "eligible_lost": len(e4 - e5),
           "eligible_lost_unexplained": len((e4 - e5) - gone), "eligible_new": len(e5 - e4)}
    print(json.dumps(rec, indent=1))
    assert "donor_mask" in str(d5)
    assert why.get("UNEXPLAINED", 0) == 0 and not new and not rows_diff
    assert set(why) <= {"not_in_pack", "dssp_all_ignore"} and rec["eligible_lost_unexplained"] == 0 and not (e5 - e4)


if __name__ == "__main__":
    main()
