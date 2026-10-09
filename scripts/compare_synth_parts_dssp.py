"""Compare a rebuilt synthetic-index part with the v4 part of the same index (same chains, same part number).

  --expect identical : the default build at this commit must reproduce v4 byte for byte (the known-good control)
  --expect dssp      : the --proline-donor-mask build; same chains and template rungs, USalign TMs unchanged, and every
                       difference must come with a changed DSSP run list (reported: how many natives / templates moved)

  python scripts/compare_synth_parts_dssp.py NEW_PART.pt V4_PART.pt --expect identical|dssp [--new-skips skips_0000.tsv]

--new-skips: chains the NEW build skipped as `native not_in_pack` (A6000 build reads natives from the training pack) are
excluded from the comparison and counted; any other missing native still fails.
"""

import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("v4")
    ap.add_argument("--expect", choices=("identical", "dssp"), required=True)
    ap.add_argument("--new-skips", default="")
    a = ap.parse_args()
    not_in_pack = set()
    if a.new_skips:
        for line in open(a.new_skips):
            f = line.rstrip("\n").split("\t")
            if len(f) == 3 and f[1] == "native" and f[2] == "not_in_pack":
                not_in_pack.add(f[0])
    x = torch.load(a.new, map_location="cpu", weights_only=False)
    y = torch.load(a.v4, map_location="cpu", weights_only=False)
    cx, cy = {c["stem"]: c for c in x["chains"]}, {c["stem"]: c for c in y["chains"]}
    assert sorted(cx) == sorted(cy), "chain sets differ"
    bad, n_nat, n_nat_runs, n_tpl, n_tpl_runs, n_tpl_align, n_rows_missing = [], 0, 0, 0, 0, 0, 0
    n_excluded = 0
    for s in cy:
        u, v = cx[s], cy[s]
        if s in not_in_pack and u["native"] is None:
            n_excluded += 1
            continue
        if (u["native"] is None) != (v["native"] is None):
            bad.append(f"{s}: native presence differs")
            continue
        if u["seq_hash"] != v["seq_hash"]:
            bad.append(f"{s}: seq_hash differs")
        if v["native"] is not None:
            n_nat += 1
            same_runs = u["native"][0] == v["native"][0]
            n_nat_runs += not same_runs
            if same_runs and repr(u["native"][:5]) != repr(v["native"][:5]):
                bad.append(f"{s}: native row differs with identical runs")
        ru = {r[3]: r for r in u["rows"]}
        for r in v["rows"]:
            n_tpl += 1
            q = ru.get(r[3])
            if q is None:
                n_rows_missing += 1
                continue
            if repr((q[1], q[2], q[5])) != repr((r[1], r[2], r[5])):  # repr: a NaN USalign TM equals itself
                bad.append(f"{s} rung {r[3]}: tm/rewind/usalign tm differ")
            same_runs = q[0][0] == r[0][0]
            n_tpl_runs += not same_runs
            n_tpl_align += q[4] != r[4]
            if same_runs and (repr(q[0]) != repr(r[0]) or q[4] != r[4]):
                bad.append(f"{s} rung {r[3]}: row differs with identical runs")
        if len(u["rows"]) != len(v["rows"]):
            bad.append(f"{s}: {len(u['rows'])} vs {len(v['rows'])} template rows")
    rec = {"expect": a.expect, "new_dssp_def": x.get("dssp_def"), "chains": len(cy), "excluded_not_in_pack": n_excluded,
           "natives": n_nat,
           "natives_runs_changed": n_nat_runs, "template_rows": n_tpl, "template_rows_missing": n_rows_missing,
           "template_runs_changed": n_tpl_runs, "template_align_changed": n_tpl_align, "n_bad": len(bad), "bad": bad[:20]}
    print(json.dumps(rec, indent=1))
    if a.expect == "identical":
        assert not bad and n_nat_runs == 0 and n_tpl_runs == 0 and n_tpl_align == 0 and n_rows_missing == 0, "not identical"
    else:
        assert "donor_mask" in str(x.get("dssp_def")), "new part was not built with --proline-donor-mask"
        assert not bad and n_rows_missing == 0 and (n_nat_runs + n_tpl_runs) > 0, "unexpected differences"


if __name__ == "__main__":
    main()
