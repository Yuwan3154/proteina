"""T8 template list: ONE chain per unique sequence, the highest-resolution one (user 2026-10-10: "1 but use the highest
resolution one"). Ties at the best resolution (common: chains of one entry) go first to a chain already in the old T2
template pool (user 2026-10-10), then to the alphabetically first chain id. Pool membership is read from the pool
manifest's `label_id` column: split ids are proteina LABEL chain ids, T2 files are keyed by AUTH ids (they differ for
~3.8k pool rows), so the manifest's `auth_id` column must never be used here.
Deduplication is WITHIN each split, so a sequence whose chains span splits (inherited mixed clusters) gets one
representative per split and a train template never comes from a val/test chain.
Chains listed in any EXCLUDE file (first column, header skipped: the coverage check's ca_only / no_confind / missing_pt)
are removed from every split first (user: Ca-only "Drop them from all splits"). Stdlib only.
Usage: python t8_template_reps.py SELECTION_CSV SPLIT_DIR POOL_MANIFEST.tsv OUT_DIR EXCLUDE.tsv [EXCLUDE.tsv ...]
"""

import collections
import csv
import os
import sys

SPLITS = ("train", "val", "test")


def read_pool(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    col = rows[0].index("label_id")
    ids = [r[col] for r in rows[1:]]
    assert ids and len(set(ids)) == len(ids), f"{path}: empty or duplicated label_id"
    return set(ids)


def main(sel_csv, split_dir, pool_manifest, out_dir, excludes):
    csv.field_size_limit(10**9)
    with open(os.path.join(split_dir, "chains_for_templates.txt")) as f:
        n_all = sum(1 for l in f if l.strip())
    split = {}
    for s in SPLITS:
        with open(os.path.join(split_dir, f"{s}_chain_ids.txt")) as f:
            for c in f:
                if c.strip():
                    assert c.strip() not in split
                    split[c.strip()] = s
    assert n_all > 0 and len(split) == n_all, f"split files hold {len(split)} chains, chains_for_templates {n_all}"
    drop, n_rows = {}, {}
    for e in excludes:
        with open(e) as f:
            rows = list(csv.reader(f, delimiter="\t"))
        assert rows and rows[0][0] == "stem", f"{e}: not a coverage exclude table"
        n_rows[os.path.basename(e)] = len(rows) - 1
        for r in rows[1:]:
            drop.setdefault(r[0], os.path.basename(e))
    assert set(drop) <= set(split), f"{len(set(drop) - set(split))} excluded ids are not in the split"
    seq, res = {}, {}
    with open(sel_csv) as f:
        for r in csv.DictReader(f):
            if r["id"] in split:
                assert r["id"] not in seq, f"duplicate csv id {r['id']}"
                x = float(r["resolution"])
                # -1.0 = not recorded (EM entries): user 2026-10-10 "Treat them as worst resolution"
                assert x == -1.0 or 0.0 <= x <= 5.0, f"{r['id']}: resolution {r['resolution']} outside the selection's 0-5 A"
                seq[r["id"]], res[r["id"]] = r["sequence"], (float("inf") if x == -1.0 else x)
    assert set(seq) == set(split), f"{len(set(split) - set(seq))} split chains missing from the csv"

    keep = {c: s for c, s in split.items() if c not in drop}
    groups = collections.defaultdict(list)
    for c in keep:
        groups[(keep[c], seq[c])].append(c)
    pool = read_pool(pool_manifest)
    assert pool & set(keep), "no kept chain is in the pool manifest's label_id column (wrong id convention?)"

    def pick(v):
        return min(v, key=lambda c: (res[c], c not in pool, c))

    reps = sorted(pick(v) for v in groups.values())
    n_pool_pick = sum(1 for v in groups.values() if pick(v) != min(v, key=lambda c: (res[c], c)))
    best2 = [sorted(res[c] for c in v)[:2] for v in groups.values()]
    n_tie = sum(1 for r in best2 if len(r) == 2 and r[0] == r[1])
    span = sum(1 for n in collections.Counter(q for _, q in groups).values() if n > 1)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "template_chains.txt"), "w") as f:
        f.write("".join(f"{c}\n" for c in reps))
    for s in SPLITS:
        with open(os.path.join(out_dir, f"{s}_chain_ids.txt"), "w") as f:
            f.write("".join(f"{c}\n" for c in sorted(c for c in keep if keep[c] == s)))
    with open(os.path.join(out_dir, "dropped.tsv"), "w") as f:
        f.write("stem\tsplit\treason\n" + "".join(f"{c}\t{split[c]}\t{drop[c]}\n" for c in sorted(drop)))
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        lines = [f"split chains {len(split)}; exclude rows read {n_rows}; dropped {len(drop)} "
                 f"({dict(collections.Counter(drop.values()))}); kept {len(keep)}",
                 f"resolution not recorded (-1.0, sorted last): {sum(1 for c in keep if res[c] == float('inf'))} kept chains, "
                 f"{sum(1 for c in reps if res[c] == float('inf'))} picked",
                 f"template chains (one per unique sequence per split): {len(reps)}; of them in the old T2 pool "
                 f"(label_id) {sum(1 for c in reps if c in pool)}; kept chains in the pool {sum(1 for c in keep if c in pool)}",
                 f"  groups with a tie at the best resolution: {n_tie} (old-pool chain preferred, then alphabetical; "
                 f"the pool preference changed {n_pool_pick} picks vs alphabetical); "
                 f"sequences whose chains span >1 split: {span}"]
        for s in SPLITS:
            lines.append(f"  {s}: kept chains {sum(1 for c in keep.values() if c == s)}, "
                         f"template chains {sum(1 for c in reps if keep[c] == s)}")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert len(sys.argv) >= 6, __doc__
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
