"""T8 max512 split by INHERITANCE from the max384 split (user 2026-10-09: "1. Inherit", "2. Train + val + test").

Rule, over every chain of the max384 dump and of the maxl512 selection:
  * a chain in the max384 split dump keeps its split (also when the maxl512 exclusion would have removed it);
  * a rosetta decoy keeps the split the repo's PDBDataSplitter gives it (val/test), never an inherited one;
  * a chain that was in the max384 selection but NOT in the dump (removed there by the exclusion or drop list) stays out;
  * any other new chain takes the split of its maxl512 25% cluster's max384 members; if those disagree it is DROPPED;
    a cluster with no max384 member keeps the PDBDataSplitter split at maxl512 (seeded 0.98/0.019/0.001, seed 42);
  * a new chain the maxl512 exclusion removed stays out.
Both splits are computed with PDBDataSplitter.split_data (same exclusion + drop files as the training configs).
CONTROL: the max384 split recomputed here must equal the dump exactly, or nothing is written.
Usage: python t8_make_split.py DATA_DIR SPLIT384_DIR UNION_LIST OUT_DIR
"""

import collections
import os
import sys

import numpy as np
import pandas as pd

from proteinfoundation.datasets.pdb_data import PDBDataSplitter
from proteinfoundation.utils.cluster_utils import read_cluster_tsv, setup_clustering_file_paths

SPLITS = ("train", "val", "test")
IDENT = "df_pdb_f1_minl50_maxl{}_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse"
SIM = 0.25
DATE_CUTOFF = "2019-08-28"  # max_deposition_date of every pdb_train_*_cutoff-190828 config
DECOYS, DROPS = "rosetta_decoys.txt", "unusable_chains.txt"


def read_ids(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("".join(f"{c}\n" for c in ids))


def code_split(data_dir, ident):
    sp = PDBDataSplitter(
        data_dir=data_dir, train_val_test=[0.98, 0.019, 0.001], split_type="sequence_similarity",
        split_sequence_similarity=SIM, overwrite_sequence_clusters=False, exclude_ids=[],
        exclude_ids_from_file=os.path.join(data_dir, DECOYS), drop_ids_from_file=os.path.join(data_dir, DROPS),
    )
    _, _, tsv = setup_clustering_file_paths(data_dir, ident, SIM)
    assert tsv.exists(), f"cluster tsv missing (would trigger a fresh mmseqs run): {tsv}"
    dfs, _ = sp.split_data(pd.read_csv(os.path.join(data_dir, f"{ident}.csv")), ident)
    out = {s: set(dfs[s]["id"]) for s in SPLITS}
    assert not (out["train"] & out["val"]) and not (out["train"] & out["test"]) and not (out["val"] & out["test"])
    return out, read_cluster_tsv(tsv)


def to_map(splits):
    return {c: s for s in SPLITS for c in splits[s]}


def assign(s384, s512, clusters, rep_of, decoys, sel384):
    """The rule (module docstring). Returns (final {id: split}, {reason: [ids]} for every chain left out, reason counts)."""
    final, out, why = {}, collections.defaultdict(list), collections.Counter()
    for c in sorted(set(s384) | set(s512)):
        if c in s384:
            final[c] = s384[c]
            why["kept max384 split" + ("" if c in s512 else " (removed by the maxl512 exclusion)")] += 1
        elif c not in s512:
            raise AssertionError(f"{c} is in neither split")
        elif c in decoys:
            final[c] = s512[c]
            why["new decoy, PDBDataSplitter split"] += 1
        elif c in sel384:
            out["excluded_at_384"].append(c)
            why["OUT: in the max384 selection but not its split (excluded/dropped there)"] += 1
        else:
            assert c in rep_of, f"{c} has no maxl512 cluster"
            inh = {s384[m] for m in clusters[rep_of[c]] if m in s384}
            if len(inh) == 1:
                final[c] = inh.pop()
                why["new, inherited from cluster"] += 1
            elif not inh:
                final[c] = s512[c]
                why["new, cluster without max384 member (PDBDataSplitter split)"] += 1
            else:
                out["disagree"].append(c)
                why["OUT: new, cluster's max384 members disagree"] += 1
    return final, out, why


def main(data_dir, split384_dir, union_list, out_dir):
    # the deposition cutoff is NOT in file_identifier, so the csv on disk could come from another config: check it
    sel = {}
    for n in (384, 512):
        df = pd.read_csv(os.path.join(data_dir, f"{IDENT.format(n)}.csv"), usecols=["id", "length", "deposition_date"])
        print(f"[csv maxl{n}] {len(df)} rows, length {df.length.min()}-{df.length.max()}, "
              f"deposition {df.deposition_date.min()} .. {df.deposition_date.max()}")
        assert len(df) and df.deposition_date.max() <= DATE_CUTOFF and df.length.max() <= n and df.length.min() >= 50
        sel[n] = df
    dump = {s: set(read_ids(os.path.join(split384_dir, f"{s}_chain_ids.txt"))) for s in SPLITS}
    print("[dump max384]", {s: len(v) for s, v in dump.items()})
    assert all(dump.values()), "a dump split is empty"

    re384, _ = code_split(data_dir, IDENT.format(384))
    print("[recomputed max384]", {s: len(v) for s, v in re384.items()})
    for s in SPLITS:
        d1, d2 = len(re384[s] - dump[s]), len(dump[s] - re384[s])
        print(f"  CONTROL {s}: recomputed-not-dump {d1}, dump-not-recomputed {d2}")
        assert d1 == 0 and d2 == 0, "CONTROL FAILED: the max384 split does not reproduce; nothing written"

    code512, clusters = code_split(data_dir, IDENT.format(512))
    print("[code max512]", {s: len(v) for s, v in code512.items()})
    s384, s512 = to_map(dump), to_map(code512)
    rep_of = {m: rep for rep, ms in clusters.items() for m in ms}
    decoys = set(read_ids(os.path.join(data_dir, DECOYS)))
    sel384 = set(sel[384].id)
    final, out, why = assign(s384, s512, clusters, rep_of, decoys, sel384)
    excluded512 = sorted(set(sel[512].id) - set(final) - set(s384) - {c for v in out.values() for c in v})
    print("[assignment]", dict(why))
    print(f"[new chains removed by the maxl512 exclusion/drop list] {len(excluded512)}")

    # gates (each prints its sample size)
    n_lost = sum(1 for c in s384 if c not in final or final[c] != s384[c])
    print(f"GATE max384 chains lost or moved: {n_lost} of {len(s384)}")
    n_decoy_train = sum(1 for c in decoys if final.get(c) == "train")
    print(f"GATE decoys in T8 train: {n_decoy_train} of {len(decoys)} ({sum(1 for c in decoys if c in final)} placed)")
    union = read_ids(union_list)
    n_union_held = sum(1 for c in union if final.get(c) in ("val", "test"))
    print(f"GATE 195-union in T8 val/test: {n_union_held} of {len(union)}")
    mixed_new = []
    for rep, ms in clusters.items():
        sp = {final[m] for m in ms if m in final}
        if "train" in sp and len(sp) > 1:
            mixed_new += [m for m in ms if m in final and m not in s384 and m not in decoys]
    print(f"GATE new non-decoy chains in maxl512 clusters that mix train with val/test: {len(mixed_new)}")
    assert n_lost == 0 and n_decoy_train == 0 and len(union) > 0 and n_union_held == len(union) and not mixed_new

    fin = {s: sorted(c for c, v in final.items() if v == s) for s in SPLITS}
    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        write_ids(os.path.join(out_dir, f"{s}_chain_ids.txt"), fin[s])
    for k in ("disagree", "excluded_at_384"):
        write_ids(os.path.join(out_dir, f"out_{k}.txt"), sorted(out[k]))
    write_ids(os.path.join(out_dir, "out_excluded_at_512.txt"), excluded512)
    allc = sorted(final)
    write_ids(os.path.join(out_dir, "chains_for_templates.txt"), allc)
    with open(os.path.join(out_dir, "clusters_maxl512.tsv"), "w") as f:
        for c in allc:
            f.write(f"{rep_of.get(c, '-')}\t{c}\t{final[c]}\n")

    L = sel[512].drop_duplicates("id").set_index("id")["length"]
    edges = [50, 128, 256, 384, 448, 512]
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        def p(*a):
            print(*a)
            print(*a, file=f)
        p("T8 max512 split (inherit from max384)")
        p("  assignment:", dict(why))
        for s in SPLITS:
            ids = fin[s]
            new = [c for c in ids if c not in s384]
            ncl = len({rep_of.get(c, c) for c in ids})
            h = np.histogram(L.reindex(ids).dropna().values, bins=edges)[0].tolist()
            p(f"  {s}: {len(ids)} chains ({len(new)} new vs max384), {ncl} maxl512 clusters, length bins {edges}: {h}")
        p(f"  out: disagree {len(out['disagree'])}, excluded at 384 {len(out['excluded_at_384'])}, "
          f"excluded at 512 {len(excluded512)}")
        p(f"  chains_for_templates.txt: {len(allc)} (train + val + test)")


if __name__ == "__main__":
    assert len(sys.argv) == 5, __doc__
    main(*sys.argv[1:5])
