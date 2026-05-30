#!/usr/bin/env python3
"""Verify CATBalancedSampler: toy self-test + optional real-data integration test.

--selftest: synthetic clusters + chain_to_cat mixing AFDB-style (case-sensitive,
e.g. AF-...-model_v6) and PDB-style (lowercased) keys. Validates the
case-sensitive chain_to_cat lookup (exact-case first, then .lower() fallback)
plus topology indexing, pure-cluster infill, 'largest' cluster choice, epoch
length, and per-topology membership. Runs anywhere (no dataset files).

--cluster-tsv TSV --chain-to-cat PKL: reads the full cluster universe via
read_cluster_tsv, loads the chain_to_cat pickle, builds a FakeDS
(database='pdb', file_names = all member ids), constructs the sampler, and
asserts the same invariants on real PDB or AFDB data. Topology counts reflect
the FULL cluster universe (not a train split), matching the prior convention.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proteinfoundation.utils.cluster_utils import CATBalancedSampler, read_cluster_tsv


class FakeDS:
    """Minimal dataset stub exposing the attributes CATBalancedSampler reads."""

    def __init__(self, member_ids):
        self.database = "pdb"
        self.file_names = list(member_ids)


def _run_checks(clusters, chain_to_cat, label):
    members = sorted({m for ms in clusters.values() for m in ms})
    ds = FakeDS(members)
    sampler = CATBalancedSampler(
        dataset=ds,
        clusterid_to_seqid_mapping=clusters,
        chain_to_cat=chain_to_cat,
        cat_cluster_draw="largest",
        member_mode="cluster-random",
        nocat_bucket=True,
        nocat_bucket_draws=1,
        shuffle=False,
        seed=0,
        v2=True,
    )
    n_topo = len(sampler._topologies)
    n_clusters = len(sampler.cluster_names)
    n_nocat = len(sampler.nocat_clusters)
    print(f"[{label}] topologies={n_topo} clusters={n_clusters} nocat_clusters={n_nocat}")

    # PRIMARY: the case fix resolves labels (not everything dumped to no-CAT).
    assert n_topo > 0, f"[{label}] zero topologies - chain_to_cat matched nothing (case-fix regressed?)"
    assert n_nocat < n_clusters, f"[{label}] all clusters are no-CAT - labels not resolving"

    # Single-rank epoch length = #topologies + nocat_bucket_draws.
    idx = list(iter(sampler))
    exp_len = n_topo + (sampler.nocat_bucket_draws if sampler.nocat_clusters else 0)
    assert len(idx) == exp_len, f"[{label}] epoch len {len(idx)} != {exp_len}"

    # shuffle=False => idx[i] corresponds to _topologies[i]; that chain's CAT set must contain it.
    idx_to_id = {i: m for m, i in sampler.sequence_id_to_idx.items()}
    checked = 0
    for i, cat in enumerate(sampler._topologies):
        chain_id = idx_to_id[idx[i]]
        cats = chain_to_cat.get(chain_id) or chain_to_cat.get(chain_id.lower(), [])
        if cats:  # infilled unlabeled members legitimately carry no CAT
            assert cat in cats, f"[{label}] topo {cat} -> chain {chain_id} cats {cats} missing topo"
            checked += 1
    print(f"[{label}] per-topology membership verified on {checked}/{n_topo} labeled draws")

    # 'largest' returns the max-eligible cluster per topology.
    for cat in sampler._topologies:
        cl = sampler.topology_to_clusters[cat]
        chosen = sorted(cl, key=lambda c: (-sampler.cluster_size_for_T[(c, cat)], c))[0]
        best = max(sampler.cluster_size_for_T[(c, cat)] for c in cl)
        assert sampler.cluster_size_for_T[(chosen, cat)] == best, f"[{label}] largest mismatch for {cat}"

    # set_epoch determinism (same epoch identical; different epoch differs when enough entropy).
    sampler.shuffle = True
    sampler.set_epoch(3)
    a = list(iter(sampler))
    sampler.set_epoch(3)
    b = list(iter(sampler))
    sampler.set_epoch(4)
    c = list(iter(sampler))
    assert a == b, f"[{label}] same epoch not deterministic"
    if n_topo >= 8:
        assert a != c, f"[{label}] different epoch produced identical sequence"
    print(f"[{label}] ALL CHECKS PASSED")


def selftest():
    chain_to_cat = {
        "AF-A1-F1-model_v6": ["1.10.10"],             # AFDB, single CAT
        "AF-A2-F1-model_v6": ["1.10.10"],             # AFDB, same CAT (in a bigger cluster)
        "AF-B1-F1-model_v6": ["2.20.20", "3.30.30"],  # AFDB, multi-domain (2 CATs)
        "1abc_a": ["2.20.20"],                        # PDB-style lowercase
    }
    clusters = {
        "AF-A1-F1-model_v6": ["AF-A1-F1-model_v6", "AF-A1b-F1-model_v6"],
        "AF-A2-F1-model_v6": ["AF-A2-F1-model_v6", "AF-A2b-F1-model_v6", "AF-A2c-F1-model_v6"],
        "AF-B1-F1-model_v6": ["AF-B1-F1-model_v6", "1abc_a"],
        "AF-Z1-F1-model_v6": ["AF-Z1-F1-model_v6", "AF-Z2-F1-model_v6"],
    }
    _run_checks(clusters, chain_to_cat, "selftest")

    ds = FakeDS(sorted({m for ms in clusters.values() for m in ms}))
    s = CATBalancedSampler(ds, clusters, chain_to_cat, cat_cluster_draw="largest",
                           member_mode="cluster-random", seed=0, v2=True, shuffle=False)
    assert "1.10.10" in s._topologies, "AFDB uppercase key did not resolve (exact-case .get failed)"
    assert "2.20.20" in s._topologies and "3.30.30" in s._topologies, "multi-domain CATs missing"
    cl = s.topology_to_clusters["1.10.10"]
    chosen = sorted(cl, key=lambda c: (-s.cluster_size_for_T[(c, "1.10.10")], c))[0]
    assert chosen == "AF-A2-F1-model_v6", f"largest picked {chosen}, expected AF-A2-F1-model_v6"
    assert "AF-Z1-F1-model_v6" in s.nocat_clusters, "no-CAT cluster not in nocat bucket"
    print("[selftest] case-fix + infill + largest + nocat bucketing OK")


def realtest(cluster_tsv, chain_to_cat_pkl):
    clusters = read_cluster_tsv(Path(cluster_tsv))
    with open(chain_to_cat_pkl, "rb") as fh:
        chain_to_cat = pickle.load(fh)
    print(f"[real] clusters={len(clusters)} chain_to_cat_keys={len(chain_to_cat)}")
    _run_checks(clusters, chain_to_cat, f"real:{Path(cluster_tsv).name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--cluster-tsv")
    p.add_argument("--chain-to-cat")
    args = p.parse_args()
    if args.selftest:
        selftest()
    if args.cluster_tsv and args.chain_to_cat:
        realtest(args.cluster_tsv, args.chain_to_cat)
    if not args.selftest and not (args.cluster_tsv and args.chain_to_cat):
        p.error("pass --selftest and/or (--cluster-tsv AND --chain-to-cat)")


if __name__ == "__main__":
    main()
