"""Precompute chain_to_cat: maps each PDB chain (lowercased "<pdb>_<chain>") to the
sorted unique list of CAT (3-level CATH: Class.Architecture.Topology) codes covering
all of its domains. Multi-domain chains keep all their CATs. Output is a pickle
consumed by CATBalancedSampler. Reuses CATHLabelTransform's parsed SIFTS/CATH maps.

Run on Engaging (DATA_PATH set), CPU-only:
    conda activate cue_openfold
    python script_utils/precompute_chain_to_cat.py
"""

import argparse
import os
import pickle

from proteinfoundation.datasets.transforms import CATHLabelTransform
from proteinfoundation.utils.ff_utils.pdb_utils import extract_cath_code_by_level
from proteinfoundation.utils.cluster_utils import read_cluster_tsv

DEFAULT_CLUSTER_TSV_NAME = (
    "cluster_seqid_0.25_df_pdb_f1_minl50_maxl256_mtprotein_etdiffractionEM_"
    "minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse.tsv"
)

# Reference counts from the 2026-05 prerequisite CAT-purity analysis (for eyeballing).
EXPECTED = {
    "clusters": 10187,
    "labeled_clusters": 7776,
    "nocat_clusters": 2411,
    "pure": 6630,
    "impure": 1146,
    "distinct_cat": 1463,
}


def build_chain_to_cat(cath_root):
    transform = CATHLabelTransform(root_dir=cath_root)
    # transform.pdbchain_to_cathid_mapping is a FrozenStrMap (refcount-free lookup-only
    # structure for forked dataloader workers, see frozen_str_map.py) -- no .items(), so
    # re-parse the raw dict directly instead (cheap: file is already downloaded/cached).
    pdbchain_to_cathid = transform._parse_cath_id()
    cathid_to_code = transform.cathid_to_cathcode_mapping
    chain_to_cat = {}
    for chain_key, cath_ids in pdbchain_to_cathid.items():
        cats = set()
        for cath_id in cath_ids:
            code = cathid_to_code.get(cath_id)
            if code is None:
                continue
            cats.add(extract_cath_code_by_level(code, "T"))
        if cats:
            chain_to_cat[chain_key.lower()] = sorted(cats)
    return chain_to_cat


def report_against_clusters(chain_to_cat, cluster_tsv):
    cluster_dict = read_cluster_tsv(cluster_tsv)
    labeled_clusters = nocat_clusters = pure = impure = 0
    labeled_chains = total_chains = 0
    distinct_cats = set()
    for members in cluster_dict.values():
        cluster_cats = set()
        for member in members:
            total_chains += 1
            cats = chain_to_cat.get(member.lower())
            if cats:
                labeled_chains += 1
                cluster_cats.update(cats)
        if cluster_cats:
            labeled_clusters += 1
            distinct_cats.update(cluster_cats)
            if len(cluster_cats) == 1:
                pure += 1
            else:
                impure += 1
        else:
            nocat_clusters += 1
    return {
        "clusters": len(cluster_dict),
        "labeled_clusters": labeled_clusters,
        "nocat_clusters": nocat_clusters,
        "pure": pure,
        "impure": impure,
        "distinct_cat": len(distinct_cats),
        "total_chains": total_chains,
        "labeled_chains": labeled_chains,
    }


def print_report(stats):
    print("\n=== chain_to_cat vs training cluster TSV ===")
    print(f"{'metric':<22}{'observed':>10}{'expected':>10}")
    for key in ["clusters", "labeled_clusters", "nocat_clusters", "pure", "impure", "distinct_cat"]:
        print(f"{key:<22}{stats[key]:>10}{EXPECTED[key]:>10}")
    print(f"{'total_chains':<22}{stats['total_chains']:>10}")
    print(f"{'labeled_chains':<22}{stats['labeled_chains']:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute chain_to_cat (chain_id_lower -> sorted list[CAT code])."
    )
    parser.add_argument("--cath-root", default=None, help="CATH data root (default: $DATA_PATH/cathdata)")
    parser.add_argument("--cluster-tsv", default=None, help="Training cluster TSV for the sanity report (default: $DATA_PATH/pdb_train/<known name>)")
    parser.add_argument("--out", default=None, help="Output pickle (default: $DATA_PATH/cath_shared/pdb_chain_to_cat.pkl)")
    parser.add_argument("--no-report", action="store_true", help="Skip the cluster-TSV sanity report.")
    args = parser.parse_args()

    data_path = os.environ.get("DATA_PATH")
    if data_path is None and None in (args.cath_root, args.out, args.cluster_tsv):
        parser.error("DATA_PATH env not set; pass --cath-root, --out, and --cluster-tsv explicitly.")
    cath_root = args.cath_root or os.path.join(data_path, "cathdata")
    out_path = args.out or os.path.join(data_path, "cath_shared", "pdb_chain_to_cat.pkl")
    cluster_tsv = args.cluster_tsv or os.path.join(data_path, "pdb_train", DEFAULT_CLUSTER_TSV_NAME)

    chain_to_cat = build_chain_to_cat(cath_root)
    all_cats = set()
    for cats in chain_to_cat.values():
        all_cats.update(cats)
    multi = sum(1 for cats in chain_to_cat.values() if len(cats) > 1)
    print(f"Built chain_to_cat: {len(chain_to_cat)} chains, {len(all_cats)} distinct CAT codes (all SIFTS chains).")
    print(f"Multi-CAT (multi-domain) chains: {multi}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(chain_to_cat, f)
    print(f"Wrote pickle -> {out_path}")

    if args.no_report:
        return
    if not os.path.exists(cluster_tsv):
        print(f"[report skipped] cluster TSV not found: {cluster_tsv}")
        return
    stats = report_against_clusters(chain_to_cat, cluster_tsv)
    print_report(stats)
    assert stats["labeled_clusters"] > 5000, (
        f"Only {stats['labeled_clusters']} labeled clusters (expected ~7776) -- likely a "
        "chain-id case/format mismatch between SIFTS keys and cluster seqids."
    )


if __name__ == "__main__":
    main()
