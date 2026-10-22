"""Regression of the SEQRES re-processing path against the stored pack (repair plan S3; Engaging, CPU).

Arm A  flag OFF on a stratified sample of side-table status-ok stems with n_ins == 0 (long 400-512, deposits with >= 4
       chains, the largest raw mmCIFs of a random pool, random rest): every field of reprocess_chain(...) torch.equal / ==
       the stored pack entry (contact_map_confind excluded; bfactor compared like every other field).
Arm B  flag ON on the same stems: identical too (no insertion codes, monotone numbering).
Arm C  flag ON on ok-with-insertion-code stems: identical except residue_id, which may differ only by the insertion code.
Integration  1dle_B (merged): L_new = modelled standard residues of the chain in pdbx_poly_seq_scheme, label_seq strictly
       increasing, every non-merged old residue torch.equal its new counterpart; 2zcy_Y (nonmonotone): the new arrays are
       a permutation of the old ones in label_seq order.
Altloc chains are counted post hoc (label_alt_id set on an ATOM row of the chain), not stratified.
Usage: python scratchpad/test_seqres_regression.py --pack PACK --data-dir DATA_PATH/pdb_train --audit AUDIT.tsv \
           --out-dir OUT [--n-a 2000] [--n-c 300] [--workers 32]
"""

import argparse
import collections
import json
import os
import pathlib
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from biopandas.mmcif import PandasMmcif

from proteinfoundation.graphein_utils.graphein_utils import STANDARD_AMINO_ACID_MAPPING_1_TO_3, read_pdb_to_dataframe
from proteinfoundation.utils.reprocess_seqres_chains import (altloc_rows, label_seq_of, load_entry, old_to_new, open_pack,
                                                             raw_path, reprocess_chain)

SKIP = ("contact_map_confind",)
G = {}


def init_worker(pack, data_dir, out_dir):
    G["ent"], G["mm"], G["fh"] = open_pack(pack)
    G["raw"] = pathlib.Path(data_dir) / "raw"
    G["man"] = json.load(open(pathlib.Path(data_dir) / "shard_manifest.json"))
    G["staging"] = pathlib.Path(out_dir) / "staging" / str(os.getpid()) / "processed"


def compare(old, new, ins_ok):
    """(names of differing fields, residue_ids changed) for new vs the stored entry; bfactor must match too."""
    ko, kn = set(old.keys()) - set(SKIP), set(new.keys())
    diffs = [f"keys:{sorted(ko ^ kn)}"] if ko != kn else []
    n_rid = 0
    for k in sorted(ko & kn):
        a, b = old[k], new[k]
        if isinstance(a, torch.Tensor):
            same = isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        else:
            same = type(a) is type(b) and a == b
        if k == "residue_id" and ins_ok and not same and len(a) == len(b):
            n_rid = sum(x != y for x, y in zip(a, b))
            same = [r.rsplit(":", 1)[0] for r in a] == [r.rsplit(":", 1)[0] for r in b]
        if not same:
            diffs.append(k)
    return diffs, n_rid


def has_altloc(path, chain) -> bool:
    a = PandasMmcif().read_mmcif(path).get_model(1).df["ATOM"]
    alt = a.loc[a["label_asym_id"] == chain, "label_alt_id"]
    return bool(alt.map(lambda v: v is not None and v == v and str(v) not in (".", "?", "None")).any())


def run_stem(job):
    stem, arm = job
    old = load_entry(G["ent"], G["mm"], stem)
    status, g = reprocess_chain(stem, G["raw"], G["staging"], G["man"], arm != "A", old)
    res = {"stem": stem, "arm": arm, "status": status, "L": len(old.residue_id)}
    if g is None:
        return res
    res["diffs"], res["n_rid_changed"] = compare(old, g, ins_ok=arm == "C")
    if arm == "A":
        res["altloc"] = has_altloc(str(raw_path(G["raw"], stem.split("_")[0], G["man"])), stem.split("_", 1)[1])
    return res


def sample_arm_a(cats, lengths, raw_dir, man, n, rng):
    ok = sorted(s for s, c in cats.items() if c == "ok")
    n_chains = collections.Counter(s.split("_")[0] for s in cats)
    strata = collections.OrderedDict()
    long_ = [s for s in ok if 400 <= lengths[s] <= 512]
    multi = [s for s in ok if n_chains[s.split("_")[0]] >= 4]
    strata["long"] = rng.sample(long_, min(n // 5, len(long_)))
    multi = [s for s in multi if s not in set(strata["long"])]
    strata["multichain"] = rng.sample(multi, min(n // 5, len(multi)))
    pool = rng.sample(ok, min(5 * n, len(ok)))
    size = {s: raw_path(raw_dir, s.split("_")[0], man).stat().st_size for s in pool}
    taken = set(strata["long"]) | set(strata["multichain"])
    strata["large_deposit"] = [s for s in sorted(pool, key=lambda s: -size[s]) if s not in taken][:n // 5]
    picked = {s for v in strata.values() for s in v}
    rest = [s for s in ok if s not in picked]
    strata["random"] = rng.sample(rest, min(n - len(picked), len(rest)))
    return strata


def integration(pack, data_dir, out_dir):
    init_worker(pack, data_dir, out_dir)
    raw, man, staging = G["raw"], G["man"], G["staging"]
    std = set(STANDARD_AMINO_ACID_MAPPING_1_TO_3.values())
    for stem in ("1dle_B", "2zcy_Y"):
        pdb, chain = stem.split("_", 1)
        old = load_entry(G["ent"], G["mm"], stem)
        status, g = reprocess_chain(stem, raw, staging, man, True, old)
        assert g is not None, (stem, status)
        path = str(raw_path(raw, pdb, man))
        df = read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True)
        ls = label_seq_of(df, path, chain, g)
        m = old_to_new(old, g, altloc_rows(df, chain))
        assert bool((ls[1:] > ls[:-1]).all()), f"{stem}: label_seq not strictly increasing"
        if stem == "1dle_B":
            ps = PandasMmcif().read_mmcif(path).data["pdbx_poly_seq_scheme"]
            n_model = sum(a == chain and pm is not None and pm not in ("?", "None") and mon in std
                          for a, pm, mon in zip(ps["asym_id"], ps["pdb_mon_id"], ps["mon_id"]))
            n_cand = collections.Counter(r.rsplit(":", 1)[0] for r in g.residue_id)
            single = np.array([n_cand[r.rsplit(":", 1)[0]] == 1 for r in old.residue_id])
            assert len(g.residue_id) == n_model, (stem, len(g.residue_id), n_model)
            assert bool(m["exact"][single].all()), f"{stem}: {int((~m['exact'][single]).sum())} non-merged residues moved"
            print(f"PASS {stem}: L {len(old.residue_id)} -> {len(g.residue_id)} = poly_seq modelled standard {n_model}; "
                  f"{int(single.sum())} non-merged old residues exact; {int((~single).sum())} merged old residues, "
                  f"{int(m['chimera'].sum())} chimeric; label_seq strictly increasing")
        else:
            idx = m["index"]
            assert len(g.residue_id) == len(old.residue_id) and sorted(idx.tolist()) == list(range(len(idx))), stem
            assert bool(m["exact"].all()) and not bool((idx == np.arange(len(idx))).all()), stem
            t = torch.from_numpy(idx.astype(np.int64))
            for k in ("coords", "coord_mask", "residue_type", "residue_pdb_idx", "ext_lig"):
                assert torch.equal(g[k][t], old[k]), (stem, k)
            assert [g.residues[j] for j in idx] == list(old.residues), stem
            print(f"PASS {stem}: permutation of the old arrays ({int((idx != np.arange(len(idx))).sum())} of {len(idx)} "
                  f"moved); coords/coord_mask/residue_type/residue_pdb_idx/ext_lig/residues match; label_seq increasing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-a", type=int, default=2000)
    ap.add_argument("--n-c", type=int, default=300)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [l.rstrip("\n").split("\t") for l in list(open(a.audit))[1:]]
    cats = {r[0]: ("ok_ins" if r[1] == "ok" and int(r[5]) > 0 else r[1]) for r in rows}
    lengths = {r[0]: int(r[2]) for r in rows}
    data_dir = pathlib.Path(a.data_dir)
    man = json.load(open(data_dir / "shard_manifest.json"))
    strata = sample_arm_a(cats, lengths, data_dir / "raw", man, a.n_a, rng)
    arm_a = [s for v in strata.values() for s in v]
    assert all(len(v) > 0 for v in strata.values()), {k: len(v) for k, v in strata.items()}
    assert len(set(arm_a)) == len(arm_a) == a.n_a, (len(set(arm_a)), len(arm_a), a.n_a)
    ins = sorted(s for s, c in cats.items() if c == "ok_ins")
    arm_c = rng.sample(ins, min(a.n_c, len(ins)))
    assert len(arm_c) == a.n_c, (len(arm_c), a.n_c)
    print("arm A strata " + ", ".join(f"{k} {len(v)}" for k, v in strata.items()) + f"; arm C {len(arm_c)}", flush=True)
    jobs = [(s, "A") for s in arm_a] + [(s, "B") for s in arm_a] + [(s, "C") for s in arm_c]
    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(a.pack, a.data_dir, a.out_dir)) as ex:
        res = list(ex.map(run_stem, jobs, chunksize=4))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"strata": strata, "arm_c": arm_c, "results": res}, open(os.path.join(a.out_dir, "regression.json"), "w"))
    fail = 0
    for arm in "ABC":
        r = [x for x in res if x["arm"] == arm]
        bad = [x for x in r if x["status"] != "processed" or x["diffs"]]
        field = collections.Counter(d for x in r for d in x.get("diffs", []))
        extra = ""
        if arm == "A":
            n_alt = sum(x.get("altloc", False) for x in r)
            assert n_alt > 0, "arm A sampled no altloc chain"
            extra = f"; altloc chains {n_alt}"
        if arm == "C":
            extra = f"; residue_id changed in {sum(x.get('n_rid_changed', 0) > 0 for x in r)} chains"
        print(f"arm {arm}: {len(r)} stems, {len(bad)} with a difference or failure, fields {dict(field)}, "
              f"bfactor compared strictly{extra}")
        for x in bad[:20]:
            print(f"  {x['stem']} {x['status']} {x.get('diffs')}")
        fail += len(bad)
    integration(a.pack, a.data_dir, a.out_dir)
    assert fail == 0, f"{fail} stems differ (see {a.out_dir}/regression.json)"
    print("ALL PASS")


if __name__ == "__main__":
    main()
