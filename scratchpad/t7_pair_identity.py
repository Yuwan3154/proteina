"""T7 pre-flight (user 2026-09-30): do the CB-8 and ConFind tris see the SAME (query, template) pairs?

Builds each tri's TopologyReferenceTransform exactly as its dataset config does (hydra compose + instantiate, with that
tri's SYNTH_INDEX_DIR), then asks each for nonself_reference(stem, L, seed) on every listed query -- the call the
validation-sampling path makes. Reports, per query, the template each definition would use and whether it is eligible
at all, and checks the index arrays that drive the choice (ids, cluster_of, members, row_tm) for identity.

Usage: python scratchpad/t7_pair_identity.py CHAINS OUT.tsv [SEED]
"""

import os
import sys

import hydra
import torch
from omegaconf import OmegaConf

S = "/orcd/scratch/orcd/011/chenxiou"
DEFS = (("cb8", "pdb_train_contact-CB8-synthtopo_S25_max384_purge-test_cutoff-190828", f"{S}/synth_index_v4"),
        ("confind", "pdb_train_contact-confind-synthtopo_S25_max384_purge-test_cutoff-190828",
         f"{S}/synth_index_confind_v1"))
KEYS = ("ids", "cluster_of", "members_offset", "members_flat", "row_tm")


def build(dataset, index_dir):
    os.environ["SYNTH_INDEX_DIR"] = index_dir
    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=dataset)
    nodes = [t for t in cfg.datamodule.transforms if str(t["_target_"]).endswith("TopologyReferenceTransform")]
    assert len(nodes) == 1, f"{dataset}: {len(nodes)} TopologyReferenceTransform nodes"
    node = OmegaConf.to_container(nodes[0], resolve=True)
    print(f"[{dataset}] index_path={node['index_path']} source={node['reference_source']} tm_range={node['tm_range']}",
          flush=True)
    t = hydra.utils.instantiate(node)
    t._ensure_loaded()
    return t


def main():
    chains, out = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    stems = sorted({ln.split()[0] for ln in open(chains) if ln.strip()})
    tr = {name: build(ds, d) for name, ds, d in DEFS}
    a, b = tr["cb8"]._index, tr["confind"]._index
    for k in KEYS:
        va, vb = a[k], b[k]
        same = (va == vb) if isinstance(va, list) else (va.shape == vb.shape and bool(torch.equal(va, vb)))
        print(f"[index] {k}: {'IDENTICAL' if same else 'DIFFERENT'}  ({len(va)} entries)", flush=True)
    rows, n_same, n_both = [], 0, 0
    for st in stems:
        r = {}
        for name in tr:
            got = tr[name].nonself_reference(st, 384, seed=seed)
            r[name] = "" if got is None else got[1]
        both = bool(r["cb8"]) and bool(r["confind"])
        same = both and r["cb8"] == r["confind"]
        n_both += both
        n_same += same
        rows.append((st, r["cb8"], r["confind"], int(same)))
    with open(out, "w") as fh:
        fh.write("query\ttemplate_cb8\ttemplate_confind\tsame\n")
        for row in rows:
            fh.write("\t".join(map(str, row)) + "\n")
    print(f"[pairs] {len(stems)} queries: eligible in both {n_both}, identical template {n_same}, "
          f"cb8-only {sum(1 for _, x, y, _ in rows if x and not y)}, confind-only "
          f"{sum(1 for _, x, y, _ in rows if y and not x)}, neither {sum(1 for _, x, y, _ in rows if not x and not y)}")
    print(f"PAIRS_IDENTICAL={int(n_same == len(stems))}")
    sys.exit(0 if n_same == len(stems) else 1)


if __name__ == "__main__":
    main()
