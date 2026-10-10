"""Gate for t8_make_split.py, end to end on a toy data dir through the real PDBDataSplitter.

1200 singleton max384 clusters (so the 0.019 / 0.001 ratios give non-empty val and test), plus planted cases:
  n1 joins a train chain's cluster at 512 -> train;  n2 joins a val chain's -> val;  n3 joins a train+val cluster -> OUT;
  n4 alone -> PDBDataSplitter split;  d (new decoy) joins train chain T3 -> d val, T3 kept train although the 512
  exclusion removes its cluster;  x shares a max384 cluster with decoy z -> removed at 384 -> stays OUT at 512.
Then: a dump with one flipped label must exit nonzero with "CONTROL FAILED"; a union holding a train chain must exit
nonzero at the gates; neither may write outputs. Run: python scratchpad/test_t8_make_split.py
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t8_make_split as M  # noqa: E402

COLS = ["id", "pdb", "chain", "length", "sequence", "deposition_date"]
N = 1200


def write_csv_and_clusters(d, n, lengths, clus):
    ident = M.IDENT.format(n)
    pd.DataFrame([[c, c[:-2], "A", L, "A" * L, "2010-01-01"] for c, L in lengths.items()], columns=COLS).to_csv(
        os.path.join(d, f"{ident}.csv"), index=False)
    with open(os.path.join(d, f"cluster_seqid_{M.SIM}_{ident}.tsv"), "w") as f:
        for m, rep in clus.items():
            f.write(f"{rep}\t{m}\n")
    with open(os.path.join(d, f"cluster_seqid_{M.SIM}_{ident}.fasta"), "w") as f:
        for rep in sorted(set(clus.values())):
            f.write(f">{rep}\n{'A' * lengths[rep]}\n")


def run(d, dump_dir, union, out):
    return subprocess.run([sys.executable, os.path.join(HERE, "t8_make_split.py"), d, dump_dir, union, out],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.dirname(HERE)})


def write_dump(sd, splits):
    os.makedirs(sd, exist_ok=True)
    for s in M.SPLITS:
        M.write_ids(os.path.join(sd, f"{s}_chain_ids.txt"), sorted(splits[s]))


def main():
    with tempfile.TemporaryDirectory() as d:
        L384 = {f"s{i}_A": 60 + i % 300 for i in range(N)}
        L384.update({"z_A": 100, "x_A": 100})
        clus384 = {c: c for c in L384}
        clus384["x_A"] = "z_A"
        M.write_ids(os.path.join(d, M.DECOYS), ["z_A", "d_A"])
        M.write_ids(os.path.join(d, "rosetta_decoys_val.txt"), ["z_A", "d_A"])
        M.write_ids(os.path.join(d, "rosetta_decoys_test.txt"), [])
        M.write_ids(os.path.join(d, M.DROPS), [])
        write_csv_and_clusters(d, 384, L384, clus384)
        re384, _ = M.code_split(d, M.IDENT.format(384))
        print("toy max384 split sizes:", {s: len(v) for s, v in re384.items()})
        assert re384["val"] and re384["test"] and "x_A" not in M.to_map(re384) and "z_A" in re384["val"]
        tr = sorted(re384["train"])
        T, T2, T3 = tr[0], tr[1], tr[2]
        V, V2 = sorted(c for c in re384["val"] if c != "z_A")[:2]

        L512 = {**L384, "n1_A": 400, "n2_A": 450, "n3_A": 500, "n4_A": 480, "d_A": 450}
        clus512 = {c: c for c in L512}
        clus512.update({"x_A": "x_A", "n1_A": T, "n2_A": V, V2: T2, "n3_A": T2, "d_A": T3})
        write_csv_and_clusters(d, 512, L512, clus512)
        union = os.path.join(d, "union.txt")
        M.write_ids(union, [V, "z_A"])

        write_dump(os.path.join(d, "good"), re384)
        M.main(d, os.path.join(d, "good"), union, os.path.join(d, "out"))
        fin = {s: set(M.read_ids(os.path.join(d, "out", f"{s}_chain_ids.txt"))) for s in M.SPLITS}
        where = {c: s for s, v in fin.items() for c in v}
        got = {k: where.get(k) for k in ("n1_A", "n2_A", "n3_A", "n4_A", "d_A", T3, "x_A", "z_A")}
        print("placements:", got)
        assert got["n1_A"] == "train" and got["n2_A"] == "val" and got["n3_A"] is None
        assert got["n4_A"] in M.SPLITS and got["d_A"] == "val" and got[T3] == "train"
        assert got["x_A"] is None and got["z_A"] == "val"
        assert M.read_ids(os.path.join(d, "out", "out_disagree.txt")) == ["n3_A"]
        assert M.read_ids(os.path.join(d, "out", "out_excluded_at_384.txt")) == ["x_A"]
        assert all(where[c] == s for s in M.SPLITS for c in re384[s]), "a max384 chain moved"
        print("PASS end to end: every rule branch placed as specified")

        bad = {s: set(v) for s, v in re384.items()}
        bad["val"].discard(V)
        bad["train"].add(V)
        write_dump(os.path.join(d, "bad"), bad)
        r = run(d, os.path.join(d, "bad"), union, os.path.join(d, "out_bad"))
        assert r.returncode != 0 and "CONTROL FAILED" in r.stderr, (r.returncode, r.stderr[-800:])
        assert not os.path.exists(os.path.join(d, "out_bad"))
        print("PASS control fires on a non-reproducing dump (exit", r.returncode, ")")

        union_bad = os.path.join(d, "union_bad.txt")
        M.write_ids(union_bad, [V, T])
        r = run(d, os.path.join(d, "good"), union_bad, os.path.join(d, "out_gate"))
        assert r.returncode != 0 and "GATE 195-union in T8 val/test: 1 of 2" in r.stdout, (r.returncode, r.stdout[-800:])
        assert not os.path.exists(os.path.join(d, "out_gate"))
        print("PASS union gate fires on a train chain in the held-out list (exit", r.returncode, ")")
    print("ALL PASS")


if __name__ == "__main__":
    main()
