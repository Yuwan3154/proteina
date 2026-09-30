"""T7-QUICK subset (user 2026-09-30): held-out union, L <= 256, clustered WITHIN the set by fold.

Native CA traces come from the processed .pt files (repo's own sharded-path logic); all-vs-all USalign (default
sequence-independent TM-align mode) in parallel; single-linkage clusters at TM >= 0.5 (same fold) normalised by the
SHORTER chain; one representative per cluster = its medoid (highest mean TM to the other members; singletons are their
own representative). Writes the subset list, a cluster table, and chunk lists (each query repeated K times, whole
queries per chunk).

Usage: python scratchpad/t7_subset.py UNION_LIST SAMPLES_JSONL OUT_DIR [--lmax 256] [--k 8] [--chunk 25] [--procs 8]
"""

import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

USALIGN = "/home/chenxiou/.local/bin/USalign"
DATA = pathlib.Path("/orcd/pool/006/chenxiou/proteina/data/pdb_train")
TM_FOLD = 0.5


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def write_ca(stem, path, man):
    g = torch.load(_processed_path_sharded(DATA / "processed", stem, man), weights_only=False)
    ca = g.coords[:, 1, :].numpy()
    ok = g.coord_mask[:, 1].numpy().astype(bool) if hasattr(g, "coord_mask") else np.ones(len(ca), bool)
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(ca):
            if ok[i]:
                fh.write(f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        fh.write("END\n")
    return int(ok.sum())


def tm_pair(a, b):
    out = subprocess.run([USALIGN, a, b, "-outfmt", "2"], capture_output=True, text=True, check=True).stdout
    f = out.strip().splitlines()[-1].split("\t")
    return float(f[2]), float(f[3])  # TM normalised by chain 1, by chain 2


def main():
    union, samples, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    lmax, k, chunk, procs = arg("--lmax", 256), arg("--k", 8), arg("--chunk", 25), arg("--procs", 8)
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted({ln.split()[0] for ln in open(union) if ln.strip()})
    L = {json.loads(l)["stem"]: json.loads(l)["L"] for l in open(samples)}
    assert set(stems) <= set(L), "union chains missing from the samples table"
    sel = [s for s in stems if L[s] <= lmax]
    print(f"[subset] union {len(stems)} -> L<={lmax}: {len(sel)}", flush=True)
    man = json.load(open(DATA / "shard_manifest.json"))
    ca_dir = out / "ca"
    ca_dir.mkdir(exist_ok=True)
    for s in sel:
        write_ca(s, ca_dir / f"{s}.pdb", man)
    pairs = [(i, j) for i in range(len(sel)) for j in range(i + 1, len(sel))]
    with ThreadPoolExecutor(procs) as ex:
        tms = list(ex.map(lambda p: tm_pair(str(ca_dir / f"{sel[p[0]]}.pdb"), str(ca_dir / f"{sel[p[1]]}.pdb")), pairs))
    n = len(sel)
    M = np.eye(n)
    for (i, j), (t1, t2) in zip(pairs, tms):
        M[i, j] = M[j, i] = max(t1, t2)  # shorter-chain normalisation = the larger of the two
    np.save(out / "tm_matrix.npy", M)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] >= TM_FOLD:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    reps = []
    with open(out / "clusters.tsv", "w") as fh:
        fh.write("cluster\trepresentative\tsize\tmembers\n")
        for c, mem in enumerate(sorted(groups.values(), key=lambda m: (-len(m), sel[m[0]]))):
            rep = mem[0] if len(mem) == 1 else max(mem, key=lambda i: np.mean([M[i, j] for j in mem if j != i]))
            reps.append(sel[rep])
            fh.write(f"{c}\t{sel[rep]}\t{len(mem)}\t{','.join(sel[i] for i in mem)}\n")
    reps = sorted(reps)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    print(f"[cluster] TM>={TM_FOLD} (shorter-chain norm), single linkage: {len(groups)} clusters from {n} chains; "
          f"largest sizes {sizes[:8]}; singletons {sum(1 for x in sizes if x == 1)}", flush=True)
    (out / "subset.txt").write_text("\n".join(reps) + "\n")
    # chunks balanced by total residues x K (sampling cost grows with L), whole queries per chunk
    order = sorted(reps, key=lambda s: -L[s])
    nch = max(1, -(-len(reps) // chunk))
    bins = [[] for _ in range(nch)]
    load = [0] * nch
    for s in order:
        b = int(np.argmin(load))
        bins[b].append(s)
        load[b] += L[s]
    for b, qs in enumerate(bins):
        (out / f"chunk{b:02d}.txt").write_text("".join(f"{s}\n" * k for s in sorted(qs)))
        (out / f"chunk{b:02d}_q.txt").write_text("".join(f"{s}\n" for s in sorted(qs)))
    print(f"[chunks] {nch} chunks x ~{chunk} queries x {k} samples; residue load {load}", flush=True)
    print(f"SUBSET_N={len(reps)}")


if __name__ == "__main__":
    main()
