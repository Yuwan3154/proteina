"""T8 fixed validation-sampling list by the old val_fixed32 rule (session-tracking RAW-ARCHIVE: "Drawn from the 69 of 282
representatives with NO neighbour at TM>=0.5, then length-stratified into 32 bins taking each bin's median-length member
(deterministic, ties by id)"), with the user's 2026-10-14 answers: (1) one representative per 25% val cluster = the
split's cluster rep, else the highest-resolution usable member (ties: chain id); (2) pair TM normalised by the SHORTER
chain (= the larger of the two US-align TMs); usable = a kept val chain with a template in the TM range (validation
conditions on a synthetic template). Step (3), pending the user: sort the isolated pool by (length, id), split into
--n-bins consecutive near-equal groups (np.array_split), take each group's median member (index (n-1)//2).
Writes pool.tsv (every representative: length, max TM to any other rep, isolated flag), pairs.tsv (every pair's TM)
and the --n-bins pick. --mode within (user 2026-10-14 "(b)"): "no pair TM >= 0.5 WITHIN the chosen set" instead of
isolation against all representatives, so the length range can reach 512: ALL representatives sorted by (length, id),
split into --n-bins near-equal groups; groups visited in length order; in each, members ordered by distance from the
group's median index (ties: id) and the first whose TM to every chain already picked is < --tm is taken; a group with
no such member is reported and left empty.
Usage (PYTHONPATH=repo): python scratchpad/t8_val_fixed32.py --clusters TSV --val TXT --eligible TXT --csv CSV
    --pack PACK --out DIR [--n-bins 32] [--mode isolated|within] [--tm 0.5] [--workers N]
"""

import argparse
import collections
import csv
import io
import itertools
import os
import subprocess
from multiprocessing import Pool

import numpy as np
import torch

from proteinfoundation.utils.precompute_synthetic_topology_index import CA_IDX, write_ca_pdb

_USALIGN, _DIR = None, None


def pair_tm(ab):
    """max(TM normalised by chain 1, TM normalised by chain 2) = TM normalised by the shorter chain."""
    a, b = ab
    out = subprocess.run([_USALIGN, os.path.join(_DIR, f"{a}.pdb"), os.path.join(_DIR, f"{b}.pdb"), "-TMscore", "0"],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"{a} {b}: USalign rc {out.returncode}"
    tms = [float(l.split("=")[1].split()[0]) for l in out.stdout.splitlines()
           if l.startswith("TM-score=") and ("Structure_1" in l or "Structure_2" in l)]
    assert len(tms) == 2, f"{a} {b}: expected 2 TM lines, got {len(tms)}"
    return a, b, max(tms)


def init(usalign, d):
    global _USALIGN, _DIR
    _USALIGN, _DIR = usalign, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="clusters_maxl512.tsv: rep, chain, split")
    ap.add_argument("--val", required=True, help="kept val chains (t8_template_list/val_chain_ids.txt)")
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--csv", required=True, help="selection csv with id + resolution (-1 = not recorded = worst)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--tm", type=float, default=0.5)
    ap.add_argument("--mode", choices=("isolated", "within"), default="isolated")
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "ca"), exist_ok=True)
    val = {l.strip() for l in open(args.val) if l.strip()}
    elig = {l.strip() for l in open(args.eligible) if l.strip()}
    usable = val & elig
    res = {}
    csv.field_size_limit(10**9)
    for r in csv.DictReader(open(args.csv)):
        if r["id"] in usable:
            x = float(r["resolution"])
            res[r["id"]] = float("inf") if x == -1.0 else x
    assert set(res) == usable, f"{len(usable - set(res))} usable chains missing from the csv"
    members = collections.defaultdict(list)
    for l in open(args.clusters):
        rep, ch, sp = l.rstrip("\n").split("\t")
        if sp == "val" and ch in val:
            members[rep].append(ch)
    reps, how = {}, collections.Counter()
    for rep, ms in sorted(members.items()):
        ok = [m for m in ms if m in usable]
        if not ok:
            how["no_usable_member"] += 1
            continue
        if rep in usable:
            reps[rep] = rep
            how["cluster_rep"] += 1
        else:
            reps[rep] = min(ok, key=lambda c: (res[c], c))
            how["highest_resolution_member"] += 1
    print(f"val clusters {len(members)}: {dict(how)}", flush=True)
    z = np.load(args.pack + ".idx.npz", allow_pickle=True)
    off = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    length = {}
    with open(args.pack, "rb") as fh:
        for c in reps.values():
            o, ln = off[c]
            fh.seek(o)
            g = torch.load(io.BytesIO(fh.read(ln)), map_location="cpu", weights_only=False)
            present = (g.coord_mask[:, CA_IDX] > 0.5).numpy()
            write_ca_pdb(g.coords[:, CA_IDX].numpy(), present, os.path.join(args.out, "ca", f"{c}.pdb"))
            length[c] = int(g.coords.shape[0])
    chains = sorted(reps.values())
    best = {c: 0.0 for c in chains}
    tm = {}
    with Pool(args.workers, initializer=init, initargs=(args.usalign, os.path.join(args.out, "ca"))) as p:
        for a, b, t in p.imap_unordered(pair_tm, itertools.combinations(chains, 2), chunksize=64):
            best[a], best[b] = max(best[a], t), max(best[b], t)
            tm[(a, b)] = tm[(b, a)] = t
    with open(os.path.join(args.out, "pairs.tsv"), "w") as f:
        f.write("a\tb\ttm_shorter\n")
        for a, b in itertools.combinations(chains, 2):
            f.write(f"{a}\t{b}\t{tm[(a, b)]:.4f}\n")
    iso = sorted((c for c in chains if best[c] < args.tm), key=lambda c: (length[c], c))
    with open(os.path.join(args.out, "pool.tsv"), "w") as f:
        f.write("chain\tlength\tmax_tm_to_other_rep\tisolated\n")
        for c in sorted(chains, key=lambda c: (length[c], c)):
            f.write(f"{c}\t{length[c]}\t{best[c]:.4f}\t{int(best[c] < args.tm)}\n")
    print(f"representatives {len(chains)}; isolated (no other rep at TM >= {args.tm}) {len(iso)}"
          + (f"; lengths {length[iso[0]]}-{length[iso[-1]]}" if iso else ""), flush=True)
    if args.mode == "isolated":
        assert len(iso) >= args.n_bins, f"only {len(iso)} isolated representatives for {args.n_bins} bins"
        pick = [g[(len(g) - 1) // 2] for g in np.array_split(np.array(iso, dtype=object), args.n_bins)]
    else:
        pick, empty = [], []
        allr = sorted(chains, key=lambda c: (length[c], c))
        for k, g in enumerate(np.array_split(np.array(allr, dtype=object), args.n_bins)):
            mid = (len(g) - 1) // 2
            order = sorted(range(len(g)), key=lambda i: (abs(i - mid), g[i]))
            got = next((g[i] for i in order if all(tm[(g[i], q)] < args.tm for q in pick)), None)
            if got is None:
                empty.append(k)
            else:
                pick.append(got)
        print(f"within mode: {len(pick)} picked, empty bins {empty}", flush=True)
    within = max((tm[(a, b)] for a, b in itertools.combinations(pick, 2)), default=0.0)
    with open(os.path.join(args.out, f"val_fixed{args.n_bins}_t8_{args.mode}.txt"), "w") as f:
        f.write("\n".join(pick) + "\n")
    print(f"pick {len(pick)}: lengths {[length[c] for c in pick]}; max TM between picks {within:.4f}")


if __name__ == "__main__":
    main()
