"""D36tc input gates on the processed .pt (no tolerance parameters anywhere).

targets : sequence == eval_bad_afdb_relaxed42 `sequence`, or an exact in-order SUBSEQUENCE of it (proteina drops
          non-standard residues: 8F3K_A MSE, 6QBL_A PCA are the D36-recorded cases); coords present.
hits    : sequence vs foldseek tSeq, containment in EITHER direction (D36m gate: proteina drops non-standard residues;
          an assembly can resolve residues PDB100 omits); anything else = a different molecule.
Optional --stage dssp / f2c additionally require dssp_target / contact_map_confind [L, L] on every .pt.
"""
import argparse
import json
import os
import sys

import torch


def is_subsequence(short, long):
    it = iter(long)
    return all(c in it for c in short)


ap = argparse.ArgumentParser()
ap.add_argument("--processed-dir", required=True)
ap.add_argument("--target-seqs", required=True, help="json {target: sequence}")
ap.add_argument("--hit-manifest", required=True, help="json {target: {ref_stem, tseq}}")
ap.add_argument("--stage", choices=("coords", "dssp", "f2c"), default="coords")
ap.add_argument("--extra", default="", help="file of extra stems that only need the --stage attribute check")
args = ap.parse_args()

tseqs = json.load(open(args.target_seqs))
hits = json.load(open(args.hit_manifest))
extra = [l.split()[0] for l in open(args.extra) if l.strip()] if args.extra else []
bad, notes, lens = [], [], {}


def load(stem):
    p = os.path.join(args.processed_dir, f"{stem}.pt")
    if not os.path.exists(p):
        bad.append(f"{stem}: no {p}")
        return None
    g = torch.load(p, map_location="cpu", weights_only=False)
    L = int(g.coords.shape[0])
    lens[stem] = L
    if args.stage in ("dssp", "f2c"):
        d = getattr(g, "dssp_target", None)
        if d is None or int(d.shape[0]) != L:
            bad.append(f"{stem}: dssp_target {None if d is None else tuple(d.shape)} for L {L}")
    if args.stage == "f2c":
        c = getattr(g, "contact_map_confind", None)
        if c is None or tuple(c.shape) != (L, L) or c.dtype != torch.float16:
            bad.append(f"{stem}: contact_map_confind {None if c is None else (tuple(c.shape), c.dtype)} for L {L}")
    return g


for t, ref in tseqs.items():
    g = load(t)
    if g is None:
        continue
    s = g.sequence
    if s == ref:
        pass
    elif is_subsequence(s, ref):
        notes.append(f"{t}: processed {len(s)} is a subsequence of eval {len(ref)}")
    else:
        bad.append(f"{t}: processed sequence ({len(s)}) is not the eval sequence ({len(ref)}) nor a subsequence")
for t, h in hits.items():
    g = load(h["ref_stem"])
    if g is None:
        continue
    s, ts = g.sequence, h["tseq"].upper()
    if s == ts:
        continue
    if is_subsequence(s, ts):
        notes.append(f"{t} -> {h['ref_stem']}: processed {len(s)} ⊂ tSeq {len(ts)}")
    elif is_subsequence(ts, s):
        notes.append(f"{t} -> {h['ref_stem']}: tSeq {len(ts)} ⊂ processed {len(s)}")
    else:
        bad.append(f"{t} -> {h['ref_stem']}: processed {len(s)} vs tSeq {len(ts)}: neither contains the other")
for e in extra:
    load(e)

n_ref = len({h["ref_stem"] for h in hits.values()})
print(f"[verify-{args.stage}] targets {len(tseqs)}, hits {len(hits)} ({n_ref} unique), extra {len(extra)}; "
      f"{len(lens)} .pt read; {len(bad)} failures")
for n in notes:
    print(f"  note {n}")
print("  lengths: " + " ".join(f"{k}:{v}" for k, v in sorted(lens.items())))
for b in bad:
    print(f"  FAIL {b}")
if bad:
    sys.exit(1)
print(f"D36TC_VERIFY_{args.stage.upper()}_OK")
