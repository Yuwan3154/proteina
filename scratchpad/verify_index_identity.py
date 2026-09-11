"""Verify on the FINISHED index that every template really is the query's own molecule.

The builder now refuses a mismatched template (identity gate) and translates label->auth through a
fail-closed alias. Both are upstream guards. This checks the ARTIFACT they produced, independently:
for a sample of chains carried by the index, re-resolve the pool npz through the alias and compare
its aatype to the query's residue_type from the processed .pt.

"COMPLETED" is not "correct", and a guard that silently no-ops looks exactly like a guard that
passed -- so this re-derives the answer from the inputs rather than trusting the build log.

Also reports how many templates the gate REFUSED during the build (from the skips files): that count
is the direct measure of how much mis-joined data the old index was carrying.
"""

import argparse
import collections
import glob
import json
import pathlib
import random
import zlib

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--alias", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--processed-dir", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--skips-glob", default="")
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

alias = {}
with open(a.alias) as fh:
    head = fh.readline().rstrip("\n").split("\t")
    il, ia = head.index("label_id"), head.index("auth_id")
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) > max(il, ia) and f[ia]:
            alias[f[il]] = f[ia]

idx = torch.load(a.index, map_location="cpu", weights_only=False)
ids = [str(x) for x in idx["ids"]]
is_native = idx["row_is_native"]
mem_off, mem_flat = idx["members_offset"], idx["members_flat"]
native_rows = torch.nonzero(is_native).flatten()
manifest = json.load(open(a.manifest))
pool = pathlib.Path(a.pool)

# chains that actually carry templates -- those are the ones a wrong join would have corrupted
with_tpl = [c for c in range(int(mem_off.numel()) - 1) if int(mem_off[c + 1]) > int(mem_off[c])]
random.Random(0).shuffle(with_tpl)

checked = mismatch = no_pool = no_pt = 0
examples = []
for c in with_tpl:
    if checked >= a.n:
        break
    stem = ids[int(native_rows[c])].split("@")[0]
    auth = alias.get(stem)
    if not auth:
        continue
    p = _processed_path_sharded(pathlib.Path(a.processed_dir), stem, manifest)
    if not p.exists():
        no_pt += 1
        continue
    g = torch.load(str(p), map_location="cpu", weights_only=False)
    q = np.asarray(g.residue_type).astype(np.int16)
    f = pool / f"shard{zlib.crc32(auth.encode()) % 1000:04d}" / f"{auth}.npz"
    if not f.exists():
        no_pool += 1
        continue
    z = np.load(f)
    if "aatype" not in z:
        continue
    t = np.asarray(z["aatype"]).astype(np.int16)
    checked += 1
    if t.shape != q.shape or not bool((t == q).all()):
        mismatch += 1
        if len(examples) < 5:
            examples.append((stem, auth, len(q), len(t)))

assert checked >= 50, f"VACUOUS: only {checked} chains verified"
print(f"chains with templates VERIFIED (non-vacuous): {checked}")
print(f"  templates that are NOT the query's molecule: {mismatch}")
print(f"  (skipped: {no_pool} without a pool npz, {no_pt} without a processed .pt)")
for stem, auth, lq, lt in examples:
    print(f"    MISMATCH {stem} -> {auth}: query {lq} res, template {lt} res")

if a.skips_glob:
    tally = collections.Counter()
    for f in glob.glob(a.skips_glob):
        with open(f) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    tally[p[2].split(":")[0]] += 1
    print("\nbuild-time skips by reason:")
    for k, v in tally.most_common(12):
        print(f"  {k:34} {v}")
    print(f"  -> identity_mismatch is the count of mis-joined templates the gate REFUSED")

print("\nVERDICT:", "PASS -- every sampled template is the query's own molecule" if mismatch == 0
      else f"FAIL -- {mismatch} of {checked} sampled templates are a different molecule")
