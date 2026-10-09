"""D36tc CONTROL A: my builder on TRAINING chains must reproduce the training data.

--part a1 : my processed .pt vs the stored pdb_train .pt -- coords, coord_mask, residue_type, sequence, dssp_target EXACT.
--part a23: A2 my Frame2ConFind map vs the stored contact_map_confind: contact Jaccard at >= 0.01 (pass rule, user
            2026-10-09: every chain >= 0.9894 = the min this F2C checkpoint reached vs the stored maps on 92 natives,
            D10 job 23623562), plus cell agreement and max|dp|;
            A3 index native rows built by d36tc_build_index.py from (i) the STORED .pt -> must equal the training index
            row EXACTLY (runs, he_size, he bytes, feat bytes); (ii) MY .pt -> differences reported.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

JACCARD_MIN = 0.9894
THR = 0.01


def stored_pt(data, stem, manifest):
    return Path(_processed_path_sharded(Path(data) / "processed", stem, manifest))


def row(idx, i):
    r = {}
    a, b = int(idx["runs_offset"][i]), int(idx["runs_offset"][i + 1])
    r["runs"] = idx["runs_flat"][a:b].clone()
    r["he_size"] = int(idx["he_size"][i])
    a, b = int(idx["he_offset"][i]), int(idx["he_offset"][i + 1])
    r["he"] = idx["he_flat"][a:b].clone()
    a, b = int(idx["feat_offset"][i]), int(idx["feat_offset"][i + 1])
    r["feat"] = idx["feat_flat"][a:b].clone()
    return r


def compare_rows(x, y):
    out = []
    if not torch.equal(x["runs"], y["runs"]):
        out.append(f"runs {x['runs'].shape[0]} vs {y['runs'].shape[0]} elements")
    if x["he_size"] != y["he_size"]:
        out.append(f"he_size {x['he_size']} vs {y['he_size']}")
    if not torch.equal(x["he"], y["he"]):
        n = min(x["he"].numel(), y["he"].numel())
        out.append(f"he bytes differ ({int((x['he'][:n] != y['he'][:n]).sum())} of {n})")
    if not torch.equal(x["feat"], y["feat"]):
        if x["feat"].shape == y["feat"].shape:
            d = (x["feat"].float() - y["feat"].float()).abs()
            out.append(f"feat differ in {int((d > 0).sum())} of {d.numel()} (max |d| {float(d.max()):.4g})")
        else:
            out.append(f"feat sizes {x['feat'].numel()} vs {y['feat'].numel()}")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=("a1", "a23"), required=True)
ap.add_argument("--stems", required=True)
ap.add_argument("--mine", required=True, help="my processed dir")
ap.add_argument("--train-data", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train")
ap.add_argument("--train-index", default="/orcd/scratch/orcd/011/chenxiou/synth_index_confind_v1/topology_index_confind_synth.pt")
ap.add_argument("--idx-stored", default="", help="index built from the stored .pt (a23)")
ap.add_argument("--idx-mine", default="", help="index built from my .pt (a23)")
ap.add_argument("--write-stored-paths", default="", help="a1: write '<stem>\\t<stored path>' lines here")
args = ap.parse_args()

stems = [l.split()[0] for l in open(args.stems) if l.strip()]
assert len(stems) == 3 == len(set(stems)), f"Control A needs exactly 3 distinct chains, got {stems}"
manifest = json.load(open(Path(args.train_data) / "shard_manifest.json"))
fails = []
if args.part == "a1":
    lines = []
    for st in stems:
        sp = stored_pt(args.train_data, st, manifest)
        lines.append(f"{st}\t{sp}")
        s = torch.load(sp, map_location="cpu", weights_only=False)
        m = torch.load(Path(args.mine) / f"{st}.pt", map_location="cpu", weights_only=False)
        for k in ("coords", "coord_mask", "residue_type", "dssp_target"):
            a, b = getattr(s, k, None), getattr(m, k, None)
            ok = a is not None and b is not None and a.shape == b.shape and torch.equal(a, b)
            print(f"  A1 {st} {k}: {'EXACT' if ok else 'MISMATCH'} "
                  f"(stored {None if a is None else tuple(a.shape)}, mine {None if b is None else tuple(b.shape)})")
            if not ok:
                fails.append(f"{st} {k}")
        ok = getattr(s, "sequence", None) == getattr(m, "sequence", None)
        print(f"  A1 {st} sequence: {'EXACT' if ok else 'MISMATCH'} (len {len(getattr(m, 'sequence', '') or '')})")
        if not ok:
            fails.append(f"{st} sequence")
    if args.write_stored_paths:
        open(args.write_stored_paths, "w").write("\n".join(lines) + "\n")
else:
    for st in stems:
        s = torch.load(stored_pt(args.train_data, st, manifest), map_location="cpu", weights_only=False)
        m = torch.load(Path(args.mine) / f"{st}.pt", map_location="cpu", weights_only=False)
        ps, pm = s.contact_map_confind.float(), m.contact_map_confind.float()
        assert ps.shape == pm.shape, f"{st}: map shapes {tuple(ps.shape)} vs {tuple(pm.shape)}"
        a, b = pm >= THR, ps >= THR
        jac = float((a & b).sum() / (a | b).sum().clamp_min(1))
        agree = float((a == b).float().mean())
        dmax = float((pm - ps).abs().max())
        verdict = "PASS" if jac >= JACCARD_MIN else "FAIL"
        print(f"  A2 {st} L {ps.shape[0]}: contact Jaccard {jac:.4f} (>= {JACCARD_MIN}: {verdict}), "
              f"cell agreement {agree:.6f}, max|dp| {dmax:.4g}, contacts mine {int(a.sum())} stored {int(b.sum())}")
        if verdict == "FAIL":
            fails.append(f"{st} A2 Jaccard {jac:.4f}")
    tr = torch.load(args.train_index, map_location="cpu", weights_only=False, mmap=True)
    pos = {s: i for i, s in enumerate(tr["ids"]) if s in set(stems)}
    i_st = torch.load(args.idx_stored, map_location="cpu", weights_only=False)
    i_mi = torch.load(args.idx_mine, map_location="cpu", weights_only=False)
    for st in stems:
        assert st in pos, f"{st} not in the training index"
        t = row(tr, pos[st])
        d_st = compare_rows(row(i_st, list(i_st["ids"]).index(st)), t)
        d_mi = compare_rows(row(i_mi, list(i_mi["ids"]).index(st)), t)
        print(f"  A3 {st} (training row {pos[st]}): stored-map row {'IDENTICAL' if not d_st else 'DIFFERS: ' + '; '.join(d_st)}"
              f" | my-map row {'identical' if not d_mi else 'differs: ' + '; '.join(d_mi)}")
        if d_st:
            fails.append(f"{st} A3 stored-map row")
print(f"[controlA {args.part}] {len(stems)} chains, {len(fails)} failures: {fails}")
sys.exit(1 if fails else 0)
