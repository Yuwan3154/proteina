"""T8b native ConFind maps: length caches for precompute_frame2confind_maps.py (CPU, Engaging compute node).

Cache format = backfill/build_full_band_caches.py: JSON list of [path, L, has_cm]; sorted by path string here (the original
band cache was in scan completion order; row order only changes equal-length batchmates, which the gate measures). The precompute trusts the cache as-is, skips L > --max-len and, without
--overwrite, has_cm=True rows; the 'file' column of its status CSV is the basename.

full (default): every status==changed stem of the status TSV, at its real path in the re-processed tree.
  Each file is loaded once: L must equal coords.shape[0] == L_new and contact_map_confind must be ABSENT.
  python t8b_confind_cache.py --status-tsv S --tree FIX/processed --out-cache C.json --out-stems stems.txt
--gate: n UNCHANGED stems with lmin <= L_new <= lmax (random.Random(seed)), ORIGINAL .pt copied into
  <gate-dir>/processed/<orig bucket>/ (buckets hold <= 1024 files), stored maps saved to <gate-dir>/ref_maps.pt,
  cache rows has_cm=True (the GPU gate run uses --overwrite), manifest.tsv.
  python t8b_confind_cache.py --gate --status-tsv S --tree ORIG/processed --gate-dir G
--padfix: the June pad-fix list (t8b_padfix_list.py list; UNCHANGED stems only) -- each ORIGINAL .pt copied to the same
  relative path (<bucket>/<stem>.pt, any depth) under the fix tree (tmp + rename; never a write under --tree), cache rows has_cm=True (the GPU run uses
  --overwrite on these copies only). Nothing is written if any destination exists with bytes != its original (a rerun
  over identical copies is a no-op), if a destination is outside --fix-tree, or if --fix-tree and --tree nest.
  python t8b_confind_cache.py --padfix --padfix-list L.tsv --status-tsv S --tree ORIG/processed --fix-tree FIX/processed \
      --out-cache C.json --out-stems stems.txt
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch


def read_status(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    stems = [r["stem"] for r in rows]
    assert len(stems) == len(set(stems)), "duplicate stems in status TSV"
    return rows


def find_paths(tree, stems):
    """One os.scandir pass over <tree>/<bucket>/*.pt -> {stem: path}; every stem must occur exactly once."""
    want = set(stems)
    hits = defaultdict(list)
    n_pt, n_other_top, n_other_bucket = 0, 0, 0
    for b in os.scandir(tree):
        if not b.is_dir():
            n_other_top += 1
            continue
        for e in os.scandir(b.path):
            if not e.name.endswith(".pt"):
                n_other_bucket += 1
                continue
            n_pt += 1
            stem = e.name[:-3]
            if stem in want:
                hits[stem].append(e.path)
    print(f"walk {tree}: {n_pt} .pt in buckets, {n_other_top} non-dir top-level entries, "
          f"{n_other_bucket} non-.pt bucket entries", flush=True)
    missing = sorted(want - set(hits))
    multi = sorted(s for s, p in hits.items() if len(p) > 1)
    assert not missing, f"{len(missing)} stems not found, e.g. {missing[:5]}"
    assert not multi, f"{len(multi)} stems found more than once, e.g. {[(s, hits[s]) for s in multi[:3]]}"
    return {s: hits[s][0] for s in stems}


def walk_paths(tree, stems):
    """find_paths at any depth (pdb_train holds flat files and <mid3>/<hash> sub-buckets too): {stem: path}, exactly once."""
    want = set(stems)
    hits = defaultdict(list)
    for d, _, files in os.walk(tree):
        for f in files:
            if f.endswith(".pt") and f[:-3] in want:
                hits[f[:-3]].append(os.path.join(d, f))
    missing = sorted(want - set(hits))
    multi = sorted(s for s, p in hits.items() if len(p) > 1)
    assert not missing, f"{len(missing)} stems not found, e.g. {missing[:5]}"
    assert not multi, f"{len(multi)} stems found more than once, e.g. {[(s, hits[s]) for s in multi[:3]]}"
    return {s: hits[s][0] for s in stems}


def probe(path):
    """(path, coords L, has contact_map_confind) -- the same fields as scan_both.scan_one."""
    g = torch.load(path, map_location="cpu", weights_only=False)
    return path, int(g.coords.shape[0]), getattr(g, "contact_map_confind", None) is not None


def probe_all(paths, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(probe, paths, chunksize=64))


def write_cache(rows, out_cache):
    rows = sorted(rows, key=lambda r: r[0])
    names = [os.path.basename(r[0]) for r in rows]
    assert len(names) == len(set(names)), "basename collision: status CSV keys on basename"
    os.makedirs(os.path.dirname(os.path.abspath(out_cache)), exist_ok=True)
    with open(out_cache, "w") as fh:
        json.dump(rows, fh)
    print(f"wrote {out_cache}: {len(rows)} rows", flush=True)


def build_full(args):
    rows = [r for r in read_status(args.status_tsv) if r["status"] == "changed"]
    L_new = {r["stem"]: int(r["L_new"]) for r in rows}
    print(f"changed stems: {len(L_new)}", flush=True)
    over = [s for s, L in L_new.items() if L > args.max_len]
    assert not over, f"{len(over)} changed stems have L_new > {args.max_len}, e.g. {over[:5]}"
    paths = find_paths(args.tree, list(L_new))
    probed = probe_all([paths[s] for s in L_new], args.workers)
    assert len(probed) == len(L_new)
    by_path = {p: (L, cm) for p, L, cm in probed}
    bad_L = [(s, L_new[s], by_path[paths[s]][0]) for s in L_new if by_path[paths[s]][0] != L_new[s]]
    has_cm = [s for s in L_new if by_path[paths[s]][1]]
    print(f"probed {len(probed)} of {len(L_new)}: L mismatch {len(bad_L)}, already has contact_map_confind "
          f"{len(has_cm)}", flush=True)
    assert not bad_L, f"coords L != L_new, e.g. {bad_L[:5]}"
    assert not has_cm, f"unexpected existing maps, e.g. {has_cm[:5]}"
    write_cache([[paths[s], L_new[s], False] for s in L_new], args.out_cache)
    with open(args.out_stems, "w") as fh:
        fh.write("".join(f"{s}\n" for s in sorted(L_new)))
    print(f"wrote {args.out_stems}: {len(L_new)} stems", flush=True)


def build_gate(args):
    rows = read_status(args.status_tsv)
    cand = sorted(r["stem"] for r in rows
                  if r["status"] == "unchanged" and args.lmin <= int(r["L_new"]) <= args.lmax)
    L_new = {r["stem"]: int(r["L_new"]) for r in rows}
    assert len(cand) >= args.n, f"only {len(cand)} candidates"
    pick = sorted(random.Random(args.seed).sample(cand, args.n))
    print(f"gate: {len(cand)} unchanged candidates with {args.lmin} <= L_new <= {args.lmax}; picked {len(pick)} "
          f"(seed {args.seed})", flush=True)
    src = find_paths(args.tree, pick)
    assert not os.path.exists(args.gate_dir), f"{args.gate_dir} exists; refusing to mix gate sets"
    dst = {s: os.path.join(args.gate_dir, "processed", os.path.basename(os.path.dirname(src[s])), f"{s}.pt")
           for s in pick}
    for s in pick:
        os.makedirs(os.path.dirname(dst[s]), exist_ok=True)
        shutil.copy2(src[s], dst[s])
    ref, cache = {}, []
    for s in pick:
        g = torch.load(dst[s], map_location="cpu", weights_only=False)
        L = int(g.coords.shape[0])
        assert L == L_new[s], (s, L, L_new[s])
        assert getattr(g, "contact_map_confind", None) is not None, s
        ref[s] = g.contact_map_confind.clone()
        cache.append([dst[s], L, True])
    per_dir = defaultdict(int)
    for s in pick:
        per_dir[os.path.dirname(dst[s])] += 1
    assert max(per_dir.values()) <= 1024
    torch.save(ref, os.path.join(args.gate_dir, "ref_maps.pt"))
    with open(os.path.join(args.gate_dir, "manifest.tsv"), "w") as fh:
        fh.write("stem\tL\torig_path\tgate_path\n")
        fh.write("".join(f"{s}\t{L_new[s]}\t{src[s]}\t{dst[s]}\n" for s in pick))
    write_cache(cache, os.path.join(args.gate_dir, "cache_gate.json"))
    print(f"gate dir {args.gate_dir}: {len(pick)} originals in {len(per_dir)} bucket dirs, ref_maps.pt + manifest.tsv",
          flush=True)


def md5_file(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def md5_all(paths, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(md5_file, paths, chunksize=16))


def build_padfix(args):
    with open(args.padfix_list, newline="") as fh:
        pad = [r["stem"] for r in csv.DictReader(fh, delimiter="\t")]
    assert len(pad) == len(set(pad)) > 0, "empty or duplicate pad-fix list"
    table = {r["stem"]: r for r in read_status(args.status_tsv)}
    bad = [s for s in pad if s not in table or table[s]["status"] != "unchanged"]
    assert not bad, f"{len(bad)} pad-fix stems are not status unchanged in the table, e.g. {bad[:5]}"
    tree, fix = args.tree.rstrip("/"), args.fix_tree.rstrip("/")
    assert not (tree + "/").startswith(fix + "/") and not (fix + "/").startswith(tree + "/"), (tree, fix)
    src = walk_paths(tree, pad)
    dst = {s: os.path.join(fix, os.path.relpath(src[s], tree)) for s in pad}
    assert all(d.startswith(fix + "/") and os.path.basename(d) == f"{s}.pt" for s, d in dst.items())
    src_md5 = dict(zip(pad, md5_all([src[s] for s in pad], args.workers)))
    exist = [s for s in pad if os.path.exists(dst[s])]
    conflict = [s for s in exist if md5_file(dst[s]) != src_md5[s]]
    assert not conflict, (f"{len(conflict)} destinations exist with other bytes (fix-tree file or an earlier recompute: "
                          f"ask), nothing written; e.g. {[dst[s] for s in conflict[:5]]}")
    print(f"padfix: {len(pad)} stems, {len(exist)} copies already present and identical, copying {len(pad) - len(exist)}",
          flush=True)
    for s in pad:
        if s not in exist:
            os.makedirs(os.path.dirname(dst[s]), exist_ok=True)
            shutil.copy2(src[s], dst[s] + ".tmp")
            os.replace(dst[s] + ".tmp", dst[s])
    dst_md5 = dict(zip(pad, md5_all([dst[s] for s in pad], args.workers)))
    assert dst_md5 == src_md5, "a copy differs from its original"
    probed = {p: (L, cm) for p, L, cm in probe_all([dst[s] for s in pad], args.workers)}
    L_new = {s: int(table[s]["L_new"]) for s in pad}
    bad_L = [(s, probed[dst[s]][0], L_new[s]) for s in pad if probed[dst[s]][0] != L_new[s]]
    no_cm = [s for s in pad if not probed[dst[s]][1]]
    assert not bad_L and not no_cm, f"coords L != L_new {bad_L[:5]}; no stored map {no_cm[:5]}"
    # every directory a copy sits in, plus its parents up to the fix-tree root (a new <mid3>/<hash> bucket adds an entry
    # to <mid3>); the processed/ root itself holds > 1024 bucket dirs (accepted by the user 2026-10-26) and is exempt
    touched = set()
    for d in dst.values():
        d = os.path.dirname(d)
        while d != fix:
            touched.add(d)
            d = os.path.dirname(d)
    full = [d for d in touched if len(os.listdir(d)) > 1024]
    assert not full, f"fix-tree directories above 1024 entries: {full[:5]}"
    write_cache([[dst[s], L_new[s], True] for s in pad], args.out_cache)
    with open(args.out_stems, "w") as fh:
        fh.write("".join(f"{s}\n" for s in sorted(pad)))
    print(f"padfix: {len(pad)} originals copied byte-identical into {len({os.path.dirname(d) for d in dst.values()})} "
          f"fix-tree buckets; wrote {args.out_stems}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status-tsv", required=True)
    ap.add_argument("--tree", required=True, help="processed/ root with <bucket>/<stem>.pt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out-cache")
    ap.add_argument("--out-stems")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate-dir")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmin", type=int, default=385)
    ap.add_argument("--lmax", type=int, default=512)
    ap.add_argument("--padfix", action="store_true")
    ap.add_argument("--padfix-list")
    ap.add_argument("--fix-tree", help="--padfix: the fix tree processed/ root the copies go to")
    args = ap.parse_args()
    args.tree = os.path.abspath(args.tree)
    args.gate_dir = args.gate_dir and os.path.abspath(args.gate_dir)
    if args.gate:
        assert args.gate_dir, "--gate needs --gate-dir"
        build_gate(args)
    elif args.padfix:
        assert args.padfix_list and args.fix_tree and args.out_cache and args.out_stems, \
            "--padfix needs --padfix-list, --fix-tree, --out-cache, --out-stems"
        args.fix_tree = os.path.abspath(args.fix_tree)
        build_padfix(args)
    else:
        assert args.out_cache and args.out_stems, "full mode needs --out-cache and --out-stems"
        build_full(args)


if __name__ == "__main__":
    main()
