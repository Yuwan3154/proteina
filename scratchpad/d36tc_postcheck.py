"""D36tc post-check of ONE tri pass (run AFTER sampling; non-zero exit fails the job).

  1. the job log (stdout + stderr merged) has no unconditioned-sample warning (BAD) and DOES contain the loguru
     positive-control lines (NEED, + NEED_MAP with --map);
  2. the checkpoint load printed `missing=0 unexpected=0` (tri_gen_eval's [EMA load] line);
  3. samples.jsonl holds exactly --n per listed stem, each L == the processed .pt residue count (coords.shape[0],
     not the mask: a crop shrinks both) and ref_id == map[stem] (template arm) or == stem (self arm).
"""
import argparse
import collections
import json
import os
import re

import torch

BAD = ("absent from the topology index", "has no entry for", "samples unconditioned", "sampling UNCONDITIONED")
# Positive controls: loguru lines that MUST be present, so a log missing the loguru stream (stderr) cannot pass
# the BAD scan vacuously (model_trainer_base.py:3368 and :2840).
NEED = ("validation_sampling: fixed chain set loaded from",)
NEED_MAP = ("validation_sampling: topology_reference_map loaded from",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--processed-dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--map", default=None)
    args = ap.parse_args()

    log = open(args.log).read()
    fails = [f"log: {ln.strip()[:200]}" for ln in log.splitlines() if any(b in ln for b in BAD)]
    for need in NEED + (NEED_MAP if args.map else ()):
        if need not in log:
            fails.append(f"log lacks positive control {need!r} (loguru stream missing?)")
    loads = re.findall(r"\[EMA load\] missing=(\d+) unexpected=(\d+)", log)
    if loads != [("0", "0")]:
        fails.append(f"checkpoint load lines {loads} (want exactly one missing=0 unexpected=0)")

    stems = sorted({ln.split()[0] for ln in open(args.chains) if ln.strip()})
    m = json.load(open(args.map)) if args.map else None
    rows = [json.loads(ln) for ln in open(os.path.join(args.dump, "samples.jsonl")) if ln.strip()]
    per = collections.Counter(r["stem"] for r in rows)
    fails += [f"{st}: {per[st]} samples, want {args.n}" for st in stems if per[st] != args.n]
    fails += [f"{st}: sampled but not listed" for st in per if st not in stems]
    L_pt = {st: int(torch.load(os.path.join(args.processed_dir, f"{st}.pt"), map_location="cpu",
                               weights_only=False).coords.shape[0]) for st in stems}
    for r in rows:
        st = r["stem"]
        if st in L_pt and int(r["L"]) != L_pt[st]:
            fails.append(f"{st} s{r['sample_index']}: L {r['L']} != .pt length {L_pt[st]} (crop)")
        want = m[st] if m is not None else st
        if r.get("ref_id") != want:
            fails.append(f"{st} s{r['sample_index']}: ref_id {r.get('ref_id')!r} != {want!r}")
    print(f"[postcheck] {len(rows)} samples, {len(stems)} stems, n={args.n}, map={args.map}: {len(fails)} failures")
    for f in fails[:50]:
        print(f"  FAIL {f}")
    assert not fails, "post-check failed"
    print("D36TC_POSTCHECK_OK")


if __name__ == "__main__":
    main()
