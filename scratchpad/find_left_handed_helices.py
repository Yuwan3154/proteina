"""Find training chains carrying NATURAL left-handed alpha-helical segments.

Motivation: the c2c model has been pushed hard toward native (right-handed) folds by the mirror
work. A contact map fixes the global fold only up to reflection, but a LOCAL left-handed segment
inside an otherwise right-handed chain IS determined by the map -- reflecting just that segment
would break its contacts. So a native alpha-L segment is a clean probe of whether the model has
learned "reproduce the handedness the map implies" or merely "make everything right-handed".

Detector: the SAME CA pseudo-dihedral statistic the mirror detector uses
(proteinfoundation/utils/c2c_dump.py::_ca_dihedrals), so nothing new is invented. Native chains sit
near 0.08-0.12 POSITIVE fraction in the helical range; a left-handed segment is a RUN of
consecutive POSITIVE dihedrals in that range.

⛔ NO THRESHOLD IS INVENTED. This reports the full distribution of longest-positive-run lengths and
ranks chains by it; selection is "take the extremes of the measured distribution", not "everything
above a number I chose".

⛔ MASKS, NOT SHAPES. coord_mask gaps are real chain breaks; dihedrals are computed only within
contiguous resolved stretches, or a break would fabricate a spurious angle.

⛔ Writes JSONL incrementally so a wall-clock kill still leaves usable results.
"""

import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch

CA = 1  # atom37 index
HELICAL_LO, HELICAL_HI = 30.0, 90.0  # the mirror detector's helical window, unchanged


def ca_dihedrals(ca):
    """CA pseudo-dihedral over every consecutive quadruple, degrees. Mirrors c2c_dump._ca_dihedrals."""
    b0, b1, b2 = ca[1:-2] - ca[:-3], ca[2:-1] - ca[1:-2], ca[3:] - ca[2:-1]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    return np.degrees(np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1)))


def contiguous_runs(mask):
    """[(start, stop)] index ranges of contiguous True in a boolean array."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def longest_positive_run(d):
    """Longest run of consecutive helical-range dihedrals that are POSITIVE (left-handed).

    Returns (run_length, start_index_into_d). Only dihedrals inside the helical window count; a
    dihedral outside the window BREAKS the run (it is not helical, so the run is not a helix).
    """
    best, best_i, cur, cur_i = 0, -1, 0, -1
    for i, v in enumerate(d):
        helical = HELICAL_LO < abs(v) < HELICAL_HI
        if helical and v > 0:
            if cur == 0:
                cur_i = i
            cur += 1
            if cur > best:
                best, best_i = cur, cur_i
        else:
            cur = 0
    return best, best_i


def scan_one(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    coords = obj["coords"] if hasattr(obj, "keys") else obj.coords
    ca = np.asarray(coords[:, CA, :], dtype=np.float64)
    if hasattr(obj, "keys") and "coord_mask" in obj:
        cm = obj["coord_mask"]
    else:
        cm = getattr(obj, "coord_mask", None)
    if cm is None:
        resolved = np.isfinite(ca).all(-1)
    else:
        cm = np.asarray(cm)
        resolved = cm[:, CA].astype(bool) if cm.ndim == 2 else cm.astype(bool)
        resolved &= np.isfinite(ca).all(-1)

    best, best_abs, n_pos, n_hel = 0, -1, 0, 0
    for s, e in contiguous_runs(resolved):
        if e - s < 8:
            continue
        d = ca_dihedrals(ca[s:e])
        if len(d) == 0:
            continue
        hel = (np.abs(d) > HELICAL_LO) & (np.abs(d) < HELICAL_HI)
        n_hel += int(hel.sum())
        n_pos += int((d[hel] > 0).sum())
        r, i = longest_positive_run(d)
        if r > best:
            best, best_abs = r, s + i
    cid = obj["id"] if hasattr(obj, "keys") and "id" in obj else getattr(obj, "id", os.path.basename(path))
    return {
        "id": str(cid),
        "path": path,
        "L": int(resolved.sum()),
        "longest_L_run": int(best),
        "run_start_ca": int(best_abs),
        "helix_pos_frac": (n_pos / n_hel) if n_hel else None,
        "n_helical": int(n_hel),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000, help="chains to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_minutes", type=float, default=40.0)
    args = ap.parse_args()

    shards = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    print(f"[scan] {len(shards)} shards under {args.root}", flush=True)
    files = []
    for sh in shards:
        files.extend(glob.glob(os.path.join(sh, "*.pt")))
    print(f"[scan] {len(files)} .pt chains total", flush=True)
    rng = random.Random(args.seed)
    if args.n and args.n < len(files):
        files = rng.sample(files, args.n)
        print(f"[scan] sampled {len(files)} (seed={args.seed})", flush=True)

    t0 = time.time()
    done = skipped = 0
    with open(args.out, "w") as fh:
        for p in files:
            if (time.time() - t0) / 60.0 > args.max_minutes:
                print(f"[scan] wall-clock budget hit after {done}", flush=True)
                break
            rec = scan_one(p)
            if rec is None:
                skipped += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            if done % 500 == 0:
                print(f"[scan] {done} chains, {time.time()-t0:.0f}s", flush=True)
    # ⛔ Every skip is reported, not silently swallowed.
    print(f"[scan] DONE scanned={done} skipped={skipped} elapsed={time.time()-t0:.0f}s -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
