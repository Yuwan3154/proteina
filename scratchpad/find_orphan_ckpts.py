"""Which checkpoint files has Lightning FORGOTTEN about? (report-only by default)

⛔⛔ THE LEAK. `save_top_k=3` is enforced per ModelCheckpoint INSTANCE, and a chained run builds a
fresh instance every segment. Lightning's `check_monitor_top_k` returns True unconditionally while
`best_k_models` holds fewer than `save_top_k` entries, so the first 3 saves after ANY resume enter
top-k regardless of their value. Files from previous segments are not in the new instance's tracking
set, so nothing ever evicts them.

Measured on c2c_cb8_tbeta 2026-09-18: **7 checkpoint files where the policy allows 4**
(save_top_k 3 + last), i.e. 3 orphans = ~9.7 GiB, with the three tracked ones 10 steps apart because
of the tightened cadence. Every segment boundary adds up to 3 more.

⭐ THE PRINCIPLED TEST, not a heuristic: a file is an orphan iff it is NOT `last.ckpt` and NOT
referenced by last.ckpt's OWN ModelCheckpoint callback state (`best_k_models` / `best_model_path`).
That is exactly "files Lightning itself no longer tracks" -- no step threshold, no age cutoff, no
invented margin.

⛔ REPORT-ONLY unless --delete is passed. Checkpoints are 3.2 GB each and regenerable only by
re-training, so deletion is not a reversible decision and is not taken on this tool's own authority.

Usage: find_orphan_ckpts.py <run> [<run> ...] [--delete]
"""

import argparse
import glob
import os
import sys

import torch

STORE = "/orcd/scratch/orcd/011/chenxiou/c2c_store"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--delete", action="store_true",
                help="actually remove the orphans (default: report only)")
args = ap.parse_args()

total_orphan_gb = 0.0
grand = []

for run in args.runs:
    d = os.path.join(STORE, run)
    last = os.path.join(d, "last.ckpt")
    files = sorted(glob.glob(os.path.join(d, "*.ckpt")))
    if not files:
        print(f"\n{run}: no checkpoints")
        continue
    print(f"\n=== {run}: {len(files)} .ckpt files ===")
    if not os.path.exists(last):
        print("  ⛔ no last.ckpt -- cannot determine what Lightning tracks. SKIPPING (refusing to "
              "guess which files are live).")
        continue

    ck = torch.load(last, map_location="cpu", weights_only=False)
    tracked = set()
    cbs = ck.get("callbacks", {}) or {}
    for key, st in cbs.items():
        if "ModelCheckpoint" not in str(key) or not isinstance(st, dict):
            continue
        for p in (st.get("best_k_models") or {}):
            tracked.add(os.path.realpath(p))
        bmp = st.get("best_model_path")
        if bmp:
            tracked.add(os.path.realpath(bmp))
        lmp = st.get("last_model_path")
        if lmp:
            tracked.add(os.path.realpath(lmp))
    print(f"  last.ckpt is at global_step {ck.get('global_step')}; it tracks {len(tracked)} file(s)")
    # ⛔ If the callback state is absent or empty, EVERY file would look orphaned. Refuse rather
    # than report a wipe -- an empty tracking set is a failure to read it, not evidence of orphans.
    if not tracked:
        print("  ⛔ tracking set is EMPTY -- refusing to call anything an orphan. Either the "
              "checkpoint has no ModelCheckpoint callback state, or the key layout changed.")
        continue

    for f in files:
        rp = os.path.realpath(f)
        gb = os.path.getsize(f) / 1024**3
        if os.path.basename(f) == "last.ckpt":
            print(f"  KEEP    {os.path.basename(f):<34} {gb:5.1f} GiB  (resume anchor)")
        elif rp in tracked:
            print(f"  KEEP    {os.path.basename(f):<34} {gb:5.1f} GiB  (tracked by Lightning)")
        else:
            print(f"  ORPHAN  {os.path.basename(f):<34} {gb:5.1f} GiB  (untracked)")
            total_orphan_gb += gb
            grand.append(f)

print(f"\n=== {len(grand)} orphan(s), {total_orphan_gb:.1f} GiB total ===")
if not grand:
    print("nothing to do")
    sys.exit(0)
if args.delete:
    for f in grand:
        os.remove(f)
        print(f"  deleted {f}")
    print(f"⭐ reclaimed {total_orphan_gb:.1f} GiB")
else:
    print("REPORT ONLY -- pass --delete to remove these. Each is 3.2 GB and regenerable only by")
    print("re-training, so this tool does not delete on its own authority.")
