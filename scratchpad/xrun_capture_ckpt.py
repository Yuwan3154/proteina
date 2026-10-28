"""Freeze the T8b EMA checkpoint(s) nearest a target step while the run trains (Engaging CPU; read-only on the run).

last-EMA.ckpt is rewritten every 50 optimizer steps (log.last_ckpt_every_n_steps 50) by an atomic os.replace, so a
reader never sees a partial file. Each new version is hard-linked into --out-dir first (stable inode). Polls its mtime every --poll-s seconds; reads each new version's global_step and,
iff |step - target| <= --half-window, copies it into --out-dir (re-reading the copy's step), renamed
last-EMA_step<step>.ckpt, chmod 0444, md5 recorded. Exits 0 once a kept copy exists on each side of the target
(or one exactly at it) and the step has passed target + half-window; exits 3 if the step passed that bound without a
copy on both sides (a missed rotation). Prints a heartbeat every --heartbeat-s seconds.
Usage: python scratchpad/xrun_capture_ckpt.py --ckpt-dir D --target S --out-dir O
"""

import argparse
import hashlib
import os
import shutil
import sys
import time

import torch


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--half-window", type=int, default=100)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--poll-s", type=float, default=15.0)  # vs a 50-step save every ~62 s (T8b ~2.9k steps/h, 10-28)
    ap.add_argument("--heartbeat-s", type=float, default=600.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    src = os.path.join(a.ckpt_dir, "last-EMA.ckpt")
    assert os.stat(a.out_dir).st_dev == os.stat(a.ckpt_dir).st_dev, "out-dir must share the ckpt filesystem (hard link)"
    last_mtime, last_beat, kept = None, 0.0, {}
    while True:
        m = os.path.getmtime(src)
        if m != last_mtime:
            # hard link = a stable inode: the writer only ever replaces the NAME, so the linked file never changes
            # (scratch is NFS: reading the live name across another client's os.replace can hit ESTALE)
            ln = os.path.join(a.out_dir, f".link_{os.getpid()}.ckpt")
            os.link(src, ln)
            step = int(torch.load(ln, map_location="cpu", weights_only=False, mmap=True)["global_step"])
            if abs(step - a.target) <= a.half_window and step not in kept:
                tmp = os.path.join(a.out_dir, f".copy_{os.getpid()}.ckpt")
                shutil.copyfile(ln, tmp)  # a real copy before chmod: chmod on the link would change the live file
                assert int(torch.load(tmp, map_location="cpu", weights_only=False, mmap=True)["global_step"]) == step
                dst = os.path.join(a.out_dir, f"last-EMA_step{step}.ckpt")
                os.replace(tmp, dst)
                os.chmod(dst, 0o444)
                kept[step] = md5(dst)
                print(f"[capture] KEPT step {step} -> {dst} md5 {kept[step]}", flush=True)
            os.remove(ln)
            last_mtime = m
            if step > a.target + a.half_window:
                lo = [s for s in kept if s <= a.target]
                hi = [s for s in kept if s >= a.target]
                print(f"[capture] passed {a.target}+{a.half_window} at step {step}; kept {sorted(kept)}", flush=True)
                with open(os.path.join(a.out_dir, "CAPTURED.tsv"), "w") as fh:
                    fh.write("".join(f"{s}\t{v}\n" for s, v in sorted(kept.items())))
                return 0 if (lo and hi) else 3
        if time.time() - last_beat >= a.heartbeat_s:
            print(f"[capture] {time.strftime('%F %T')} watching {src} (mtime {time.ctime(m)}); kept {sorted(kept)}",
                  flush=True)
            last_beat = time.time()
        time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
