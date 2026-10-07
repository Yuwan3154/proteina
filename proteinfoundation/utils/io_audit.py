"""Env-gated record of every file a process opens (user 2026-10-07: prove nothing per-step still reads NFS).

With IO_AUDIT_DIR set, install() adds a sys audit hook that appends each distinct (event, path) to
IO_AUDIT_DIR/<tag>_<pid>.tsv with seconds since install. Off (a no-op) when the variable is unset.
"""

import os
import sys
import time

_seen = set()


def install(tag):
    d = os.environ.get("IO_AUDIT_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    fh = open(os.path.join(d, f"{tag}_{os.getpid()}.tsv"), "a", buffering=1)  # opened BEFORE the hook: writes don't re-enter
    t0 = time.time()

    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "mmap.__new__", "glob.glob") and args:
            p = args[0]
            if isinstance(p, int) or p is None:
                return
            p = os.fspath(p) if not isinstance(p, (str, bytes)) else p
            p = p.decode(errors="replace") if isinstance(p, bytes) else p
            key = (event, p)
            if key not in _seen:
                _seen.add(key)
                fh.write(f"{time.time() - t0:.1f}\t{event}\t{p}\n")

    sys.addaudithook(hook)
