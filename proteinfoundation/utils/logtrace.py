"""Per-rank tracing of ``self.log`` calls, for diagnosing DDP collective mismatches.

A ``self.log(..., sync_dist=True)`` that only SOME ranks execute deadlocks DDP: Lightning's
``_sync_ddp`` issues ``barrier()`` then ``all_reduce()``, so a rank that skips the call strands
the others in the barrier. Diffing two ranks' traces names the offending metric directly.

Deliberately free of torch/lightning imports so it can be unit tested on its own.

Handles are cached per rank and opened **line-buffered**: the failure being diagnosed kills the
process, so anything sitting in a userspace buffer would be lost -- and the tail is precisely
the interesting part. Line buffering pushes each line to the OS on the newline, so a SIGKILL
still leaves every completed line on disk.
"""

import os
from typing import Dict, TextIO

_HANDLES: Dict[int, TextIO] = {}


def enabled() -> bool:
    return os.environ.get("LOGTRACE") == "1"


def trace(rank: int, name: str, sync_dist: bool, dirpath: str = "logs") -> bool:
    """Append one ``name<TAB>sync_dist=...`` line for ``rank``. Returns True if written."""
    if not enabled():
        return False
    fh = _HANDLES.get(rank)
    if fh is None or fh.closed:
        os.makedirs(dirpath, exist_ok=True)
        fh = open(os.path.join(dirpath, f"logtrace_rank{rank}.txt"), "a", buffering=1)
        _HANDLES[rank] = fh
    fh.write(f"{name}\tsync_dist={bool(sync_dist)}\n")
    return True


def reset() -> None:
    """Close and forget all handles. For tests and for a clean re-arm."""
    for fh in _HANDLES.values():
        if not fh.closed:
            fh.close()
    _HANDLES.clear()
