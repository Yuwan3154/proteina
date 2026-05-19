"""One-shot cleanup for orphan EMA checkpoint companions.

Background
----------
`EmaModelCheckpoint._save_checkpoint` writes TWO files per checkpoint:
- ``<filepath>.ckpt`` (regular weights + optimizer state with EMA inside)
- ``<filepath>-EMA.ckpt`` (model state with EMA weights swapped in)

Lightning's ``ModelCheckpoint`` rotates the top-k by deleting the OLD regular
``.ckpt`` when a new best comes in, but the base ``_remove_checkpoint`` method
doesn't know about our ``-EMA.ckpt`` companion. We fixed this for future runs
by overriding ``EmaModelCheckpoint._remove_checkpoint`` (see
``proteinfoundation/utils/ema_utils/ema_callback.py``).

This script cleans up the orphans left behind by PAST runs: any
``*-EMA.ckpt`` whose companion ``*.ckpt`` no longer exists. Always runs in
``--dry-run`` mode first (lists targets without deleting); pass
``--apply`` to actually delete.

Usage:
    python scripts/cleanup_orphan_ema_checkpoints.py /path/to/checkpoints_dir
    python scripts/cleanup_orphan_ema_checkpoints.py /path/to/checkpoints_dir --apply

The ``last-EMA.ckpt`` file is always paired with ``last.ckpt`` and is never
treated as orphan — Lightning's ``save_last`` mechanism is independent of
top-k.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_orphan_ema_checkpoints(directory: Path) -> list[Path]:
    """Return a list of ``-EMA.ckpt`` files whose regular ``.ckpt`` companion is missing."""
    orphans: list[Path] = []
    for ema_ckpt in sorted(directory.glob("*-EMA.ckpt")):
        # Reconstruct the regular companion filename: replace "-EMA.ckpt" with ".ckpt".
        stem_ema = ema_ckpt.name
        if not stem_ema.endswith("-EMA.ckpt"):
            continue
        regular_name = stem_ema[: -len("-EMA.ckpt")] + ".ckpt"
        regular_path = ema_ckpt.parent / regular_name
        # "last-EMA.ckpt" pairs with "last.ckpt" — never orphan.
        if regular_name == "last.ckpt":
            continue
        if not regular_path.exists():
            orphans.append(ema_ckpt)
    return orphans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Path to a Lightning checkpoint directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the orphan files (default: dry-run, just list)",
    )
    args = parser.parse_args()

    directory: Path = args.directory
    if not directory.is_dir():
        print(f"ERROR: {directory} is not a directory.")
        return 2

    orphans = find_orphan_ema_checkpoints(directory)
    if not orphans:
        print(f"No orphan EMA checkpoints in {directory}. Nothing to clean up.")
        return 0

    total_bytes = sum(p.stat().st_size for p in orphans)
    print(f"Found {len(orphans)} orphan EMA checkpoint(s) totalling "
          f"{total_bytes / 1e9:.2f} GB in {directory}:")
    for p in orphans:
        sz_gb = p.stat().st_size / 1e9
        print(f"  {p.name:80s}  {sz_gb:6.2f} GB")

    if not args.apply:
        print("\n[dry-run] Re-run with --apply to delete these files.")
        return 0

    print("\nDeleting…")
    failed = []
    for p in orphans:
        try:
            os.remove(p)
            print(f"  deleted {p.name}")
        except OSError as e:
            failed.append((p, str(e)))
            print(f"  FAILED {p.name}: {e}")

    if failed:
        print(f"\n{len(failed)} files failed to delete.")
        return 1
    print(f"\nDone. Reclaimed {total_bytes / 1e9:.2f} GB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
