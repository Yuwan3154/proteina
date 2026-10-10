"""Read-only size estimate for a max512 pack (T8): total bytes of the processed .pt of every listed chain.
Usage: python t8_pack_size.py DATA_DIR(pdb_train) LIST [LIST ...]
"""

import json
import pathlib
import sys

from proteinfoundation.datasets.pdb_data import _processed_path_sharded


def main(data_dir, lists):
    d = pathlib.Path(data_dir)
    man = json.load(open(d / "shard_manifest.json"))
    stems = sorted({l.split()[0] for f in lists for l in open(f) if l.strip()})  # same parsing as build_pack.py
    tot, miss = 0, []
    for s in stems:
        p = _processed_path_sharded(d / "processed", s, man)
        if p.exists():
            tot += p.stat().st_size
        else:
            miss.append(s)
    assert stems, "no stems read"
    assert not miss, f"{len(miss)}/{len(stems)} stems have no .pt, e.g. {miss[:10]}"  # build_pack.py requires all
    print(f"stems {len(stems)}, total {tot / 1e9:.2f} GB, mean {tot / len(stems) / 1e3:.0f} kB per stem")


if __name__ == "__main__":
    assert len(sys.argv) >= 3, __doc__
    main(sys.argv[1], sys.argv[2:])
