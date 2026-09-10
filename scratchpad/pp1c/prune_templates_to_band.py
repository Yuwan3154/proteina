"""Repack the generated template npz keeping only the templates inside the TM band.

Generation wrote all 64 rungs per chain (103 GB), but training only ever samples the ones inside
`min_tm..max_tm` -- roughly 56% of them. The rest are dead weight on any host that has to hold a
copy, which matters because the training box has far less free disk than SuperCloud.

⛔ This writes a SEPARATE tree and never touches the source. The full 64-rung set stays on
SuperCloud, so widening the band later is a re-prune, not a regeneration.

⭐ The index stays rectangular (n_chain, 64) so every existing analysis script keeps working. The
only addition is `slot`, mapping each original rung to its row in the pruned npz (-1 = dropped);
`SyntheticTemplatePool` uses it to translate a pick before indexing coords.
"""

import argparse
import zlib
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--index", required=True)
p.add_argument("--src-root", required=True)
p.add_argument("--dst-root", required=True)
p.add_argument("--out-index", required=True)
p.add_argument("--min-tm", type=float, default=0.3)
p.add_argument("--max-tm", type=float, default=0.9)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--num-shards", type=int, default=1)
# ⛔ Without this every shard would np.savez the SAME --out-index path concurrently. Copying shards
# now skip the index entirely and one final invocation writes it after they all finish.
p.add_argument("--index-only", action="store_true",
               help="write --out-index and exit; copy nothing")
a = p.parse_args()

z = np.load(a.index, allow_pickle=False)
chains = [str(c) for c in z["chains"]]
tm, rewind = z["tm"], z["rewind"]
band = (tm > a.min_tm) & (tm < a.max_tm)

src_root, dst_root = Path(a.src_root), Path(a.dst_root)
# computed for EVERY chain, not just this shard's, so each shard writes an identical complete
# index instead of a partial one that the next shard would clobber
slot = np.full(tm.shape, -1, np.int16)
for i in range(len(chains)):
    slot[i, np.flatnonzero(band[i])] = np.arange(int(band[i].sum()), dtype=np.int16)

if a.index_only:
    np.savez(
        a.out_index,
        chains=np.array(chains), tm=tm, rewind=rewind, length=z["length"],
        slot=slot, min_tm=np.float32(a.min_tm), max_tm=np.float32(a.max_tm),
    )
    kept = (slot >= 0).sum()
    print(f"wrote {a.out_index}: {len(chains)} chains, {kept:,}/{slot.size:,} rungs retained "
          f"({100*kept/slot.size:.1f}%), band {a.min_tm}-{a.max_tm}")
    raise SystemExit(0)

mine = list(range(a.shard, len(chains), a.num_shards))
n_bytes_in = n_bytes_out = 0
n_empty = 0

for n, i in enumerate(mine):
    chain = chains[i]
    keep = np.flatnonzero(band[i])
    if len(keep) == 0:
        n_empty += 1
        continue
    # sharding must match generate_templates.py exactly: crc32, never builtin hash()
    sub = f"shard{zlib.crc32(chain.encode()) % 1000:04d}"
    src = src_root / sub / f"{chain}.npz"
    dst_dir = dst_root / sub
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{chain}.npz"
    d = np.load(src, allow_pickle=False)
    n_bytes_in += src.stat().st_size
    np.savez(
        dst,
        coords=d["coords"][keep],
        atom_mask=d["atom_mask"],
        aatype=d["aatype"],
        residue_index=d["residue_index"],
        rewind_steps=d["rewind_steps"][keep],
    )
    n_bytes_out += dst.stat().st_size
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(mine)}  "
              f"{n_bytes_in/2**30:.1f} -> {n_bytes_out/2**30:.1f} GiB", flush=True)

print(f"shard {a.shard}: {n_bytes_in/2**30:.1f} -> {n_bytes_out/2**30:.1f} GiB "
      f"({100*n_bytes_out/max(n_bytes_in,1):.0f}%), {n_empty} chains had nothing in band")
