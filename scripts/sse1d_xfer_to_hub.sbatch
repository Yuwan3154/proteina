#!/bin/bash
#SBATCH --job-name=sse1d_xfer
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/%x-%j.out
# Copies exactly the files the trace job (sse1d_trace_data_files) saw opened to SuperCloud:sse1d_data/<abs path>,
# for the A6000 to pull (SuperCloud = cross-node hub). Small files first, the 65.5 GB pack last; md5s for the A6000 check.
set -uo pipefail
: "${TRACE_JSON:?}"
OUT=/orcd/scratch/orcd/011/chenxiou/sse1d_xfer
mkdir -p "$OUT"
python3 - "$TRACE_JSON" "$OUT" <<'PY'
import json, os, sys
r = json.load(open(sys.argv[1]))
fs = [p for p in r["opened"] if p.startswith("/orcd/") and "/.venv/" not in p]
big = [p for p in fs if os.path.getsize(p) > 1e9]
small = [p for p in fs if p not in big]
open(os.path.join(sys.argv[2], "small.txt"), "w").write("".join(p[1:] + "\n" for p in small))
open(os.path.join(sys.argv[2], "big.txt"), "w").write("".join(p[1:] + "\n" for p in big))
print(len(small), "small", len(big), "big", sum(map(os.path.getsize, fs)) / 1e9, "GB")
PY
rc=$?; [ $rc -eq 0 ] || exit $rc
date; rsync -a --files-from="$OUT/small.txt" / SuperCloud:sse1d_data/
rc=$?; echo "small rsync rc=$rc"; [ $rc -eq 0 ] || exit $rc
date; rsync -a --partial --files-from="$OUT/big.txt" / SuperCloud:sse1d_data/
rc=$?; echo "big rsync rc=$rc"; [ $rc -eq 0 ] || exit $rc
date; (cd / && cat "$OUT/small.txt" "$OUT/big.txt" | xargs md5sum) > "$OUT/md5.txt"
rc=$?; echo "md5 rc=$rc"; date
exit $rc
