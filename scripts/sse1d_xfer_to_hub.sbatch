#!/bin/bash
#SBATCH --job-name=sse1d_xfer
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/%x-%j.out
# Copies exactly the files the trace job (sse1d_trace_data_files) saw opened to SuperCloud:sse1d_data/<abs path>, for the
# A6000 to pull (SuperCloud = cross-node hub). mit_quicktest (15 min): rsync is cut at 13 min and the job resubmits
# itself; --partial --append-verify resumes the 65.5 GB pack. MODE=md5 instead writes the source md5s.
set -uo pipefail
: "${TRACE_JSON:?}"
OUT=/orcd/scratch/orcd/011/chenxiou/sse1d_xfer
mkdir -p "$OUT"
if [ ! -s "$OUT/files.txt" ]; then
  python3 - "$TRACE_JSON" "$OUT/files.txt" <<'PY'
import json, os, sys
r = json.load(open(sys.argv[1]))
fs = sorted((p for p in r["opened"] if p.startswith("/orcd/") and "/.venv/" not in p), key=os.path.getsize)
open(sys.argv[2], "w").write("".join(p[1:] + "\n" for p in fs))  # small first, the pack last
print(len(fs), "files", sum(map(os.path.getsize, fs)) / 1e9, "GB")
PY
  rc=$?; [ $rc -eq 0 ] || exit $rc
fi
if [ "${MODE:-copy}" = md5 ]; then
  (cd / && xargs -a "$OUT/files.txt" md5sum) > "$OUT/md5.txt"
  rc=$?; echo "md5 rc=$rc"; date; exit $rc
fi
date
timeout 780 rsync -a --partial --append-verify --files-from="$OUT/files.txt" / SuperCloud:sse1d_data/
rc=$?
echo "rsync rc=$rc"; date
if [ $rc -eq 124 ]; then
  sbatch --export=ALL,TRACE_JSON="$TRACE_JSON" "$OUT/xfer.sbatch"  # $OUT/xfer.sbatch = a copy of this file
  rc=$?; echo "resubmit rc=$rc"; exit $rc
fi
[ $rc -eq 0 ] && touch "$OUT/COPY_DONE"
exit $rc
