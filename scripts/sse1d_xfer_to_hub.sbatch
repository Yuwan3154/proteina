#!/bin/bash
#SBATCH --job-name=sse1d_xfer
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/%x-%j.out
# Copies exactly the files the trace job (sse1d_trace_data_files) saw opened to the SuperCloud hub, for the A6000 to pull.
# Small files: rsync to SuperCloud:sse1d_data/<abs path>. Files > 1 GB (pack, topology index): 512 MiB chunks over 8
# parallel ssh streams to SuperCloud:sse1d_chunks/<basename>.<k> (rsync ran at 2-3 MB/s, one plain ssh stream ~100 MB/s);
# a chunk is renamed from .tmp only when complete. mit_quicktest: cut at 12.5 min, then the job resubmits itself
# ($OUT/xfer.sbatch = a copy of this file). MODE=md5 instead writes the source md5s.
set -uo pipefail
: "${TRACE_JSON:?}"
OUT=/orcd/scratch/orcd/011/chenxiou/sse1d_xfer
CH=512  # MiB per chunk
mkdir -p "$OUT"
if [ ! -s "$OUT/files.txt" ]; then
  python3 - "$TRACE_JSON" "$OUT/files.txt" <<'PY'
import json, os, sys
r = json.load(open(sys.argv[1]))
fs = sorted((p for p in r["opened"] if p.startswith("/orcd/") and "/.venv/" not in p), key=os.path.getsize)
open(sys.argv[2], "w").write("".join(p[1:] + "\n" for p in fs))
print(len(fs), "files", sum(map(os.path.getsize, fs)) / 1e9, "GB")
PY
  rc=$?; [ $rc -eq 0 ] || exit $rc
fi
if [ "${MODE:-copy}" = md5 ]; then
  (cd / && xargs -a "$OUT/files.txt" md5sum) > "$OUT/md5.txt"
  rc=$?; echo "md5 rc=$rc"; date; exit $rc
fi
date
python3 - "$OUT/files.txt" "$OUT" "$CH" <<'PY'
import os, sys
small, chunks, ch = [], [], int(sys.argv[3]) << 20
names = set()
for ln in open(sys.argv[1]):
    p = "/" + ln.strip()
    s = os.path.getsize(p)
    if s > 1e9:
        assert os.path.basename(p) not in names, p  # chunk names use the basename
        names.add(os.path.basename(p))
        chunks += [f"{p} {k} {min(ch, s - k * ch)}" for k in range(-(-s // ch))]
    else:
        small.append(ln.strip())
open(os.path.join(sys.argv[2], "small.txt"), "w").write("".join(x + "\n" for x in small))
open(os.path.join(sys.argv[2], "chunks.txt"), "w").write("".join(x + "\n" for x in chunks))
print(len(small), "small files,", len(chunks), "chunks")
PY
rc=$?; [ $rc -eq 0 ] || exit $rc
ATTEMPT=${ATTEMPT:-1}
resubmit() {
  if [ "$ATTEMPT" -ge 12 ]; then echo "giving up after $ATTEMPT attempts"; exit 5; fi
  sbatch --export=ALL,TRACE_JSON="$TRACE_JSON",ATTEMPT=$((ATTEMPT + 1)) "$OUT/xfer.sbatch"  # = a copy of this file
  rc=$?; echo "resubmit rc=$rc"; exit $rc
}
timeout 120 rsync -a --files-from="$OUT/small.txt" / SuperCloud:sse1d_data/
rc=$?; echo "small rsync rc=$rc"; [ $rc -eq 0 ] || resubmit
# a chunk counts as sent only if its size is right (a stream cut at the timeout can still rename a short .tmp)
ssh -o BatchMode=yes SuperCloud 'mkdir -p sse1d_chunks && cd sse1d_chunks && for f in *; do [ -e "$f" ] && stat -c "%n %s" "$f"; done; true' > "$OUT/have.txt"
rc=$?; [ $rc -eq 0 ] || resubmit
send() {  # $1 path, $2 chunk index, $3 expected bytes
  set -o pipefail
  local b; b=$(basename "$1")
  dd if="$1" bs=1M skip=$(( $2 * CH )) count=$CH iflag=fullblock status=none \
    | ssh -o BatchMode=yes SuperCloud "cat > sse1d_chunks/$b.$2.tmp && [ \$(stat -c %s sse1d_chunks/$b.$2.tmp) = $3 ] && mv sse1d_chunks/$b.$2.tmp sse1d_chunks/$b.$2"
}
export -f send; export CH
while read -r p k sz; do grep -qxF "$(basename "$p").$k $sz" "$OUT/have.txt" || echo "$p $k $sz"; done < "$OUT/chunks.txt" > "$OUT/todo.txt"
echo "attempt $ATTEMPT: todo $(wc -l < "$OUT/todo.txt") of $(wc -l < "$OUT/chunks.txt") chunks"; date
[ -s "$OUT/todo.txt" ] || { touch "$OUT/COPY_DONE"; echo COPY_DONE; exit 0; }
timeout 600 xargs -r -a "$OUT/todo.txt" -P 8 -n 3 bash -c 'send "$0" "$1" "$2"'
rc=$?; echo "chunks rc=$rc"; date
resubmit
