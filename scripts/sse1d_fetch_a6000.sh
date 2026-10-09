#!/bin/bash
# A6000 side of the data copy: pull the hub copy (scripts/sse1d_xfer_to_hub.sbatch) into ~/sse1d_data/<abs path>,
# reassemble the chunked big files (each chunk size-checked; written to .tmp then renamed), and check every file
# against the Engaging md5s.
#   scripts/sse1d_fetch_a6000.sh <md5.txt> <chunks.txt>   (both from Engaging sse1d_xfer/)
set -uo pipefail
MD5=$1; CHUNKS=$2
ROOT=$HOME/sse1d_data; CHD=$HOME/sse1d_chunks_dl
mkdir -p "$ROOT" "$CHD"
date; rsync -a --exclude tri_v2.pack --exclude topology_index_cb8_synth.pt SuperCloud:sse1d_data/ "$ROOT/"
rc=$?; echo "small rsync rc=$rc"; [ $rc -eq 0 ] || exit $rc
get() {  # $1 chunk name, $2 expected bytes
  [ "$(stat -c %s "$CHD/$1" 2>/dev/null)" = "$2" ] && return 0
  ssh -o BatchMode=yes SuperCloud "cat sse1d_chunks/$1" > "$CHD/$1.tmp" && [ "$(stat -c %s "$CHD/$1.tmp")" = "$2" ] \
    && mv "$CHD/$1.tmp" "$CHD/$1"
}
export -f get; export CHD
while read -r p k sz; do echo "$(basename "$p").$k $sz"; done < "$CHUNKS" | xargs -r -P 8 -n 2 bash -c 'get "$0" "$1"'
rc=$?; echo "chunk pull rc=$rc"; date; [ $rc -eq 0 ] || exit $rc
for p in $(awk '{print $1}' "$CHUNKS" | uniq); do
  rel=${p#/}; b=$(basename "$p"); n=$(grep -c "^$p " "$CHUNKS")
  mkdir -p "$ROOT/$(dirname "$rel")"
  for k in $(seq 0 $((n - 1))); do cat "$CHD/$b.$k" || exit 4; done > "$ROOT/$rel.tmp" && mv "$ROOT/$rel.tmp" "$ROOT/$rel"
  echo "assembled $rel from $n chunks"
done
date; (cd "$ROOT" && md5sum -c --quiet "$MD5"); rc=$?
echo "md5 check rc=$rc ($(wc -l < "$MD5") files)"; date
exit $rc
