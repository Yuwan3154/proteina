#!/bin/bash
# A6000 inputs for the v5 index: the two Engaging tars from the SuperCloud hub (sha256 checked against the source-side
# sums) and the 52,864 T2 template npz from SuperCloud ~/pp1c_work/templates_band (byte-exact copy of the original;
# 4 tar streams; file count checked; the 400-file sha256 sample of the ORIGINAL Engaging tree checked where listed).
#   bash scripts/synth_index_v5_pro_fetch_a6000.sh <hash_engaging_templates_band.txt>
set -uo pipefail
SAMPLE=$(realpath "$1")
D=$HOME/sse1d_idx; mkdir -p "$D/in" "$D/t2_templates_band" "$D/fetch"
[ -z "$(ls -A "$D/t2_templates_band")" ] || { echo "FATAL: $D/t2_templates_band is not empty"; exit 9; }
cd "$D/fetch" || exit 2
for t in engaging_trees meta; do
  ssh -n -o BatchMode=yes SuperCloud "cat sse1d_idx_in/$t.sha" > "$t.sha.src" || exit 3
  ssh -n -o BatchMode=yes SuperCloud "cat sse1d_idx_in/$t.tar" | tee >(sha256sum > "$t.sha.dst") | tar xf - -C "$D/in"
  rc=$?; sleep 3
  echo "$t: rc=$rc src $(cut -c1-16 "$t.sha.src") dst $(cut -c1-16 "$t.sha.dst")"
  [ $rc -eq 0 ] && [ "$(cut -d' ' -f1 "$t.sha.src")" = "$(cut -d' ' -f1 "$t.sha.dst")" ] || { echo "FAIL $t"; exit 4; }
done
ssh -n -o BatchMode=yes SuperCloud 'cat sse1d_idx_in/t2_list.txt' > t2_list.txt || exit 5
for k in 0 1 2 3; do awk -v k=$k 'NR % 4 == k' t2_list.txt > "t2_list.$k"; scp -q -o BatchMode=yes "t2_list.$k" SuperCloud:sse1d_idx_in/ || exit 6; done
for k in 0 1 2 3; do
  ( ssh -n -o BatchMode=yes SuperCloud "cd ~/pp1c_work/templates_band && tar cf - -T ~/sse1d_idx_in/t2_list.$k" \
      | tar xf - -C "$D/t2_templates_band"; echo "${PIPESTATUS[0]} ${PIPESTATUS[1]}" > "stream_$k.rc" ) &
done
wait
for k in 0 1 2 3; do echo "stream $k rc: $(cat "stream_$k.rc")"; [ "$(cat "stream_$k.rc")" = "0 0" ] || { echo "FAIL stream $k"; exit 10; }; done
n=$(find "$D/t2_templates_band" -name '*.npz' | wc -l)
echo "t2 npz: $n of $(wc -l < t2_list.txt)"
[ "$n" -eq "$(wc -l < t2_list.txt)" ] || { echo "FAIL t2 count"; exit 7; }
grep -F -f <(sed 's/^/  /' t2_list.txt) "$SAMPLE" > sample_in_list.txt
(cd "$D/t2_templates_band" && sha256sum -c --quiet "$D/fetch/sample_in_list.txt"); rc=$?
echo "original-tree sha sample: $(wc -l < sample_in_list.txt) of $(wc -l < "$SAMPLE") in our list, check rc=$rc"
[ $rc -eq 0 ] && [ -s sample_in_list.txt ] || exit 8
echo FETCH_DONE
