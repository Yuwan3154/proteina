#!/bin/bash
# Cross-run TopoRef comparison, stage C submit (login node: sbatch only). Per (model, arm, chunk): one L40S sampling pass
# (xrun_pass.sbatch) -> afterok the matching c2c (stageB_c2c_from_maps.sbatch from the T7 checkout, as the T7-NEWEST
# harness: CB-8 -> tbeta 29,620, T8 / T8b -> ConFind twin 29,000; 1 c2c draw per map, seeded by chain order only).
#   chunks: the 88 chains sorted, CHUNK_N chains per chunk, each listed N_SAMPLES times (8 paired samples, user 10-28);
#           identical lists for every model and arm, so samples (seed_everything) and c2c draws are paired.
#   usage: MODELS="cb8 t8 t8b" ARMS="self equal indel" CHUNKS="0 1 ..." T8B_CKPT=... bash scratchpad/xrun_submit_c.sh
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
X=$S/toporef_xrun
N_SAMPLES=8                     # user 2026-10-28 "8 paired samples"
CHUNK_N=11                      # 88 chains -> 8 chunks of 88 samples: one pass well inside the 6 h mit_normal_gpu limit
TB=$S/c2c_store/c2c_cb8_tbeta/final_tbeta_s29620.ckpt
TW=$S/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt
DCB=pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828
DCF=pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828
G="--gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=06:00:00 --no-requeue --exclude=node5101,node5204"
mkdir -p "$X/stageB" "$X/logs"
if [ ! -d "$X/chunks" ]; then  # built once, atomically (tmp dir + mv), then shared by every model and arm
  cmp -s "$X/select/chains.txt" "$X/cb8pool/chains_equal.txt" || { echo "FATAL: equal and indel chain sets differ"; exit 2; }
  T=$X/chunks.tmp.$$; mkdir -p "$T"
  LC_ALL=C sort "$X/select/chains.txt" | grep . > "$T/chains_sorted.txt"
  n=0
  while read -r st; do
    k=$(( n / CHUNK_N ))
    for arm in self equal indel; do
      for _ in $(seq $N_SAMPLES); do echo "$st" >> "$T/${arm}_c$k.txt"; done
      echo "$st" >> "$T/${arm}_c${k}_q.txt"
    done
    n=$(( n + 1 ))
  done < "$T/chains_sorted.txt"
  mv "$T" "$X/chunks"
fi
NC=$(grep -c . "$X/chunks/chains_sorted.txt")
for arm in self equal indel; do
  [ "$(cat "$X/chunks/${arm}"_c[0-9]*_q.txt | sort -u | wc -l)" -eq "$NC" ] || { echo "FATAL: $arm _q != $NC chains"; exit 2; }
  [ "$(cat "$X/chunks/${arm}"_c[0-9].txt "$X/chunks/${arm}"_c[0-9][0-9].txt 2>/dev/null | sort | uniq -c | awk -v n=$N_SAMPLES '$1!=n' | wc -l)" -eq 0 ] \
    || { echo "FATAL: $arm chunk lists do not hold every chain exactly $N_SAMPLES times"; exit 2; }
  for f in "$X/chunks/self"_c*; do cmp -s "$f" "$X/chunks/${arm}${f##*/self}" || { echo "FATAL: arm chunk files differ"; exit 2; }; done
done
case " ${MODELS:?} " in *" t8b "*) [ -f "${T8B_CKPT:-}" ] || { echo "FATAL: T8B_CKPT missing"; exit 2; };; esac
cd "$S/proteina_xrun" || exit 3
for m in ${MODELS:?}; do
  if [ "$m" = cb8 ]; then C2C=$TB; D=$DCB; else C2C=$TW; D=$DCF; fi
  for arm in ${ARMS:?}; do
    for k in ${CHUNKS:?}; do
      L=${m}_${arm}_c$k
      J=$(env MODEL=$m ARM=$arm CHUNK=$k SEED=0 T8B_CKPT="${T8B_CKPT:-}" sbatch --parsable --export=ALL -J xp_$L \
          --partition=mit_normal_gpu $G scratchpad/xrun_pass.sbatch) || exit 4  # passes as T7: not preemptable
      B=$(env REPO=$S/proteina_t7 CKPT=$C2C DATASET=$D CHAINS=$X/chunks/${arm}_c${k}_q.txt MAPS_DIR=$X/dumps/$L LABEL=$L \
          OUT=$X/stageB/$L.jsonl sbatch --parsable --export=ALL --dependency=afterok:$J -J xb_$L \
          --partition=mit_normal_gpu,mit_preemptable $G \
          --output=$X/logs/%x-%j.out --error=$X/logs/%x-%j.err $S/proteina_t7/scratchpad/stageB_c2c_from_maps.sbatch) || exit 5
      echo "$L PASS $J C2C $B"
    done
  done
done
