#!/bin/bash
#SBATCH --job-name=valleakALL
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Is the ENTIRE validation split contaminated, or were the 12 probed chains unlucky?
# Reuses the 13152-chain training database job 22215316 already built.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR"
SRC=/orcd/scratch/orcd/011/chenxiou/valleak
WORK=/orcd/scratch/orcd/011/chenxiou/valleak_all
mkdir -p "$WORK"
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh

FS=$(command -v foldseek || echo /home/chenxiou/.conda/envs/foldseek/bin/foldseek)
[ -s "$SRC/tdb.index" ] || { echo "FATAL: no training db at $SRC/tdb"; exit 77; }
echo "reusing training db: $(wc -l < "$SRC/tdb.index") chains"

if [ ! -d "$WORK/query" ]; then
    python scratchpad/valleak_full_val.py --out "$WORK" || exit 78
fi
NQ=$(find "$WORK/query" -name '*.pdb' | wc -l)
echo "val queries=$NQ"
[ "$NQ" -gt 0 ] || { echo "FATAL: no val PDBs"; exit 79; }

BYTES=$(find "$WORK/query" -name '*.pdb' | wc -c)
[ "$BYTES" -lt $(( $(getconf ARG_MAX) / 2 )) ] || { echo "FATAL: argv $BYTES too large"; exit 80; }

rm -f "$WORK"/vdb* "$WORK"/aln* "$WORK"/hits.tsv*
# ⛔ FILES, never directories: createdb with several input dirs builds nothing and returns success.
"$FS" createdb "$WORK"/query/s*/*.pdb "$WORK/vdb"
[ -s "$WORK/vdb.index" ] || { echo "FATAL: createdb produced no vdb.index"; exit 81; }
echo "vdb entries: $(wc -l < "$WORK/vdb.index")"

# ⛔ -a or convertalis core-dumps on the fident/alnlen fields. ⛔ TM mode, not 3Di.
"$FS" search "$WORK/vdb" "$SRC/tdb" "$WORK/aln" "$WORK/tmp" \
    --alignment-type 1 -a -e 10 --max-seqs 2000 --threads "${SLURM_CPUS_PER_TASK:-32}"
"$FS" convertalis "$WORK/vdb" "$SRC/tdb" "$WORK/aln" "$WORK/hits.tsv" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,fident,alnlen,evalue"
[ -s "$WORK/hits.tsv" ] || { echo "FATAL: convertalis produced no hits.tsv"; exit 82; }

echo "=== validation-set contamination, full split ==="
WORK="$WORK" NQ="$NQ" python - <<'PY'
import os
work = os.environ["WORK"]
best = {}
for line in open(os.path.join(work, "hits.tsv")):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4:
        continue
    q, t = f[0], f[1]
    if q == t:
        continue
    try:
        qtm = float(f[3])          # normalised by the QUERY = the val chain. Never max().
    except ValueError:
        continue
    if q not in best or qtm > best[q][1]:
        best[q] = (t, qtm)

nq = int(os.environ["NQ"])
hit = len(best)
# ⛔ Chains with NO hit at all never appear in hits.tsv, so they must be counted from nq, not from
# len(best) -- otherwise the denominator silently becomes "chains that matched something".
print(f"validation chains written      : {nq}")
print(f"  with ANY training hit        : {hit}")
print(f"  with NO hit at all           : {nq - hit}")
print()
print(f"{'TM band':>14} {'chains':>8} {'% of all val':>14}")
bands = [(0.9, 9.9), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
for lo, hi in bands:
    n = sum(1 for _, tm in best.values() if lo <= tm < hi)
    lab = f">= {lo:.1f}" if hi > 1 else f"{lo:.1f} - {hi:.1f}"
    print(f"{lab:>14} {n:>8} {100.0 * n / nq:>13.1f}%")
n_no = nq - hit
print(f"{'no hit':>14} {n_no:>8} {100.0 * n_no / nq:>13.1f}%")

n_fold = sum(1 for _, tm in best.values() if tm >= 0.5)
print(f"\n⭐ {n_fold}/{nq} ({100.0*n_fold/nq:.1f}%) of validation chains have a SAME-FOLD training")
print("   neighbour (TM >= 0.5). The split is sequence_similarity at 25% identity, which separates")
print("   SEQUENCES, not FOLDS.")
print("\n⚠️ Lower bound: the training side is 13152 chains (one per cluster, one epoch) out of a")
print("   259k split, so true nearest neighbours can only be CLOSER and this % can only RISE.")
PY
echo "VALLEAKALL_EXIT=$?"
