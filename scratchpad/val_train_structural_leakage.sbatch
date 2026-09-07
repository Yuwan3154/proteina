#!/bin/bash
#SBATCH --job-name=valleak
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Do the validation chains have STRUCTURAL neighbours in training?
#
# ⛔ The split is `split_type: sequence_similarity` at 25% identity. That separates SEQUENCES, not
# FOLDS -- two chains under 25% identity can share a fold. So structural homologs in training are
# possible and must be measured, not assumed away.
#
# ⛔ TM mode (--alignment-type 1), NOT the default 3Di. 3Di is fast but produces false negatives on
# remote structural similarity, which would UNDERSTATE leakage -- the exact direction that would let
# us wrongly conclude the validation set is clean.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
WORK=/orcd/scratch/orcd/011/chenxiou/valleak
mkdir -p "$WORK"
cd "$WORK"

# foldseek lives in its own conda env on this host, not on PATH and not a module.
FS=$(command -v foldseek || echo /home/chenxiou/.conda/envs/foldseek/bin/foldseek)
echo "foldseek: $FS"
"$FS" version 2>/dev/null || { echo "FATAL: foldseek not found"; exit 77; }

# The 12 probed validation chains, and the training set to search against, are written by the python
# helper -- it resolves chain ids to .pt files and emits CA-only PDBs.
python /orcd/scratch/orcd/011/chenxiou/proteina_sh/scratchpad/build_leakage_pdbs.py \
    --out "$WORK" --n_train "${NTRAIN:-40000}" || exit 78

# ⛔ Resume-safe: on a timeout or crash, rerun the IDENTICAL command against the SAME tmp dir.
# Never rm -rf tmp to "start clean" -- foldseek resumes from it and a wipe throws away hours.
"$FS" createdb "$WORK/query" "$WORK/qdb"
"$FS" createdb "$WORK/train" "$WORK/tdb"
"$FS" search "$WORK/qdb" "$WORK/tdb" "$WORK/aln" "$WORK/tmp" \
    --alignment-type 1 -e 10 --max-seqs 2000 --threads "${SLURM_CPUS_PER_TASK:-32}"
"$FS" convertalis "$WORK/qdb" "$WORK/tdb" "$WORK/aln" "$WORK/hits.tsv" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,fident,alnlen,evalue"

echo "=== top structural neighbour per validation chain ==="
python - <<'PY'
import collections, os
best = {}
path = "/orcd/scratch/orcd/011/chenxiou/valleak/hits.tsv"
for line in open(path):
    f = line.rstrip("\n").split("\t")
    if len(f) < 4:
        continue
    q, t = f[0], f[1]
    # ⛔ qtmscore is normalised by the QUERY -- the validation chain, whose fold is the thing that
    # could have been memorised. NEVER max(qtm, ttm): that inflates 1.5-2x.
    try:
        qtm = float(f[3])
    except ValueError:
        continue
    if q == t:
        continue
    if q not in best or qtm > best[q][1]:
        best[q] = (t, qtm)
print(f"{'val chain':>12} {'best train hit':>16} {'TM(norm by val)':>16}")
for q in sorted(best):
    t, tm = best[q]
    flag = "  <-- FOLD MATCH" if tm >= 0.5 else ""
    print(f"{q:>12} {t:>16} {tm:>16.3f}{flag}")
n_hi = sum(1 for _, tm in best.values() if tm >= 0.5)
print(f"\n{n_hi}/{len(best)} validation chains have a training neighbour at TM >= 0.5 (same fold).")
print("Chains with NO hit at all are absent from this table -- that is itself informative.")
PY
echo "VALLEAK_EXIT=$?"
