#!/bin/bash
#SBATCH --job-name=valleakFS
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
# foldseek stage ONLY, over PDBs that job 22212863 already wrote. Rewriting them would cost another
# ~30 min for no gain.
#
# ⛔ WHY THIS EXISTS: `foldseek createdb dir1/ dir2/ ... out` with MULTIPLE directories produces
# NOTHING and returns success. In 22212863 it printed its parameter block and then went straight on
# -- no "Output file:", no progress bar, no database -- so search and convertalis silently had no
# target and hits.tsv was never created. A single directory works (the 12-file query db was fine);
# several do not. Pass FILES instead, and verify the db exists before searching.
set -uo pipefail
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
source "$ENV_DIR/.venv/bin/activate"
WORK=${WORK:-/orcd/scratch/orcd/011/chenxiou/valleak}
cd "$WORK"

FS=$(command -v foldseek || echo /home/chenxiou/.conda/envs/foldseek/bin/foldseek)
"$FS" version >/dev/null 2>&1 || { echo "FATAL: foldseek not found"; exit 77; }

NPDB=$(find "$WORK/train" -name '*.pdb' | wc -l)
NQ=$(find "$WORK/query" -name '*.pdb' | wc -l)
echo "targets=$NPDB  queries=$NQ"
[ "$NPDB" -gt 0 ] && [ "$NQ" -gt 0 ] || { echo "FATAL: missing PDBs"; exit 78; }

# ⛔ The file list goes on the command line, so guard ARG_MAX explicitly rather than discovering the
# limit as a truncated database. ~55 bytes/path; getconf ARG_MAX is the kernel ceiling.
BYTES=$(find "$WORK/train" -name '*.pdb' | wc -c)
LIMIT=$(getconf ARG_MAX)
echo "argv bytes=$BYTES  ARG_MAX=$LIMIT"
[ "$BYTES" -lt $((LIMIT / 2)) ] || {
    echo "FATAL: file list $BYTES B too close to ARG_MAX $LIMIT."
    echo "       Shard into per-directory createdb + iterative 'foldseek concatdbs' (also _ss/_ca/_h)."
    exit 79; }

rm -f "$WORK"/tdb* "$WORK"/aln* "$WORK"/hits.tsv
"$FS" createdb "$WORK"/train/s*/*.pdb "$WORK/tdb"
# ⛔ Verify, do not assume: this is the exact step that failed silently last time.
[ -s "$WORK/tdb.index" ] || { echo "FATAL: createdb produced no tdb.index"; exit 80; }
echo "tdb entries: $(wc -l < "$WORK/tdb.index")"

"$FS" createdb "$WORK/query" "$WORK/qdb2"
[ -s "$WORK/qdb2.index" ] || { echo "FATAL: createdb produced no qdb2.index"; exit 81; }

# ⛔ TM mode (--alignment-type 1), never the default 3Di: 3Di false-negatives on remote structural
# similarity would UNDERSTATE leakage, i.e. bias us toward wrongly calling the split clean.
"$FS" search "$WORK/qdb2" "$WORK/tdb" "$WORK/aln" "$WORK/tmp2" \
    --alignment-type 1 -e 10 --max-seqs 2000 --threads "${SLURM_CPUS_PER_TASK:-32}"
"$FS" convertalis "$WORK/qdb2" "$WORK/tdb" "$WORK/aln" "$WORK/hits.tsv" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,fident,alnlen,evalue"
[ -s "$WORK/hits.tsv" ] || { echo "FATAL: convertalis produced no hits.tsv"; exit 82; }

echo "=== top structural neighbour per validation chain ==="
WORK="$WORK" NPDB="$NPDB" python - <<'PY'
import os
best = {}
work = os.environ["WORK"]
for line in open(os.path.join(work, "hits.tsv")):
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

# Verdicts from generation job 22211889, joined by the loader-order map (job 22214928), which was
# checked against the probe's own length column for all 16 chains.
MIRROR = {"2xg4_B", "5h9l_A", "4xc9_B", "2epi_C", "2o5v_A"}
print(f"{'val chain':>12} {'gen':>6} {'best train hit':>16} {'TM(by val)':>11}  flag")
for q in sorted(best):
    t, tm = best[q]
    stem = q[:-4] if q.endswith(".pdb") else q
    tag = "MIRROR" if stem in MIRROR else ""
    fold = "  <-- FOLD MATCH" if tm >= 0.5 else ""
    print(f"{stem:>12} {'':>6} {t:>16} {tm:>11.3f}  {tag}{fold}")

n_hi = sum(1 for _, tm in best.values() if tm >= 0.5)
print(f"\n{n_hi}/{len(best)} validation chains have a training neighbour at TM >= 0.5 (same fold).")
mir = [(q, tm) for q, (t, tm) in best.items() if (q[:-4] if q.endswith('.pdb') else q) in MIRROR]
oth = [(q, tm) for q, (t, tm) in best.items() if (q[:-4] if q.endswith('.pdb') else q) not in MIRROR]
for lab, grp in (("mirrored", mir), ("others", oth)):
    if grp:
        vals = sorted(tm for _, tm in grp)
        print(f"  {lab:>9}: n={len(vals)}  TMs={[round(v,3) for v in vals]}")
print(f"\n⚠️ Searched against {os.environ['NPDB']} training chains, not the full 259k split: the")
print("   train_dataloader yields one chain per cluster, so an epoch ends well before 40k.")
print("   A POSITIVE (fold match) is therefore solid; a NEGATIVE is weak -- a neighbour could sit")
print("   in the part of training that was never searched.")
print("⚠️ Counts, not medians: with n this small a median moves on one chain.")
PY
echo "VALLEAKFS_EXIT=$?"
