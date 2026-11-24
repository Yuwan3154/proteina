#!/bin/bash
#SBATCH --job-name=d36tc_score
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc final scoring (CPU): d36tc_score.py -> $P/d36tc_per_target.csv + counts per arm.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P="${P:-/orcd/pool/006/chenxiou/d36tc}"   # D36tc-v3: P overridable (default = published dir)
REPO="${REPO:-$S/proteina_d36tc}"
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
# D36tc-v3: optional passthroughs (unset = published call): TC_OUTCSV TC_PREFIX TC_GROUPS_DIR TC_C2C_PDB TC_AF2RANK TC_NATIVES TC_GROUPS TC_SIZES TC_SCORE_CHAINS
XA=()
[ -n "${TC_PREFIX:-}" ] && XA+=(--prefix "$TC_PREFIX")
[ -n "${TC_GROUPS_DIR:-}" ] && XA+=(--groups-dir "$TC_GROUPS_DIR")
[ -n "${TC_C2C_PDB:-}" ] && XA+=(--c2c-pdb "$TC_C2C_PDB")
[ -n "${TC_AF2RANK:-}" ] && XA+=(--af2rank "$TC_AF2RANK")
[ -n "${TC_NATIVES:-}" ] && XA+=(--natives "$TC_NATIVES")
[ -n "${TC_GROUPS:-}" ] && XA+=(--groups "$TC_GROUPS")   # not GROUPS: a bash builtin array (user gids)
[ -n "${TC_SIZES:-}" ] && XA+=(--expect-sizes "$TC_SIZES")
[ -n "${TC_SCORE_CHAINS:-}" ] && XA+=(--score-chains "$TC_SCORE_CHAINS")
echo "[score] extra args: ${XA[*]:-none}"
python scratchpad/d36tc_score.py --out "$P/${TC_OUTCSV:-d36tc_per_target.csv}" "${XA[@]}"
rc=$?
echo "SCORE_RC=$rc"
exit $rc
