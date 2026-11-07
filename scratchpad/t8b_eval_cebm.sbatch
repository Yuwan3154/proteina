#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/t8b_eval/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/t8b_eval/logs/%x-%j.err
# T8b eval arm (b): sse_ens24 + ContactEBM (scratchpad/t8b_eval_cebm.py) over one chunk's three T7 passes. CHUNK=<k>
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
C=/orcd/compute/so3/002/chenxi
E=$S/t8b_eval
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
REPO=$S/proteina_xrun
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q L40S || { echo "FATAL: not an L40S"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export PYTHONPATH=$REPO
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"
H=$(git -C "$S/ContactEBM_t8beval" rev-parse --short=7 HEAD)
echo "[cebm-repo] $S/ContactEBM_t8beval HEAD $H"
[ "$H" = f981961 ] || { echo "FATAL: ContactEBM HEAD $H != f981961 (the SuperCloud checkout the regimes session ran)"; exit 75; }
k=${CHUNK:?}
D=$E/dumps
python scratchpad/t8b_eval_cebm.py \
    --dumps "$D/t8b_t7_h_bond_c$k,$D/t8b_t7_tmalign_c$k,$D/t8b_t7_tmalign_smooth_c$k" \
    --chains "$E/chunks/c${k}_q.txt" --pack "$C/proteina_pack/tri_t8b_512_v2.pack" \
    --cebm-repo "$S/ContactEBM_t8beval" --cebm-ckpt "$E/ckpt/confind_a1_prod100_epoch82.pt" \
    --out "$D/t8b_t7_cebm_c$k"
rc=$?
echo "CEBM_RC=$rc"
exit $rc
