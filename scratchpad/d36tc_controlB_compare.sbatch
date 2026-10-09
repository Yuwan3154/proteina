#!/bin/bash
#SBATCH --job-name=d36tc_cmpB
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc CONTROL B compare (CPU): (a) tri full-chunk reruns, 1dmx_A (chunk00) and 1csb_B (chunk01) vs the record dumps;
# (b) c2c isolated on the record dumps, every row of both chunks vs newest_out.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
B=/orcd/pool/006/chenxiou/d36tc/controlB
REPO="${REPO:-$S/proteina_d36tc}"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
python "$REPO/scratchpad/d36tc_controlB_compare.py" \
  --tri 1dmx_A:$B/tri_c00:$S/stageA/t7qn_cf_c00 --tri 1csb_B:$B/tri_c01:$S/stageA/t7qn_cf_c01 \
  --c2c $B/B_c2c_c00.jsonl:$S/t7/newest_out/B_t7qn_cf29000_c00.jsonl \
  --c2c $B/B_c2c_c01.jsonl:$S/t7/newest_out/B_t7qn_cf29000_c01.jsonl
rc=$?
echo "COMPARE_RC=$rc"
exit $rc
