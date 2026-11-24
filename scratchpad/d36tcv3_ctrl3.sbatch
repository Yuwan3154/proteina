#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# D36tc-v3 plain-kernel control (CPU): d36w8_tc_ctrl3.py on the plain-kernel re-score of the published d36tc_A_tmpl
# 8TVL_A + 6UCK_A pools (written by d36tc_af2rank.sbatch KERNELS=plain into NEW) vs W4b control (iii) (REF); exit 1
# unless PASS_EXACT / PASS_6DP (the W8 pre-registered rule). Partition/time/name/output on the sbatch command line.
#   NEW=<plain af2rank out root>   optional REF (default = the W4b control (iii) root W8 compared against)  REPO
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_d36tc}"
REF="${REF:-$S/d36_af2rank_mask/w4_20261121/ctrl_plain/prod_plain}"
: "${NEW:?}"
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
python -m scratchpad.d36w8_tc_ctrl3 --new "$NEW" --ref "$REF"
rc=$?
echo "CTRL3_RC=$rc"
exit $rc
