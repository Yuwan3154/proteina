#!/bin/bash
# Queue the SAME c2c run against every fast card type at once and let the scheduler decide.
# The mkdir lock in train_c2c.sbatch guarantees exactly one of them trains; the losers exit 0
# without touching the checkpoint directory.
#
# ⛔ No L40S and no A100: the user asked for RTX PRO 6000 / H200 / H100 specifically.
#
# Env passed through to every candidate (so the winner, whichever it is, gets the same recipe):
#   LR, NDIFF, ACCUM, INIT_FROM
set -uo pipefail
cd /orcd/scratch/orcd/011/chenxiou/proteina_sh

submit () {   # $1 partition, $2 gres
    GRES="$2" CHAIN=4 \
    LR="${LR:-}" NDIFF="${NDIFF:-}" ACCUM="${ACCUM:-}" INIT_FROM="${INIT_FROM:-}" \
    PRECISION="${PRECISION:-}" VAL_EVERY="${VAL_EVERY:-}" \
        sbatch --partition="$1" --gres="$2" --time=06:00:00 \
               --job-name=c2c scratchpad/train_c2c.sbatch | sed "s/$/  [$1 $2]/"
}

submit pi_so3          gpu:rtx_pro_6000:2
submit pi_so3          gpu:h200:2
submit mit_normal_gpu  gpu:h200:2
submit mit_normal_gpu  gpu:h100:2
submit mit_preemptable gpu:rtx_pro_6000:2
submit mit_preemptable gpu:h200:2
submit mit_preemptable gpu:h100:2
