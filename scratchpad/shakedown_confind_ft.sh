#!/bin/bash
#SBATCH --job-name=tri_cf_shake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
# Shakedown of the ConFind tri fine-tune through the REAL launcher (user rule: first ckpt + resume before a long run).
# Toy: the 200-chain ConFind smoke index, accum 2, val every 2 optim steps, val-sampling every val round on 4 chains,
# ckpt every 2 steps, wandb off. Phase 1: fresh -> step 6 (pretrain surgery load, sanity val, accum boundary, val,
# val-sampling, ckpt). Phase 2: same RUN -> step 10 (resume from last.ckpt). Pass --gres=gpu:<type>:2 at submit.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
export REPO="${REPO:?pinned checkout}"
# one RUN per GPU-type variant: the rm -rf below must never hit another variant's store
export CONFIG=training_contact_tri_full384_confindsynth_ft_v1 RUN=tri_cf_shakedown_${SHAKE_TAG:-h200} NGPU=2 CHAIN=0
export SYNTH_INDEX_DIR=$S/synth_index_confind_v1/toy INDEX_FILE=topology_index_confind_synth.pt
head -4 $S/valset_analysis/val_fixed32_max256.txt > $S/.tmp/toy_val4.txt
TOY="opt.accumulate_grad_batches=2 opt.val_check_interval_optim_steps=2 opt.limit_val_batches=2 opt.warmup_steps=2
     log.checkpoint_every_n_steps=2 log.last_ckpt_every_n_steps=2 log.log_wandb=False
     validation_sampling.tmscore_every_n_val_epochs=1 validation_sampling.fixed_chain_list=$S/.tmp/toy_val4.txt"
rm -rf "$REPO/store/$RUN"
for STEPS in 6 10; do
  echo "===== PHASE max_steps=$STEPS ====="
  OVERRIDES="$TOY opt.max_steps=$STEPS" bash "$REPO/run_tri_cb8synth_h200x4.sbatch"
  rc=$?
  echo "PHASE_RC[$STEPS]=$rc"
  ls -la "$REPO/store/$RUN/checkpoints/" 2>&1 | tail -8
  [ $rc -eq 0 ] || exit $rc
done
exit 0
