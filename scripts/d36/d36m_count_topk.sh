#!/bin/bash
# How many decoys did af2rank ACTUALLY score per target, in the arm D36 read (af2rank_uniform_rc6)?
# The sbatch says --af2rank_top_k 5, but the flag and the realised count need not agree, and the
# ContactEBM arm's NFE match was built on the assumption that they do.
set -uo pipefail
C=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_200-epoch_pdb_purge-test_warmup_maxlen-256_cutoff-190828_unified
D=/home/gridsan/cou/proteina/inference/$C/seq_cond_segment
while read -r pid; do
  T="$D/$pid.tar"
  [ -f "$T" ] || { printf '%-9s MISSING TAR\n' "$pid"; continue; }
  m1=$(tar tf "$T" 2>/dev/null | grep -c "af2rank_uniform_rc6/af2rank_analysis/predicted_structures/.*\.pdb$")
  m2=$(tar tf "$T" 2>/dev/null | grep -c "af2rank_uniform_rc6/af2rank_analysis_model_2_ptm/predicted_structures/.*\.pdb$")
  printf '%-9s m1=%-4s m2=%-4s\n' "$pid" "$m1" "$m2"
done < /home/gridsan/cou/d36/all42_ids.txt
