#!/bin/bash
# Submit the tiered GPU-escalation queue for the 48M stage-1 run (see run_stage1_catbalanced_domaincrop_sbatch.sh).
# Idempotent: submits a tier ONLY if I don't already have that cb_dc_s1_48m_<tier> job queued (skips dups).
# H200 eligible now; H100 at +6h; L40S at +24h (SLURM --begin gates the escalation; the 4-GPU/user cap
# ensures at most one ever runs -> no checkpoint contention). Safe to re-run anytime to backfill missing tiers.
set -euo pipefail
REPO=/home/chenxiou/proteina
L=run_stage1_catbalanced_domaincrop_sbatch.sh
cd "$REPO"

# A tier counts as already-queued ONLY if one of MY cb_dc_s1_48m_<tier> jobs exists.
# Check the JOB NAME, not gpu type: the chenxiou account is SHARED with another agent whose
# gpu:h200 jobs would otherwise trip a gpu-type check and wrongly skip my h200 tier.
has_tier() { squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qx "cb_dc_s1_48m_$1"; }

submit_tier() {  # $1=tier  $2=gres  $3=begin
  if has_tier "$1"; then
    echo "tier $1: cb_dc_s1_48m_$1 already queued -> skip (no duplicate)."
    return
  fi
  echo "tier $1: submitting (gres $2, begin $3)"
  # the launcher derives the tier from --job-name (cb_dc_s1_48m_<tier>); --export=ALL just carries the env.
  sbatch --job-name="cb_dc_s1_48m_$1" --gres="$2" --begin="$3" --export=ALL "$L"
}

submit_tier h200 gpu:h200:4 now
submit_tier h100 gpu:h100:4 now+6hours
submit_tier l40s gpu:l40s:4 now+24hours

echo "--- my queue ---"
squeue -u "$USER" -o "%.12i %.20j %.10T %.14b %.12l %.20S %.16r" | grep -E "cb_dc_s1_48m|JOBID"
