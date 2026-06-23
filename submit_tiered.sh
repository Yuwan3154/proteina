#!/bin/bash
# Submit the tiered GPU-escalation queue for the 48M stage-1 run (see run_stage1_catbalanced_domaincrop_sbatch.sh).
# Idempotent: submits a tier ONLY if I don't already have a job using that GPU type (skips duplicates).
# H200 eligible now; H100 at +6h; L40S at +24h (SLURM --begin gates the escalation; the 4-GPU/user cap
# ensures at most one ever runs -> no checkpoint contention). Safe to re-run anytime to backfill missing tiers.
set -euo pipefail
REPO=/home/chenxiou/proteina
L=run_stage1_catbalanced_domaincrop_sbatch.sh
cd "$REPO"

has_gpu_type() { squeue -u "$USER" -h -o "%b" 2>/dev/null | grep -q "gpu:$1"; }

submit_tier() {  # $1=tier  $2=gres  $3=begin
  if has_gpu_type "$1"; then
    echo "tier $1: I already have a job using gpu:$1 -> skip (no duplicate)."
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
