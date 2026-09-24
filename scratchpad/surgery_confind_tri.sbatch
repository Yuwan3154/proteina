#!/bin/bash
#SBATCH --job-name=tri_surgery
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
# ConFind tri surgery + gate: old code/old weights vs new code/surgered weights on the mask arm must match exactly.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
NEW=${REPO:-$S/proteina_dev}
OLD=$S/proteina_cmhier
OLD_CKPT="$OLD/store/tri_full384/checkpoints/chk_best_contact_precL_epoch=00000175_step=000000071950-EMA.ckpt"
OUT=$S/branch_points_tri/tri_confind_ft_init_from71950ema.ckpt
mkdir -p "$(dirname "$OUT")" "$S/.tmp"
export CUE_HOST=engaging TORCH_CUDA_ARCH_LIST=9.0   # activate.sh requires a value; CPU job
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
source /home/chenxiou/cue-openfold-env-sm120/activate.sh
export TMPDIR=$S/.tmp
git -C "$NEW" log -1 --format="[new] %h"; git -C "$OLD" log -1 --format="[old] %h"
PYTHONPATH=$NEW python $NEW/scratchpad/surgery_confind_tri.py "$OLD_CKPT" "$OUT" || exit 1
PYTHONPATH=$OLD python $NEW/scratchpad/tri_forward_probe.py \
    $OLD/configs/experiment_config/model/nn/contact_map_tri_30M.yaml "$OLD_CKPT" $S/.tmp/probe_old.pt || exit 1
PYTHONPATH=$NEW python $NEW/scratchpad/tri_forward_probe.py \
    $NEW/configs/experiment_config/model/nn/contact_map_tri_30M_cb8.yaml "$OUT" $S/.tmp/probe_new.pt || exit 1
python - <<'PY'
import torch
a = torch.load("/orcd/scratch/orcd/011/chenxiou/.tmp/probe_old.pt")["contact_map_logits"]
b = torch.load("/orcd/scratch/orcd/011/chenxiou/.tmp/probe_new.pt")["contact_map_logits"]
d = (a - b).abs().max().item()
print(f"[gate] mask-arm contact logits, old code+old weights vs new code+surgered weights: max|diff| = {d:.3g} "
      f"over {a.numel()} cells ({'PASS' if d == 0 else 'FAIL'}; logits abs-mean {a.abs().mean().item():.4f})")
raise SystemExit(0 if d == 0 else 1)
PY
exit $?
