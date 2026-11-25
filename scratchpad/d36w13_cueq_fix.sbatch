#!/bin/bash
#SBATCH --job-name=d36w13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/orcd/pool/006/chenxiou/d36_w13_cueq_fix/logs/%x-%j.out
# D36 W13: test the OpenFold cuEq triangle-multiplication in-place fix (openfold branch fix-cueq-trimul-inplace).
#   1. tests.test_cuequivariance.TestCuEquivarianceInplaceContract on FIX_SHA (must PASS) and on FIX_SHA with the trimul
#      module reverted to BUG_SHA (known-bad control: must FAIL).
#   2. W11 e2 harness (d36w11_forensics.py @0df4cdb, unchanged) on 6UCK_A with the fixed openfold first on PYTHONPATH:
#      arms cueq / cueq_mult must now equal plain.
# Runs on fresh clones only; the production openfold checkout and the cue_openfold env are not modified.
# Partition/time/exclude on the sbatch command line; FIX_SHA via --export.
set -uo pipefail
: "${FIX_SHA:?}"
BUG_SHA=74f2f8a  # fork main before the fix (== W11's openfold)
W=/orcd/pool/006/chenxiou/d36_w13_cueq_fix
OF_SRC=/orcd/scratch/orcd/011/chenxiou/openfold
PR=/orcd/pool/006/chenxiou/d36_w11_forensics/repo
JOB=$W/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}  # a requeue gets a fresh dir
OUT=$JOB/out
mkdir -p "$OUT" || exit 3
[ "$(git -C "$PR" rev-parse --short=7 HEAD)" = 0df4cdb ] || { echo "FATAL: W11 harness repo not at 0df4cdb"; exit 70; }
echo "[harness] $PR HEAD $(git -C "$PR" log -1 --format=%h) tracked changes $(git -C "$PR" status --porcelain --untracked-files=no | wc -l)"
for arm in of_fix of_bugctl; do
  git clone -q --no-checkout "$OF_SRC" "$JOB/$arm" || exit 4
  git -C "$JOB/$arm" checkout -q "$FIX_SHA" || exit 5
  cp -r "$OF_SRC/openfold/resources" "$JOB/$arm/openfold/" || exit 6  # untracked data (stereo_chemical_props.txt)
done
git -C "$JOB/of_bugctl" checkout -q "$BUG_SHA" -- openfold/model/triangular_multiplicative_update.py || exit 7
echo "[of_fix] HEAD $(git -C "$JOB/of_fix" log -1 --format='%h %s')"
echo "[of_bugctl] staged revert vs FIX_SHA:"; git -C "$JOB/of_bugctl" diff --cached --stat
source "$PR/scratchpad/d36tc_af2.env.sh"
export TMPDIR=/tmp/chenxiou_w13_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pm in "params_model_1_ptm.npz dfca7cd972028fc6adcc0c8995857a2a"; do
  set -- $pm; h=$(md5sum < "$AF2RANK_PARAMS_DIR/$1" | cut -d' ' -f1); echo "[params] $1 $h (SC $2)"
  [ "$h" = "$2" ] || { echo "FATAL: $1 differs from the SuperCloud production params"; exit 72; }
done
rc=0
declare -A urc
for arm in of_fix of_bugctl; do
  cd "$JOB/$arm" || exit 3
  python -c "import torch, cuequivariance_torch, openfold; print('[env]', openfold.__file__, 'cueq', cuequivariance_torch.__version__, 'torch', torch.__version__, 'tf32', torch.backends.cuda.matmul.allow_tf32)"
  python -c "import os, sys, openfold; sys.exit(not openfold.__file__.startswith('$JOB/$arm/'))" || { echo "FATAL: openfold not imported from $JOB/$arm"; exit 73; }
  python -m unittest -v tests.test_cuequivariance.TestCuEquivarianceInplaceContract > "$OUT/unittest_$arm.log" 2>&1
  urc[$arm]=$?
  echo "[unittest] $arm rc ${urc[$arm]}"; cat "$OUT/unittest_$arm.log"
done
for arm in of_bugctl of_fix; do  # the upstream-issue reproduction snippet, verbatim
  cd "$JOB/$arm" || exit 3
  echo "[repro] $arm"
  python - <<'EOF'
import torch
from openfold.config import model_config
from openfold.model.primitives import Linear, lecun_normal_init_
from openfold.model.template import TemplatePairStack
from openfold.model.triangular_multiplicative_update import TriangleMultiplicationOutgoing

torch.manual_seed(0)

def randomize(m):  # "final"/"gating" inits are zero, which would hide the bug
    for x in m.modules():
        if isinstance(x, Linear):
            lecun_normal_init_(x.weight)
    return m

cfg = model_config("model_1_ptm").model.template.template_pair_stack

# 1) aliasing: the cuEq branch returns a new tensor and leaves z unchanged
tmu = randomize(TriangleMultiplicationOutgoing(cfg.c_t, cfg.c_hidden_tri_mul)).cuda().eval()
z = torch.randn(1, 22, 22, cfg.c_t, device="cuda")
mask = torch.ones(1, 22, 22, device="cuda")
z0 = z.clone()
with torch.no_grad():
    out = tmu(z, mask=mask, inplace_safe=True, _add_with_inplace=True,
              use_cuequivariance_multiplicative_update=True)
print(out.data_ptr() == z.data_ptr(), torch.equal(z, z0))   # main: False True   (plain path: True False)

# 2) template pair stack: cuEq vs plain at inference
stack = randomize(TemplatePairStack(**cfg)).cuda().eval()
t = torch.randn(3, 22, 22, cfg.c_t, device="cuda")
m = torch.ones(3, 22, 22, device="cuda")
with torch.no_grad():
    ref = stack(t.clone(), m, chunk_size=None, inplace_safe=False)
    plain = stack(t.clone(), m, chunk_size=None, inplace_safe=True)
    cueq = stack(t.clone(), m, chunk_size=None, inplace_safe=True,
                 use_cuequivariance_multiplicative_update=True)
print((plain - ref).abs().max().item(), (cueq - ref).abs().max().item())
EOF
  echo "[repro] $arm rc $?"
done
# fix: rc 0, both tests ran, none skipped; control: fails by assertion only (not by an import/CUDA error)
FL=$OUT/unittest_of_fix.log; BL=$OUT/unittest_of_bugctl.log
{ [ "${urc[of_fix]}" -eq 0 ] && grep -q "^Ran 2 tests" "$FL" && grep -q "^OK$" "$FL" && ! grep -qi "skipped" "$FL"; } \
  || { echo "GATE FAIL: tests do not cleanly pass on the fix"; rc=10; }
{ [ "${urc[of_bugctl]}" -ne 0 ] && grep -q "AssertionError" "$BL" && ! grep -q "^ERROR:" "$BL" \
  && grep -q "^FAIL: test_trimul_inplace_add_returns_z" "$BL" && grep -q "^FAIL: test_template_pair_stack_cueq_mult_inference" "$BL"; } \
  || { echo "GATE FAIL: buggy control does not fail by assertion in both tests (vacuous or broken)"; [ $rc -eq 0 ] && rc=11; }
cd "$PR" || exit 3
PYTHONPATH=$JOB/of_fix:$PR python -c "import sys, openfold; sys.exit(not openfold.__file__.startswith('$JOB/of_fix/'))" \
  || { echo "FATAL: harness would not import the fixed openfold"; exit 74; }
# -m from PR: the script dir holds scratchpad/openfold/, which would shadow openfold.utils
PYTHONPATH=$JOB/of_fix:$PR python -m scratchpad.d36w11_forensics --out "$OUT/e2" --mode e2 --e2-pids 6UCK_A
r=$?; echo "[rc] e2 $r"; [ $r -eq 0 ] || { [ $rc -eq 0 ] && rc=$r; }
# result gate: with the fix, arms cueq and cueq_mult must reproduce plain pTM and TM exactly
python - "$OUT/e2/e2_endtoend.csv" <<'EOF'
import csv, sys
rows = {r["arm"]: r for r in csv.DictReader(open(sys.argv[1]))}
anchor = (rows["plain"]["ptm"], rows["plain"]["tm"]) == ("0.4629422724246979", "0.8615")  # W4-diag / W11 plain
print(f"[e2gate] plain anchor (W4-diag 0.4629422724246979 / 0.8615) reproduced={anchor}")
for arm in ("cueq", "cueq_mult", "cueq_attn", "cueq_fix"):
    same = all(rows[arm][k] == rows["plain"][k] for k in ("ptm", "tm"))
    print(f"[e2gate] {arm}: ptm {rows[arm]['ptm']} tm {rows[arm]['tm']} vs plain {rows['plain']['ptm']} {rows['plain']['tm']} equal={same}")
sys.exit(not (anchor and all(rows[a][k] == rows["plain"][k] for a in ("cueq", "cueq_mult") for k in ("ptm", "tm"))))
EOF
r=$?; echo "[rc] e2gate $r"; [ $r -eq 0 ] || { [ $rc -eq 0 ] && rc=20; }
ls "$OUT/e2" | wc -l
echo "W13_RC=$rc"
exit $rc
