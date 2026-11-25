#!/bin/bash
#SBATCH --job-name=d36w13v
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/home/gridsan/cou/d36/w13_verify/logs/%x-%j.out
# D36 W13 install verification on SuperCloud: the INSTALLED openfold checkout (editable in cathfold, cathfold_extlig,
# pottsmpnn, cue-openfold-env) must be at EXPECT_OF_SHA; each env must resolve it; the fix's unit tests pass in place
# (cathfold, the AF2Rank production env); W11 e2 6UCK_A (harness from the PRW worktree at 0df4cdb's descendant, its
# Engaging paths repointed to copies on SuperCloud) gives cuEq == the in-process fix exactly (cuEq vs plain reported). Partition/time on the command line.
set -uo pipefail
: "${EXPECT_OF_SHA:?}" "${PRW:?}"
W=/home/gridsan/cou/d36/w13_verify
OF=/home/gridsan/cou/openfold
JOB=$W/install_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}
OUT=$JOB/out
mkdir -p "$OUT" || exit 3
[ "$(git -C "$OF" rev-parse HEAD)" = "$EXPECT_OF_SHA" ] || { echo "FATAL: $OF HEAD $(git -C "$OF" rev-parse HEAD) != $EXPECT_OF_SHA"; exit 70; }
[ "$(git -C "$OF" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "FATAL: $OF has tracked changes"; exit 71; }
git -C "$PRW" merge-base --is-ancestor 0df4cdb HEAD || { echo "FATAL: $PRW does not contain the W11 harness 0df4cdb"; exit 72; }
git -C "$PRW" diff --quiet 0df4cdb -- scratchpad/d36w11_forensics.py proteinfoundation/ || { echo "FATAL: harness or proteinfoundation (working tree) differs from 0df4cdb"; exit 72; }
echo "[of] $OF $(git -C "$OF" branch --show-current) $(git -C "$OF" log -1 --format='%H %s')"
echo "[harness] $PRW $(git -C "$PRW" log -1 --format=%h)"
rc=0
OFR=$(readlink -f "$OF")
set +u  # /etc/profile, Lmod and conda activate expand unset variables
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
set -u
for py in /home/gridsan/cou/.conda/envs/cathfold/bin/python /home/gridsan/cou/.conda/envs/cathfold_extlig/bin/python \
          /home/gridsan/cou/.conda/envs/pottsmpnn/bin/python /home/gridsan/cou/cue-openfold-env/.venv/bin/python \
          /home/gridsan/cou/.conda/envs/cathfold_foldseek/bin/python; do
  f=$(cd /tmp && "$py" -c "import importlib.util as u, os; s=u.find_spec('openfold'); print(s and os.path.realpath(s.origin), 'cueq_installed=%s' % (u.find_spec('cuequivariance_torch') is not None))" 2>&1 | tail -1)
  echo "[envimport] $py -> $f"
  case "$py" in *cathfold_foldseek*) continue;; esac  # report only: its editable finder file is missing (pre-existing)
  case "$f" in "$OFR/openfold/__init__.py "*) ;; *) echo "GATE FAIL: $py does not resolve openfold to $OFR"; [ $rc -eq 0 ] && rc=12;; esac
done
set +u
source activate cathfold
set -u
export OMP_NUM_THREADS=20
export TMPDIR=/tmp/cou_w13v_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton
export PATH="$HOME/.local/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1  # keep the installed checkout free of new artifacts
export AF2RANK_PARAMS_DIR=$OF/openfold/resources/params W13_BATCHES=$W/batches W13_NATIVES=$W/natives W13_E2_OUT=$OUT/e2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
h=$(md5sum < "$AF2RANK_PARAMS_DIR/params_model_1_ptm.npz" | cut -d' ' -f1); echo "[params] params_model_1_ptm.npz $h"
[ "$h" = dfca7cd972028fc6adcc0c8995857a2a ] || { echo "FATAL: params differ from the production params"; exit 73; }
h=$(md5sum < "$W13_BATCHES/6UCK_A_none_model_1_ptm_d36ab.pt" | cut -d' ' -f1); echo "[batch] $h"
[ "$h" = 06a393362beb8405a8c54623f2c4b93f ] || { echo "FATAL: batch differs from the Engaging W4-diag batch"; exit 73; }
cd "$OF" || exit 3
python -c "import os, sys, openfold, cuequivariance_torch; from cuequivariance_ops_torch.triangle_multiplicative_update import CUEQ_TRIMUL_FALLBACK_THRESHOLD as t; print('[env]', openfold.__file__, 'cueq', cuequivariance_torch.__version__, 'trimul_fallback', t); sys.exit(not os.path.realpath(openfold.__file__).startswith('$OFR/'))" \
  || { echo "FATAL: cathfold does not import the installed checkout (or the harness's cuEq import fails)"; exit 74; }
python -m unittest -v tests.test_cuequivariance.TestCuEquivarianceInplaceContract > "$OUT/unittest.log" 2>&1
u=$?; echo "[unittest] rc $u"; cat "$OUT/unittest.log"
{ [ $u -eq 0 ] && grep -q "^Ran 2 tests" "$OUT/unittest.log" && grep -q "^OK$" "$OUT/unittest.log" && ! grep -qi "skipped" "$OUT/unittest.log"; } \
  || { echo "GATE FAIL: unit tests do not cleanly pass on the installed checkout"; [ $rc -eq 0 ] && rc=10; }
cd "$PRW" || exit 3
export PYTHONPATH=$PRW
python -c "import os, sys, openfold; sys.exit(not os.path.realpath(openfold.__file__).startswith('$OFR/'))" \
  || { echo "FATAL: harness would not import the installed openfold"; exit 75; }
python - <<'EOF'
import os
import sys

import scratchpad.d36w11_forensics as m

m.PARAMS = os.environ["AF2RANK_PARAMS_DIR"]  # the harness hard-codes Engaging paths; repoint to the SuperCloud copies
m.BATCHES = os.environ["W13_BATCHES"]
m.NAT_W4 = os.environ["W13_NATIVES"]
sys.argv = ["d36w11_forensics", "--out", os.environ["W13_E2_OUT"], "--mode", "e2", "--e2-pids", "6UCK_A"]
m.main()
EOF
r=$?; echo "[rc] e2 $r"; [ $r -eq 0 ] || { [ $rc -eq 0 ] && rc=$r; }
python - "$OUT/e2/e2_endtoend.csv" <<'EOF'
import csv, sys
rows = {r["arm"]: r for r in csv.DictReader(open(sys.argv[1]))}
# Gate: the installed cuEq path equals the in-process W11 fix wrapper exactly (independent of GPU and cuEq version).
# Reported only: cuEq vs plain and the L40S W4-diag anchor (V100 + cuEq 0.9.1 may differ from L40S + cuEq 0.7.0 at round-off).
anchor = (rows["plain"]["ptm"], rows["plain"]["tm"]) == ("0.4629422724246979", "0.8615")
print(f"[e2gate] plain {rows['plain']['ptm']} / {rows['plain']['tm']}; equals the L40S W4-diag anchor: {anchor}")
for arm in ("cueq", "cueq_mult", "cueq_attn", "cueq_fix"):
    same = all(rows[arm][k] == rows["plain"][k] for k in ("ptm", "tm"))
    print(f"[e2gate] {arm}: ptm {rows[arm]['ptm']} tm {rows[arm]['tm']} vs plain {rows['plain']['ptm']} {rows['plain']['tm']} equal_plain={same}")
gate = all(rows[a][k] == rows["cueq_fix"][k] for a in ("cueq", "cueq_mult") for k in ("ptm", "tm"))
print(f"[e2gate] cueq == cueq_mult == cueq_fix (ptm, tm): {gate}")
sys.exit(not gate)
EOF
r=$?; echo "[rc] e2gate $r"; [ $r -eq 0 ] || { [ $rc -eq 0 ] && rc=20; }
echo "[of_after] tracked changes $(git -C "$OF" status --porcelain --untracked-files=no | wc -l)"
echo "W13V_RC=$rc"
exit $rc
