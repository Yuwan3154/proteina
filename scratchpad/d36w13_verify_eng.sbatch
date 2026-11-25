#!/bin/bash
#SBATCH --job-name=d36w13v
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/orcd/pool/006/chenxiou/d36_w13_cueq_fix/logs/%x-%j.out
# D36 W13 install verification on Engaging: the INSTALLED openfold checkout (editable in every openfold env) must be at
# EXPECT_OF_SHA; every env must import it; the fix's unit tests pass in place; W11 e2 6UCK_A through the env's own
# openfold gives cuEq == plain exactly. Partition/time/exclude on the sbatch command line; EXPECT_OF_SHA via --export.
set -uo pipefail
: "${EXPECT_OF_SHA:?}"
W=/orcd/pool/006/chenxiou/d36_w13_cueq_fix
OF=/orcd/scratch/orcd/011/chenxiou/openfold
PR=/orcd/pool/006/chenxiou/d36_w11_forensics/repo
JOB=$W/install_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}
OUT=$JOB/out
mkdir -p "$OUT" || exit 3
[ "$(git -C "$OF" rev-parse HEAD)" = "$EXPECT_OF_SHA" ] || { echo "FATAL: $OF HEAD $(git -C "$OF" rev-parse HEAD) != $EXPECT_OF_SHA"; exit 70; }
[ "$(git -C "$OF" status --porcelain --untracked-files=no | wc -l)" -eq 0 ] || { echo "FATAL: $OF has tracked changes"; exit 71; }
[ "$(git -C "$PR" rev-parse --short=7 HEAD)" = 0df4cdb ] || { echo "FATAL: W11 harness repo not at 0df4cdb"; exit 72; }
echo "[of] $OF $(git -C "$OF" branch --show-current) $(git -C "$OF" log -1 --format='%H %s')"
echo "[harness] $PR HEAD $(git -C "$PR" log -1 --format=%h) tracked changes $(git -C "$PR" status --porcelain --untracked-files=no | wc -l)"
rc=0
OFR=$(readlink -f "$OF")
for py in /home/chenxiou/.conda/envs/cue_openfold/bin/python /home/chenxiou/.conda/envs/cue_openfold_fb/bin/python \
          /home/chenxiou/cue-openfold-env/.venv/bin/python /home/chenxiou/cue-openfold-env-sm120/.venv/bin/python \
          /orcd/scratch/orcd/011/chenxiou/cue-openfold-env-sm120-dssp/.venv/bin/python; do
  f=$(cd /tmp && OPENFOLD_DISABLE_FLASH_ATTN=1 "$py" -c "import importlib.util as u, os; s=u.find_spec('openfold'); print(os.path.realpath(s.origin), 'cueq_installed=%s' % (u.find_spec('cuequivariance_torch') is not None))" 2>&1 | tail -1)
  echo "[envimport] $py -> $f"
  case "$f" in "$OFR/openfold/__init__.py "*) ;; *) echo "GATE FAIL: $py does not resolve openfold to $OFR"; [ $rc -eq 0 ] && rc=12;; esac
done
source "$PR/scratchpad/d36tc_af2.env.sh"
export TMPDIR=/tmp/chenxiou_w13v_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton
export PYTHONDONTWRITEBYTECODE=1  # keep the installed checkout free of new artifacts
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
h=$(md5sum < "$AF2RANK_PARAMS_DIR/params_model_1_ptm.npz" | cut -d' ' -f1); echo "[params] params_model_1_ptm.npz $h"
[ "$h" = dfca7cd972028fc6adcc0c8995857a2a ] || { echo "FATAL: params differ from the SuperCloud production params"; exit 73; }
cd "$OF" || exit 3
python -c "import os, sys, openfold, cuequivariance_torch; print('[env]', openfold.__file__, 'cueq', cuequivariance_torch.__version__); sys.exit(not os.path.realpath(openfold.__file__).startswith('$OFR/'))" \
  || { echo "FATAL: cue_openfold does not import the installed checkout"; exit 74; }
python -m unittest -v tests.test_cuequivariance.TestCuEquivarianceInplaceContract > "$OUT/unittest.log" 2>&1
u=$?; echo "[unittest] rc $u"; cat "$OUT/unittest.log"
{ [ $u -eq 0 ] && grep -q "^Ran 2 tests" "$OUT/unittest.log" && grep -q "^OK$" "$OUT/unittest.log" && ! grep -qi "skipped" "$OUT/unittest.log"; } \
  || { echo "GATE FAIL: unit tests do not cleanly pass on the installed checkout"; [ $rc -eq 0 ] && rc=10; }
cd "$PR" || exit 3
export PYTHONPATH=$PR  # as W11: proteinfoundation from the harness repo; openfold from the env's editable install
python -c "import os, sys, openfold; sys.exit(not os.path.realpath(openfold.__file__).startswith('$OFR/'))" \
  || { echo "FATAL: harness would not import the installed openfold"; exit 75; }
python -m scratchpad.d36w11_forensics --out "$OUT/e2" --mode e2 --e2-pids 6UCK_A
r=$?; echo "[rc] e2 $r"; [ $r -eq 0 ] || { [ $rc -eq 0 ] && rc=$r; }
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
echo "[of_after] tracked changes $(git -C "$OF" status --porcelain --untracked-files=no | wc -l)"
echo "W13V_RC=$rc"
exit $rc
