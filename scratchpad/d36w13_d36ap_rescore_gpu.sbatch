#!/bin/bash
#SBATCH --job-name=d36w13_rescore
#SBATCH -p debug-gpu,xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH -t 2:00:00
#SBATCH -o /home/gridsan/cou/logs/d36w13_rescore-%j.out
# W13-B: re-run D36ap D1 (OpenFold cells O1/O6w/O6s) and D2 (conformer-1 controls) with the FIXED OpenFold
# (fix-cueq-trimul-inplace). The original scripts in ~/d36/d36ap are run UNCHANGED (md5-pinned) with their
# original env, kernel flags and inputs; only the output dirs are new. Resources = the original d36ap sbatches.
# STAGE=shake: D1 on 6ZYG_A (O1 + O6s, GPU0) and D2 phase 0 (6XRY_A c01/c02, model_1_ptm, sc + unm, GPU1).
# STAGE=d2: masked c01 control (worklist phase<=1, sc+both) then unmasked c01 control (worklist_c01, unm); GPU0 m1, GPU1 m2.
# STAGE=d1: GPU0 O1 then O6w, GPU1 O6s over the 130-decoy manifest; rows already written are skipped, so a rerun resumes.
# Submit: git show <sha>:scratchpad/d36w13_d36ap_rescore_gpu.sbatch | env STAGE=.. EXPECT_OF_SHA=.. LAUNCH_SHA=<sha> sbatch
set -o pipefail
source /etc/profile
STAGE=${STAGE:?}
EXPECT_OF_SHA=${EXPECT_OF_SHA:?}
OF=/home/gridsan/cou/openfold
P=/home/gridsan/cou/proteina
D=/home/gridsan/cou/d36/d36ap
D2=$D/d2
WT=/home/gridsan/cou/proteina_d36ap
WT_SHA=6e198a3f464ed8f7e064a5c29803531e11bd05a9  # D2's worktree commit (status_d36ap.txt)
R=$D/rescore_w13/$STAGE
mkdir -p $R/logs
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID PART=$SLURM_JOB_PARTITION STAGE=$STAGE LAUNCH_SHA=${LAUNCH_SHA:-unset} START=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

fail() { echo "GATE FAIL: $*"; exit 3; }
# --- gate 1: openfold checkout = the installed fix, no local edits to the package
OF_HEAD=$(git -C $OF rev-parse HEAD) || fail "git rev-parse on $OF"
[ "$OF_HEAD" = "$EXPECT_OF_SHA" ] || fail "openfold HEAD $OF_HEAD != EXPECT_OF_SHA $EXPECT_OF_SHA"
git -C $OF diff --quiet HEAD -- openfold || fail "openfold/ has uncommitted changes"
echo "GATE openfold HEAD=$OF_HEAD clean"
# --- gate 2: the original scripts and inputs are byte-identical to what D1/D2 ran
echo "33cb3525941dd6322adae615c66fc67c  $D/d36ap_gate_of.py
b29255a7f63722873f9d080fac338a8d  $D/run_of_env.sh
51bb37c90673f76868b2fb9336316b3d  $D/gate/manifest.csv
0183382faf98b80d58704b39ab78590d  $D2/d2_af2.py
5888c852a7f45da82f2da2cc9aba7493  $D2/worklist.csv
50e12eeeb1cf4923069e99e785387383  $D2/worklist_c01.csv" | md5sum -c - || fail "original script/input md5"
echo "GATE kernel flags as written in the original scripts:"
grep -n "use_deepspeed_evoformer_attention\|use_cuequivariance" $D/d36ap_gate_of.py $D2/d2_af2.py
# inputs not covered by md5: none modified after the prep step that wrote them (D2 worklist.csv, D1 manifest.csv)
NEWER=$( { find $D2/inputs $D2/conformers /home/gridsan/cou/openfold/openfold/resources/params -type f -newer $D2/worklist.csv
  find $D/gate/ca /home/gridsan/cou/openfold/openfold/resources/params -type f -newer $D/gate/manifest.csv
  tail -n +2 $D/gate/manifest.csv | awk -F, '{print $3; print $5}' | sort -u | xargs sh -c 'find "$@" -newer '"$D"'/gate/manifest.csv' _ ; } | wc -l)
[ "$NEWER" -eq 0 ] || fail "$NEWER input files modified after the original prep"
MISSING=$( { for d in $D2/inputs $D2/conformers $D/gate/ca /home/gridsan/cou/openfold/openfold/resources/params; do echo $d; done
  tail -n +2 $D/gate/manifest.csv | awk -F, '{print $3; print $5}' | sort -u; } | while read f; do [ -e "$f" ] || echo "$f"; done | wc -l)
[ "$MISSING" -eq 0 ] || fail "$MISSING input paths missing"
echo "GATE inputs: 0 files modified after the original prep (d2 inputs/conformers/params, gate/ca, 156 manifest decoys+refs)"
env | grep -E '^(PYTHONPATH|CUDA|TORCH|CUEQ|NVIDIA|OMP|LD_LIBRARY_PATH)' | sort
P_HEAD=$(git -C $P rev-parse HEAD)
same_of() { [ "$(git -C $OF rev-parse HEAD)" = "$EXPECT_OF_SHA" ] && git -C $OF diff --quiet HEAD -- openfold; }
same_p() { [ "$(git -C $P rev-parse HEAD)" = "$P_HEAD" ] && git -C $P diff --quiet HEAD -- proteinfoundation; }  # live checkout D1 imports
same_wt() { [ "$(git -C $WT rev-parse HEAD)" = "$WT_SHA" ] && [ -z "$(git -C $WT status --porcelain --untracked-files=no)" ]; }
same_code() {  # code unchanged since the gate, for what this STAGE imports
  same_of || return 1
  if [ "$STAGE" != d2 ]; then same_p || return 1; fi
  if [ "$STAGE" != d1 ]; then same_wt || return 1; fi
  return 0
}

GATE_PY='
import importlib.metadata as md, os, subprocess, sys
import torch
import openfold
assert openfold.__file__.startswith("/home/gridsan/cou/openfold/"), openfold.__file__
print("GATE openfold", openfold.__file__)
print("GATE versions", {p: md.version(p) for p in ("cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12", "torch", "openfold")})
print("GATE device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
if sys.argv[1] == "d1":
    P = "/home/gridsan/cou/proteina/"
    sys.path.insert(0, P + "proteinfoundation/prediction_pipeline")
    import af2rank_openfold_scorer
    files = sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and m.__file__.startswith(P)})
    for f in files:
        rel = f[len(P):]
        ref = "2fde18e54f16b50bba28f40996cfafe1afda708e" if rel.endswith("af2rank_openfold_scorer.py") else "86c7a3af72115c36de4841f8183ed9df5682a52d"
        want = subprocess.run(["git", "-C", P, "rev-parse", f"{ref}:{rel}"], capture_output=True, text=True, check=True).stdout.strip()
        have = subprocess.run(["git", "-C", P, "hash-object", f], capture_output=True, text=True, check=True).stdout.strip()
        print("GATE proteina", rel, "@" + ref[:7], "OK" if want == have else "MISMATCH")
        assert want == have, rel
    assert any(f.endswith("af2rank_openfold_scorer.py") for f in files)
else:
    import proteinfoundation.prediction_pipeline.native_template_af2 as nta
    import proteinfoundation.utils.openfold_inference as ofi
    for m in (nta, ofi):
        print("GATE worktree module", m.__file__)
        assert m.__file__.startswith("/home/gridsan/cou/proteina_d36ap/"), m.__file__
print("GATE_PY_OK", sys.argv[1])
'

d2_env() {  # = d2_gpu.sbatch's environment lines
  module load conda/Python-ML-2025b-pytorch cuda/12.6
  conda deactivate 2>/dev/null
  source activate cathfold
  export OMP_NUM_THREADS=20
  export PATH="$HOME/.local/bin:$PATH"
  export PYTHONPATH=$WT:${PYTHONPATH:-}
  cd $D2
}

if [ "$STAGE" = shake ] || [ "$STAGE" = d2 ]; then
  if [ ! -d $WT ]; then
    git -C $P worktree add --detach $WT $WT_SHA || fail "worktree add"
  fi
  [ "$(git -C $WT rev-parse HEAD)" = "$WT_SHA" ] || fail "worktree HEAD != $WT_SHA"
  [ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || fail "worktree has tracked changes"
  echo "GATE worktree $WT @ $WT_SHA clean"
  ( d2_env; CUDA_VISIBLE_DEVICES=0 python -c "$GATE_PY" d2 ) || fail "python gate d2"
fi
if [ "$STAGE" = shake ] || [ "$STAGE" = d1 ]; then
  CUDA_VISIBLE_DEVICES=0 bash $D/run_of_env.sh python -c "$GATE_PY" d1 || fail "python gate d1"
fi

PARAMS=/home/gridsan/cou/openfold/openfold/resources/params
run_d1() {  # $1 gpu, $2 manifest, $3 out csv, $4 cells
  same_of && same_p || { echo "CODE CHANGED since the gate (~/proteina $P_HEAD or openfold)"; return 4; }
  CUDA_VISIBLE_DEVICES=$1 bash $D/run_of_env.sh python $D/d36ap_gate_of.py --manifest $2 --out $3 --cells $4
}
# a row cut by the wall limit would be appended to on resume (as d36ap_gate2 found): keep only complete rows
CLEAN_PY='
import csv, sys
for fn in sys.argv[1:]:
    rows = list(csv.reader(open(fn)))
    if not rows:
        print("CLEAN", fn, "empty, left as is")
        continue
    good = [r for r in rows[1:] if len(r) == len(rows[0]) and r[-1].endswith("Z")]
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(rows[0]); w.writerows(good)
    print("CLEAN", fn, "rows", len(rows) - 1, "kept", len(good), "dropped", len(rows) - 1 - len(good))
'
complete_rows() {  # rows with every field and a utc stamp
  bash $D/run_of_env.sh python -c 'import csv, sys; r = list(csv.reader(open(sys.argv[1]))); print(sum(len(x) == len(r[0]) and x[-1].endswith("Z") for x in r[1:]) if r else 0)' $1
}
run_d2() {  # $1 gpu, $2 model, $3 worklist, $4 max_phase, $5 pred_root, $6 cells
  same_of && same_wt || { echo "CODE CHANGED since the gate (openfold or $WT)"; return 4; }
  ( d2_env; CUDA_VISIBLE_DEVICES=$1 python -u $D2/d2_af2.py --model_name $2 --worklist $3 --max_phase $4 \
      --samples_dir $D2/inputs/gtctl --pred_root $5 --params_dir $PARAMS --recycles 6 --seed 0 --cells $6 )
}

case $STAGE in
  shake)
    mkdir -p $R/d1 $R/d2
    awk -F, 'NR == 1 || $1 == "6ZYG_A"' $D/gate/manifest.csv > $R/d1/manifest_6ZYG_A.csv
    ( run_d1 0 $R/d1/manifest_6ZYG_A.csv $R/d1/o1.csv O1 && run_d1 0 $R/d1/manifest_6ZYG_A.csv $R/d1/o6s.csv O6s ) > $R/logs/gpu0_$SLURM_JOB_ID.log 2>&1 &
    p0=$!
    run_d2 1 model_1_ptm $D2/worklist.csv 0 $R/d2/pred sc,unm > $R/logs/gpu1_$SLURM_JOB_ID.log 2>&1 &
    p1=$!
    ;;
  d2)
    ( run_d2 0 model_1_ptm $D2/worklist.csv 1 $R/pred sc,both && run_d2 0 model_1_ptm $D2/worklist_c01.csv 1 $R/pred_unmasked unm ) > $R/logs/gpu0_$SLURM_JOB_ID.log 2>&1 &
    p0=$!
    ( run_d2 1 model_2_ptm $D2/worklist.csv 1 $R/pred sc,both && run_d2 1 model_2_ptm $D2/worklist_c01.csv 1 $R/pred_unmasked unm ) > $R/logs/gpu1_$SLURM_JOB_ID.log 2>&1 &
    p1=$!
    ;;
  d1)
    EXIST=$(ls $R/o1.csv $R/o6w.csv $R/o6s.csv 2>/dev/null)
    [ -z "$EXIST" ] || bash $D/run_of_env.sh python -c "$CLEAN_PY" $EXIST || fail "clean partial rows"
    ( run_d1 0 $D/gate/manifest.csv $R/o1.csv O1 && run_d1 0 $D/gate/manifest.csv $R/o6w.csv O6w ) > $R/logs/gpu0_$SLURM_JOB_ID.log 2>&1 &
    p0=$!
    run_d1 1 $D/gate/manifest.csv $R/o6s.csv O6s > $R/logs/gpu1_$SLURM_JOB_ID.log 2>&1 &
    p1=$!
    ;;
  *) fail "unknown STAGE $STAGE" ;;
esac
while kill -0 $p0 2>/dev/null || kill -0 $p1 2>/dev/null; do
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo " $(date +%T)"
  sleep 300
done
wait $p0; rc0=$?
wait $p1; rc1=$?
rows() { if [ -f "$1" ]; then complete_rows "$1"; else echo 0; fi; }
case $STAGE in
  shake) echo "ARTIFACTS d1 o1=$(rows $R/d1/o1.csv)/5 o6s=$(rows $R/d1/o6s.csv)/5 d2 json=$(find $R/d2/pred -name '*.json' | wc -l)/4 tmp=$(find $R/d2/pred -name '*.tmp' | wc -l)" ;;
  d2) echo "ARTIFACTS masked json=$(find $R/pred -name '*.json' | wc -l)/104 unmasked json=$(find $R/pred_unmasked -name '*.json' | wc -l)/50 tmp=$(find $R -name '*.tmp' | wc -l)" ;;
  d1) echo "ARTIFACTS o1=$(rows $R/o1.csv)/130 o6w=$(rows $R/o6w.csv)/130 o6s=$(rows $R/o6s.csv)/130" ;;
esac
same_code; rcc=$?
echo "W13B_RC gpu0=$rc0 gpu1=$rc1 code_unchanged_rc=$rcc openfold_HEAD_end=$(git -C $OF rev-parse HEAD) proteina_HEAD_end=$(git -C $P rev-parse HEAD) END=$(date)"
[ $rc0 -eq 0 ] && [ $rc1 -eq 0 ] && [ $rcc -eq 0 ]
exit $?
