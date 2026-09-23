#!/bin/bash
#SBATCH --job-name=stageB_c2c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
#
# Stage B (Directive B): tri-sampled contact maps -> the matching c2c -> all-atom -> quality vs native.
# Partition, --gres and --time go on the sbatch command line; none is fixed here.
#   CKPT=<c2c ckpt>  DATASET=<the c2c's OWN dataset yaml>  CHAINS=<chain list>  OUT=<fresh .jsonl>
#   LABEL=<arm name>  and EXACTLY ONE of  MAPS_DIR=<tri dump dir>  |  NATIVE=1 (ceiling arm)
#   optional: WEIGHTS=ema|raw (ema)  N_SEEDS (1)  STEPS (script default C2C_INFERENCE_STEPS)
#             PDB_DIR  USALIGN  REPO
# Env block copied from train_c2c.sbatch.
set -uo pipefail
REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_tri}"
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/orcd/scratch/orcd/011/chenxiou/.tmp
mkdir -p "$TMPDIR" /orcd/scratch/orcd/011/chenxiou/c2c_store/logs
[ ${#TMPDIR} -le 70 ] || { echo "FATAL: TMPDIR ${#TMPDIR} chars > 70: $TMPDIR"; exit 78; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ⛔ The value tbeta and its ConFind twin train with. af3_diffusion reads it at IMPORT, so it is set
# here, never inherited: an unset variable would also give 16.0, but only by a default nobody sees.
export SIGMA_DATA=16.0
echo "[edm] SIGMA_DATA=$SIGMA_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

: "${CKPT:?}" "${DATASET:?}" "${CHAINS:?}" "${OUT:?}" "${LABEL:?}"
NATIVE="${NATIVE:-0}"
MAPS_DIR="${MAPS_DIR:-}"
N_SEEDS="${N_SEEDS:-1}"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint $CKPT"; exit 77; }
[ -f "$CHAINS" ] || { echo "FATAL: no chain list $CHAINS"; exit 77; }
if [ "$NATIVE" = 1 ] && [ -n "$MAPS_DIR" ]; then
    echo "FATAL: set MAPS_DIR or NATIVE=1, not both"; exit 77
fi
if [ "$NATIVE" != 1 ] && [ ! -f "$MAPS_DIR/samples.jsonl" ]; then
    echo "FATAL: NATIVE!=1 and no samples.jsonl in MAPS_DIR='$MAPS_DIR'"; exit 77
fi
if [ -s "$OUT" ]; then
    echo "FATAL: $OUT exists and is not empty -- one fresh file per arm"; exit 78
fi
mkdir -p "$(dirname "$OUT")"

ARGS=(--ckpt "$CKPT" --weights "${WEIGHTS:-ema}" --dataset "$DATASET" --chains "$CHAINS"
      --n_seeds "$N_SEEDS" --out "$OUT" --label "$LABEL")
if [ "$NATIVE" = 1 ]; then ARGS+=(--native); else ARGS+=(--maps_dir "$MAPS_DIR"); fi
if [ -n "${STEPS:-}" ]; then ARGS+=(--steps "$STEPS"); fi
if [ -n "${PDB_DIR:-}" ]; then ARGS+=(--pdb_dir "$PDB_DIR"); fi
if [ -n "${USALIGN:-}" ]; then ARGS+=(--usalign "$USALIGN"); fi

if [ "$NATIVE" = 1 ]; then SRC=native; else SRC="$MAPS_DIR"; fi
echo "[stageB] label=$LABEL dataset=$DATASET weights=${WEIGHTS:-ema} n_seeds=$N_SEEDS steps=${STEPS:-default} maps=$SRC"
echo "[stageB] chains=$CHAINS ($(grep -c . "$CHAINS") lines) ckpt=$CKPT md5=$(md5sum < "$CKPT" | cut -c1-12)"
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h %cd"
python scratchpad/stageB_c2c_from_maps.py "${ARGS[@]}"
rc=$?
echo "STAGEB_RC[$LABEL]=$rc"

# ⛔ Verify the ARTEFACT, not the exit code: exactly one row per (chain, tri sample, seed).
python - "$OUT" "$CHAINS" "$N_SEEDS" "$NATIVE" "$MAPS_DIR" <<'PY'
import collections, json, os, sys
out, chains, n_seeds, native, maps_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
stems = sorted({l.split()[0] for l in open(chains) if l.strip()})
want_stems = set(stems)
per = collections.defaultdict(list)
if native:
    for st in stems:
        per[st].append(None)
else:
    for l in open(os.path.join(maps_dir, "samples.jsonl")):
        if l.strip():
            r = json.loads(l)
            if r["stem"] in want_stems:
                per[r["stem"]].append(r["sample_index"])
n_samp = sorted({len(per[st]) for st in stems})
rows = [json.loads(l) for l in open(out) if l.strip()] if os.path.exists(out) else []
want = {(st, si, sd) for st in stems for si in per[st] for sd in range(n_seeds)}
got = [(r["stem"], r["sample_index"], r["seed"]) for r in rows]
print(f"[verify] chains {len(stems)}  samples/chain {n_samp}  seeds {n_seeds}  "
      f"expected {len(want)}  rows {len(rows)}  distinct {len(set(got))}")
ok = (len(n_samp) == 1 and len(rows) == len(stems) * n_samp[0] * n_seeds
      and len(got) == len(set(got)) and set(got) == want)
if not ok:
    print(f"[verify] FAIL: missing {len(want - set(got))}, unexpected {len(set(got) - want)}, "
          f"duplicates {len(got) - len(set(got))}")
    sys.exit(1)
print("[verify] OK: one row per (chain, tri sample, seed)")
PY
VERIFY_RC=$?
echo "VERIFY_RC=$VERIFY_RC"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
exit "$VERIFY_RC"
