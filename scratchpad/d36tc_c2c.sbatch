#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# ⛔ No --error: one log for stdout + stderr.
#
# D36tc c2c pass: stageB_c2c_from_maps.py (env block and artefact verify = stageB_c2c_from_maps.sbatch), plus a
# REQUIRED checkpoint md5 and a state_dict == ema.params check (the file must be an EMA freeze).
#   CKPT=<c2c ckpt>  EXPECT_MD5=<full md5, required>  CHAINS=<chain list>  LABEL=<arm name>
#   and EXACTLY ONE of  MAPS_DIR=<tri dump>  |  NATIVE=1 (native-map ceiling: the target's own F2C map)
#   optional: N_SEEDS (1)  DATASET (default d36tc_c2c_max384)  REPO ($S/proteina_d36tc)
# ⭐ Seed = 1234 + position in the SORTED chain list (+ seed_idx * 1e6): arms of one group must get the SAME CHAINS
#    list so their draws are paired.
# Outputs: $P/c2c/B_<LABEL>.jsonl, PDBs under $P/c2c_pdb/<LABEL>/<shard>/<stem>/. Partition/gres/time/name on sbatch.
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
DATASET="${DATASET:-d36tc_c2c_max384}"
PROC=$P/data/processed
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
module load cuda/12.8 2>/dev/null || module load cuda/12.6 2>/dev/null || true
export CUE_HOST=engaging
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$TORCH_CUDA_ARCH_LIST" ] || { echo "FATAL: could not read compute_cap"; exit 79; }
source "$ENV_DIR/.venv/bin/activate"
source "$ENV_DIR/activate.sh"
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data
export SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO:$REPO/scratchpad${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp/chenxiou_${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$P/logs" "$P/c2c" "$P/c2c_pdb"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SIGMA_DATA=16.0   # the value the twin trained with (af3_diffusion reads it at import)
echo "[edm] SIGMA_DATA=$SIGMA_DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

: "${CKPT:?}" "${EXPECT_MD5:?}" "${CHAINS:?}" "${LABEL:?}"
NATIVE="${NATIVE:-0}"
MAPS_DIR="${MAPS_DIR:-}"
N_SEEDS="${N_SEEDS:-1}"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint $CKPT"; exit 77; }
[ -f "$CHAINS" ] || { echo "FATAL: no chain list $CHAINS"; exit 77; }
if [ "$NATIVE" = 1 ] && [ -n "$MAPS_DIR" ]; then echo "FATAL: set MAPS_DIR or NATIVE=1, not both"; exit 77; fi
if [ "$NATIVE" != 1 ] && [ ! -f "$MAPS_DIR/samples.jsonl" ]; then echo "FATAL: NATIVE!=1 and no samples.jsonl in MAPS_DIR='$MAPS_DIR'"; exit 77; fi
OUT=$P/c2c/B_${LABEL}.jsonl
[ -e "$OUT" ] && { echo "FATAL: $OUT exists -- one fresh file per arm"; exit 78; }
[ -e "$P/c2c_pdb/$LABEL" ] && { echo "FATAL: $P/c2c_pdb/$LABEL exists -- one fresh PDB dir per arm"; exit 78; }
MD5=$(md5sum < "$CKPT" | cut -d" " -f1)
echo "[d36tc-c2c] $LABEL dataset=$DATASET native=$NATIVE maps=${MAPS_DIR:-none} n_seeds=$N_SEEDS chains=$CHAINS ($(grep -c . "$CHAINS"))"
echo "[d36tc-c2c] ckpt=$CKPT md5=$MD5 expect=$EXPECT_MD5"
[ "$MD5" = "$EXPECT_MD5" ] || { echo "FATAL: checkpoint md5 $MD5 != EXPECT_MD5 $EXPECT_MD5"; exit 75; }
cd "$REPO" || exit 76
git log -1 --format="[repo] $REPO HEAD %h"
# Specific to the frozen twin file: a full Lightning checkpoint (optimizer_states present, job 25359190) whose
# state_dict ("model." prefix) must equal ema.params tensor for tensor (1067/1067), and whose key set must equal
# the c2c model's own (same MODEL_CFG stageB builds).
python - "$CKPT" <<'PY'
import sys, torch
from gen_c2c_structures import MODEL_CFG
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
sd, ema = ck["state_dict"], ck["ema"]["params"]
model_keys = set(ContactToCoordTrainer(model_cfg=MODEL_CFG).model.state_dict())
keys = {k[len("model."):]: k for k in sd if k.startswith("model.")}
assert len(ema) > 0 and len(keys) == len(sd) == len(ema) == len(model_keys), \
    f"counts: state_dict {len(sd)} (model.-prefixed {len(keys)}), ema {len(ema)}, model {len(model_keys)}"
assert set(keys) == set(ema) == model_keys, "key sets differ between state_dict, ema.params and the model"
bad = [k for k in ema if not torch.equal(sd[keys[k]], ema[k])]
print(f"[ema-check] {len(ema)} tensors (= model's {len(model_keys)}), {len(bad)} differ between state_dict and ema.params")
sys.exit(73 if bad else 0)
PY
erc=$?
[ "$erc" -eq 0 ] || { echo "FATAL: EMA check failed rc=$erc"; exit 73; }
SRC=(--maps_dir "$MAPS_DIR"); [ "$NATIVE" = 1 ] && SRC=(--native)
python scratchpad/stageB_c2c_from_maps.py --ckpt "$CKPT" --weights ema --dataset "$DATASET" --chains "$CHAINS" \
    --n_seeds "$N_SEEDS" --out "$OUT" --label "$LABEL" "${SRC[@]}" --pdb_dir "$P/c2c_pdb" 2>&1
rc=$?
echo "STAGEB_RC[$LABEL]=$rc"

# ⛔ Verify the ARTEFACT: one row per (chain, map, seed), one gen PDB per row, and L == the .pt length (no crop).
python - "$OUT" "$CHAINS" "$NATIVE" "$MAPS_DIR" "$N_SEEDS" "$P/c2c_pdb/$LABEL" "$PROC" <<'PY'
import collections, glob, json, os, sys
import torch
out, chains, native, maps_dir, n_seeds, pdb_root, proc = sys.argv[1:8]
native, n_seeds = native == "1", int(n_seeds)
stems = sorted({l.split()[0] for l in open(chains) if l.strip()})
per = collections.defaultdict(list)
if native:
    for st in stems:
        per[st].append(None)
else:
    for l in open(os.path.join(maps_dir, "samples.jsonl")):
        if l.strip():
            r = json.loads(l)
            if r["stem"] in stems:
                per[r["stem"]].append(r["sample_index"])
want = {(st, si, sd) for st in stems for si in per[st] for sd in range(n_seeds)}
rows = [json.loads(l) for l in open(out) if l.strip()] if os.path.exists(out) else []
got = [(r["stem"], r["sample_index"], r["seed"]) for r in rows]
pdbs = glob.glob(os.path.join(pdb_root, "*", "*", "*_gen.pdb"))
L_pt = {st: int(torch.load(os.path.join(proc, f"{st}.pt"), map_location="cpu", weights_only=False).coords.shape[0])
        for st in stems}
crop = [(r["stem"], r["L"], L_pt[r["stem"]]) for r in rows if int(r["L"]) != L_pt.get(r["stem"], -1)]
print(f"[verify] chains {len(stems)}  expected {len(want)}  rows {len(rows)}  distinct {len(set(got))}  "
      f"gen PDBs {len(pdbs)}  L != .pt length: {len(crop)}")
ok = len(got) == len(set(got)) and set(got) == want and len(pdbs) == len(want) and not crop
if not ok:
    print(f"[verify] FAIL: missing {len(want - set(got))}, unexpected {len(set(got) - want)}, "
          f"duplicates {len(got) - len(set(got))}, gen PDBs {len(pdbs)} vs {len(want)}, crop {crop[:10]}")
    sys.exit(1)
print("[verify] OK: one row and one gen PDB per (chain, map, seed); every L == .pt length")
PY
vrc=$?
echo "VERIFY_RC=$vrc"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$vrc"
