#!/bin/bash
#SBATCH --job-name=d36tc_build
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc S1a (CPU, cue_openfold = the D36m / synth-index build env). Real #!/bin/bash file, never --wrap.
#   0  tri EMA-ness file test (raw last.ckpt vs frozen -EMA, same step) + last-EMA md5 pin
#   1  raw cifs: 34 targets from d36u/natives (the scoring natives), hits from d36m_tri_data/raw, Control-A training
#      chains from pdb_train/raw (the files training processed); md5 manifest
#   2  rows CSV under the derived name -> prepare_data -> coords/sequence gates
#   3  dssp_target -> gate; Control A1 (processed .pt vs stored training .pt, exact)
#   4  sequence column backfill + datamodule setup (sequence clusters, both splits non-empty)
set -euo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
D=$P/data
I=$P/inputs
DS=d36tc_tri_max384
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SKIP_CONFIND_PRECOMPUTE=1 PYTHONPATH=$REPO PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
[ ! -e "$D" ] || { echo "FATAL: $D exists -- fresh data dir expected"; exit 78; }

echo "=== STAGE 0: tri EMA-ness file test ==="
python scratchpad/d36tc_ema_vs_raw.py --step 60505 \
  --raw $S/proteina_cfft/store/tri_confindsynth_ft/checkpoints/last.ckpt \
  --run-last-ema $S/proteina_cfft/store/tri_confindsynth_ft/checkpoints/last-EMA.ckpt \
  --ema $S/stageA/ckpt/cfft_tri_last-EMA_step60505.ckpt

echo "=== STAGE 1: raw cifs ==="
mkdir -p "$D/raw" "$P/logs"
python - "$I/d36tc_rows.csv" "$D/raw" "$S" <<'PY'
import csv, gzip, hashlib, os, shutil, sys
rows, raw, S = sys.argv[1], sys.argv[2], sys.argv[3]
man = []
for r in csv.DictReader(open(rows)):
    dst = os.path.join(raw, f"{r['pdb']}.cif")
    if os.path.exists(dst):
        continue                                    # an entry shared by two rows (e.g. 4FPQ)
    if r["role"] == "target":
        src = f"{S}/d36u/natives/{r['pdb']}.cif"
        shutil.copyfile(src, dst)
        alt = f"{S}/d36m_tri_data/raw/{r['pdb']}.cif"
        same = os.path.exists(alt) and open(alt, "rb").read() == open(src, "rb").read()
        note = f"d36m_raw_identical={same if os.path.exists(alt) else 'absent'}"
    elif r["role"] == "reference":
        src = f"{S}/d36m_tri_data/raw/{r['pdb']}.cif"
        shutil.copyfile(src, dst)
        note = ""
    else:
        src = f"/orcd/pool/006/chenxiou/proteina/data/pdb_train/raw/{r['pdb'][1:3]}/{r['pdb']}.cif.gz"
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        note = "gunzipped"
    man.append(f"{r['pdb']}\t{r['role']}\t{src}\t{hashlib.md5(open(dst, 'rb').read()).hexdigest()}\t{note}")
open(os.path.join(raw, "..", "raw_manifest.tsv"), "w").write("\n".join(man) + "\n")
print(f"raw: {len(man)} files copied"); print("\n".join(m for m in man if "False" in m or "absent" in m))
PY

echo "=== STAGE 2: process ==="
python scratchpad/d36tc_setup_dataset.py --repo "$REPO" --dataset "$DS" --rows "$I/d36tc_rows.csv"
python - <<'PY'
import os, hydra
with hydra.initialize_config_dir(os.path.join(os.environ["PYTHONPATH"], "configs/datasets_config/pdb"), version_base=hydra.__version__):
    cfg = hydra.compose(config_name="d36tc_tri_max384")
dm = hydra.utils.instantiate(cfg.datamodule)
dm.prepare_data()
print("prepare_data returned")
PY
python scratchpad/d36tc_verify_inputs.py --processed-dir "$D/processed" --target-seqs "$I/d36tc_target_seqs.json" \
  --hit-manifest "$I/d36tc_hit_manifest.json" --extra "$I/controlA_stems.txt" --stage coords

echo "=== STAGE 3: DSSP + Control A1 ==="
python -m proteinfoundation.utils.precompute_dssp_targets --processed-dir "$D/processed" --workers 16 \
  --output-csv "$P/logs/dssp_progress.csv" --log-path "$P/logs/dssp.log"
python scratchpad/d36tc_verify_inputs.py --processed-dir "$D/processed" --target-seqs "$I/d36tc_target_seqs.json" \
  --hit-manifest "$I/d36tc_hit_manifest.json" --extra "$I/controlA_stems.txt" --stage dssp
set +e
python scratchpad/d36tc_controlA.py --part a1 --stems "$I/controlA_stems.txt" --mine "$D/processed" \
  --write-stored-paths "$I/controlA_stored_paths.tsv"
a1=$?
set -e
echo "CONTROL_A1_RC=$a1"

echo "=== STAGE 4: sequence backfill + datamodule setup (clusters) ==="
python - <<'PY'
import glob, os, torch, pandas as pd, hydra
D = "/orcd/pool/006/chenxiou/d36tc/data"
cands = glob.glob(os.path.join(D, "df_pdb_*minl40_maxl700*.csv"))
assert len(cands) == 1, cands
df = pd.read_csv(cands[0])
df["sequence"] = [torch.load(os.path.join(D, "processed", f"{i}.pt"), map_location="cpu", weights_only=False).sequence
                  for i in df["id"].astype(str)]
assert (df["sequence"].str.len() > 0).all()
df.to_csv(cands[0], index=False)
print(f"backfilled sequence for {len(df)} rows")
with hydra.initialize_config_dir(os.path.join(os.environ["PYTHONPATH"], "configs/datasets_config/pdb"), version_base=hydra.__version__):
    cfg = hydra.compose(config_name="d36tc_tri_max384")
dm = hydra.utils.instantiate(cfg.datamodule)
dm.prepare_data(); dm.setup("fit")
print(f"splits: train {len(dm.train_ds)} val {len(dm.val_ds)}")
assert len(dm.train_ds) > 0 and len(dm.val_ds) > 0
PY
echo "D36TC_BUILD_DONE A1_RC=$a1"
exit $a1
