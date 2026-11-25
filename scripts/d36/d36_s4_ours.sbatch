#!/bin/bash
#SBATCH --job-name=d36s4_ours
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=40
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/gridsan/cou/logs/d36s4_ours-%A_%a.out
#SBATCH --error=/home/gridsan/cou/logs/d36s4_ours-%A_%a.err

# D36 S4: OUR method (proteina -> ProteinEBM -> af2rank) on the v3 targets, from ~/proteina on branch d36aq-c8.
# Copied from the D36f launcher (tmp/p0/d36f_proteina16.sbatch, itself the LOCKED bad_afdb-26 protocol) with
# exactly four changes:
#   1. Config = *_unified_v3.yaml, a byte copy of the published *_unified.yaml, so outputs land in their own
#      inference/<config>/ dir instead of the published pools.
#   2. AF2Rank kernels: cuEq triangle-multiplication OFF by default (CUEQ_MULT=0). W11 verdict 2026-11-24:
#      openfold@74f2f8a triangular_multiplicative_update.py cuEq branch returns a new tensor under inplace_safe
#      and TemplatePairStackBlock drops it, so template trimul + pair-transition updates are lost every recycle.
#      The attention flag is passed exactly as in D36f (the top-k AF2Rank step never forwards it; its downstream
#      default is off). CUEQ_MULT=1 reproduces the published (biased) kernel set only.
#   3. AF2Rank query rule (QFD, maps to PROTEINA_AF2RANK_QUERY_FROM_DATASET):
#      QFD=0 (default) = locked protocol: AF2 query = the native's resolved residues. QFD=0 also clears a
#      PROTEINA_AF2RANK_QUERY_FROM_DATASET inherited from the submit shell.
#      QFD=1 ONLY for targets whose QFD=0 af2rank step crashed (e.g. IndexError at af2rank_openfold_scorer.py:462
#      when the native has unresolved residues; the pipeline can still exit 0, so check every target's
#      af2rank_scores CSV). Resubmit those targets alone (CSV with only them, same OUT) and record the list.
#   4. CIF_DIR is a variable (default = the D36f dir) because the v3 natives live in a separate symlink dir.
# CSV and OUT are required (no defaults). PROTEINA_INFERENCE_PRECISION is not set here, as in the locked runs.
# ~/proteina must stay on d36aq-c8 for the whole run: AF2Rank subprocesses import the checked-out code hours
# after start, so never switch the checkout (e.g. to d36w8-plainkernel-proteina for the W8 re-score) while an
# S4 job is queued or running.
set -o pipefail
source /etc/profile
module load conda/Python-ML-2025b-pytorch cuda/12.6
source activate cathfold
export OMP_NUM_THREADS=20
export PATH="$HOME/.local/bin:$PATH"          # USalign, or every tm_ref_* silently reads 0
export DATA_PATH=/home/gridsan/cou/proteina/data
export PROTEINA_CONDITIONING_MODE=seq
export PROTEINA_NSAMPLES_PER_PROTEIN=${NSAMP:-1024}
cd /home/gridsan/cou/proteina
C=ca_dssp_extlig_no-sin-pos-emb_beta-1p3-2_finetune-all_v1p6_default-fold-sum_2-seq-S25_16-eff-bs_200-epoch_pdb_purge-test_warmup_maxlen-256_cutoff-190828_unified_v3
CSV=${CSV:?set CSV}
OUT=${OUT:?set OUT}
CUEQ_MULT=${CUEQ_MULT:-0}
QFD=${QFD:-0}
CIF_DIR=${CIF_DIR:-/home/gridsan/cou/data/bad_afdb/pdb}

# locked inference.py (43d5802 blob) must be what runs
INF_BLOB=$(git rev-parse HEAD:proteinfoundation/inference.py)
[ "$INF_BLOB" = 4e7a8d83318bfca1e812e667f9b3284ba349ed83 ] || { echo "inference.py blob $INF_BLOB != 43d5802's"; exit 2; }
git diff --quiet HEAD -- proteinfoundation || { echo "uncommitted changes under proteinfoundation/"; exit 2; }
git merge-base --is-ancestor 2fde18e HEAD || { echo "HEAD does not contain d36aq-c8 2fde18e"; exit 2; }

case "$CUEQ_MULT" in
  0) KFLAGS=(--use_cuequivariance_attention) ;;
  1) KFLAGS=(--use_cuequivariance_attention --use_cuequivariance_multiplicative_update) ;;
  *) echo "CUEQ_MULT must be 0 or 1, got $CUEQ_MULT"; exit 2 ;;
esac
case "$QFD" in
  0) unset PROTEINA_AF2RANK_QUERY_FROM_DATASET ;;
  1) export PROTEINA_AF2RANK_QUERY_FROM_DATASET=1 ;;
  *) echo "QFD must be 0 or 1, got $QFD"; exit 2 ;;
esac

echo "HOST=$(hostname) HEAD=$(git rev-parse --short HEAD) CSV=$CSV NSAMP=$PROTEINA_NSAMPLES_PER_PROTEIN OUT=$OUT CIF_DIR=$CIF_DIR CUEQ_MULT=$CUEQ_MULT QFD=$QFD KFLAGS=${KFLAGS[*]} START=$(date)"
python proteinfoundation/prediction_pipeline/run_prediction_pipeline.py \
    --dataset_file "$CSV" --id_col pdb --tms_col tm_score \
    --cif_dir "$CIF_DIR" \
    --output_dir "$OUT" \
    --inference_config $C \
    --segment_mode joint \
    --segment_min_len 30 \
    --no-mask_inter_segment \
    --scorer proteinebm \
    --proteinebm_checkpoint /home/gridsan/cou/ProteinEBM/weights/pae.ckpt \
    --proteinebm_config /home/gridsan/cou/ProteinEBM/protein_ebm/config/pae_config.yaml \
    --proteinebm_t 0.05 \
    --skip_diversity \
    --mask_sidechains \
    --af2rank_subdir af2rank_uniform_rc6 \
    --af2rank_top_k 5 \
    --recycles 6 \
    --af2rank_backend openfold \
    --no-use_deepspeed_evoformer_attention \
    "${KFLAGS[@]}" \
    --proteinebm_batch_size 16 \
    --proteina_force_compile \
    --direct_python --num_gpus 2
rc=$?
echo "D36S4_OURS_EXIT=$rc END=$(date)"
exit $rc
