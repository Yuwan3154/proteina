#!/bin/bash
#SBATCH --job-name=refavail
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/c2c_store/logs/%x-%j.err
# CPU-only, read-only on training data. Picks the leak-free chain list for the Directive-B tri ablation:
#  (1) per-chain reference availability for BOTH tri models (old: retrieved non-self mate from the max512
#      topology_index.pt; new: synthetic template) over every max384 val + test chain and the named lists;
#  (2) sequence-level leak check the S25 cluster split cannot rule out (mmseqs is not transitive).
# ⛔ No dm.setup()/dump_splits here: prepare_data can rewrite the shared csv and invalidate the cluster
#    TSV that the live tri/tbeta segments depend on. The split is read from the 2026-08-25 dump, whose
#    identity with every run's split is already evidenced by identical logged counts.
source /etc/profile >/dev/null 2>&1
set -uo pipefail
B=/orcd/scratch/orcd/011/chenxiou
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data SYNTH_INDEX_DIR=$B/synth_index_v4 SKIP_CONFIND_PRECOMPUTE=1
D=$DATA_PATH/pdb_train
ID=df_pdb_f1_minl50_maxl384_mtprotein_etdiffractionEM_minoNone_maxoNone_minr0.0_maxr5.0_hl_rl_rnsrTrue_rpuTrue_l_rcuFalse
V=$B/valset_analysis
OUT=$V/leakage_${SLURM_JOB_ID}
mkdir -p "$OUT"
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null; conda activate cue_openfold; set -u
HERE=$B/proteina_tri/scratchpad
STEMS="$V/splits/val_chain_ids.txt $V/splits/test_chain_ids.txt $V/val_fixed32_max256.txt $V/val_fixed_nonself.txt $B/novel_fold_val_chains.txt"
python $HERE/ref_availability.py --repo $B/proteina_cmhier --mode old --index $D/topology_index.pt \
    --csv $D/$ID.csv --splits_dir $V/splits --stems $STEMS --out "$OUT/ref_old.tsv"; echo "OLD_RC=$?"
python $HERE/ref_availability.py --repo $B/proteina_tri --mode new \
    --index $SYNTH_INDEX_DIR/topology_index_cb8_synth.pt --csv $D/$ID.csv --splits_dir $V/splits --stems $STEMS \
    --out "$OUT/ref_new.tsv"; echo "NEW_RC=$?"

FA=$D/seq_$ID.fasta
awk 'NR==FNR{k[">"$1]=1;next} /^>/{p=($1 in k)} p' <(cat $V/splits/val_chain_ids.txt $V/splits/test_chain_ids.txt) $FA > "$OUT/q.fa"
awk 'NR==FNR{k[">"$1]=1;next} /^>/{p=($1 in k)} p' $V/splits/train_chain_ids.txt $FA > "$OUT/t.fa"
echo "[fasta] query $(grep -c '>' "$OUT/q.fa") (expect 4331)  target $(grep -c '>' "$OUT/t.fa") (expect 254946)"
mmseqs easy-search "$OUT/q.fa" "$OUT/t.fa" "$OUT/hits_all.m8" "$B/.tmp/mmtmp_${SLURM_JOB_ID}" --min-seq-id 0.25 -e 10 \
    --threads 16 --format-output query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue; echo "MMSEQS_RC=$?"
rm -rf "$B/.tmp/mmtmp_${SLURM_JOB_ID}"
awk '$3>=0.25 && ($7>=0.8 || $8>=0.8)' "$OUT/hits_all.m8" > "$OUT/hits_clusterlike.m8"
echo "[mmseqs] val/test chains with ANY >=25% hit: $(cut -f1 "$OUT/hits_all.m8" | sort -u | wc -l)"
echo "[mmseqs] ... at id>=0.25 AND (qcov>=0.8 OR tcov>=0.8): $(cut -f1 "$OUT/hits_clusterlike.m8" | sort -u | wc -l)"
echo "[done] $OUT"
