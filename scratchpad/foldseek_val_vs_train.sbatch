#!/bin/bash
#SBATCH --job-name=fsvaltrain
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

V=/orcd/scratch/orcd/011/chenxiou/valset_analysis
W=$V/foldseek_val_vs_train
RAW=/orcd/pool/006/chenxiou/proteina/data/pdb_train/raw
FSROOT=/orcd/scratch/orcd/011/chenxiou/proteina
mkdir -p "$W"

# ⛔ TMP IS NEVER DELETED. foldseek resumes from an intact tmp dir by rerunning the IDENTICAL
# command; wiping it to "restart clean" throws away all prior progress. This job is --requeue on a
# preemptable partition, so resume is the expected path, not the exception.
TMP=$W/tmp
mkdir -p "$TMP"

FS=$FSROOT/foldseek/bin/foldseek
if [ ! -x "$FS" ]; then
  echo "[setup] extracting foldseek"
  tar -xzf $FSROOT/foldseek-linux-avx2.tar.gz -C $FSROOT
fi
"$FS" version || { echo "foldseek unavailable"; exit 1; }

# ── query: the 32 fixed validation chains ────────────────────────────────────────────────────
if [ ! -e "$W/queryDB" ]; then
  mkdir -p "$W/query_pdbs"
  n_missing=0
  while read -r id; do
    [ -z "$id" ] && continue
    if [ -f "$V/pdbs/$id.pdb" ]; then cp "$V/pdbs/$id.pdb" "$W/query_pdbs/"; else
      echo "  MISSING query structure: $id"; n_missing=$((n_missing + 1)); fi
  done < "$V/val_fixed32_max256.txt"
  echo "[query] $(ls "$W/query_pdbs" | wc -l) structures, $n_missing missing"
  "$FS" createdb "$W/query_pdbs" "$W/queryDB"
fi

# ── target: the mmCIF entries backing the TRAINING chains ────────────────────────────────────
# foldseek emits ONE ENTRY PER CHAIN from an mmCIF, named <entry>_<chain>, matching the split's
# convention -- so no manual chain extraction. PDB shards by characters 2-3 of the ID.
if [ ! -e "$W/targetDB" ]; then
  cut -d_ -f1 "$V/splits/train_chain_ids.txt" | sort -u > "$W/train_entries.txt"
  : > "$W/train_cif_paths.txt"
  n_have=0; n_gone=0
  while read -r e; do
    p="$RAW/${e:1:2}/${e}.cif.gz"
    if [ -f "$p" ]; then echo "$p" >> "$W/train_cif_paths.txt"; n_have=$((n_have + 1));
    else n_gone=$((n_gone + 1)); fi
  done < "$W/train_entries.txt"
  echo "[target] entries: $n_have present, $n_gone absent"
  "$FS" createdb "$W/train_cif_paths.txt" "$W/targetDB"
fi

# ── search: TM mode ──────────────────────────────────────────────────────────────────────────
# ⛔ --alignment-type 1 (TM). The 3Di+AA default under-reports remote structural hits, which here
# would UNDERSTATE leakage -- the opposite of a conservative error.
"$FS" search "$W/queryDB" "$W/targetDB" "$W/resDB" "$TMP" \
    --alignment-type 1 -e 10 --max-seqs 4000 --threads "${SLURM_CPUS_PER_TASK:-16}"
"$FS" convertalis "$W/queryDB" "$W/targetDB" "$W/resDB" "$W/hits.tsv" \
    --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen,qlen,tlen,evalue"

echo "[hits] $(wc -l < "$W/hits.tsv") rows"

# A silent ID-convention mismatch would look exactly like "no leakage", so the overlap between the
# hit targets and the training list is checked LOUDLY before any conclusion is drawn.
cut -f2 "$W/hits.tsv" | sed 's/\.pdb$//;s/\.cif$//' | sort -u > "$W/hit_targets.txt"
sort -u "$V/splits/train_chain_ids.txt" > "$W/train_sorted.txt"
echo "[sanity] hit targets: $(wc -l < "$W/hit_targets.txt")"
echo "[sanity] of which in the training list: $(comm -12 "$W/hit_targets.txt" "$W/train_sorted.txt" | wc -l)"
echo "  ^ if this is ~0 the ID conventions disagree and the result means NOTHING"
