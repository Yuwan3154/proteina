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
  # ⛔ MEASURED (diag job 21237409): this foldseek build reads gzipped mmCIF fine, but a
  # newline-delimited PATH LIST yields 0 entries -- gz_dir 3, gz_list 0, plain_dir 3, plain_list 0.
  # Directories work, so pass the raw shard dirs as arguments. Symlinking 254k files into one
  # directory is not an option: it breaks the <=1024-files-per-directory rule.
  # This admits every entry in raw/, not only training ones, so hits are filtered against the
  # training chain list AFTER the search -- which the sanity block below already does.
  "$FS" createdb "$RAW"/*/ "$W/targetDB"
  n_tgt=$("$FS" createdb --help >/dev/null 2>&1; wc -l < "$W/targetDB.index" 2>/dev/null || echo 0)
  echo "[target] targetDB entries: $n_tgt"
  if [ "$n_tgt" -lt 1000 ]; then
    echo "!! targetDB has $n_tgt entries from $n_have mmCIF paths -- the build did not work."
    echo "!! Candidates: this foldseek build may not read gzipped mmCIF, or may not accept a"
    echo "!! newline-delimited path list as input. Diagnose before rerunning the search."
    exit 4
  fi
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
# ⛔ foldseek names an entry <entry>_<chain> ONLY when it has MULTIPLE chains; single-chain
# structures keep the BARE entry id (measured, job 21251300: "101m 200l 4jgo_A 4jgo_B"). The split
# uses <entry>_<chain> for EVERY chain, so exact-string matching would silently drop every
# single-chain training structure -- UNDER-reporting leakage, biased toward single-chain proteins,
# which is the wrong direction to err on this question. Match both ways and report both.
cut -f2 "$W/hits.tsv" | sed 's/\.pdb$//;s/\.cif$//;s/\.gz$//' | sort -u > "$W/hit_targets.txt"
sort -u "$V/splits/train_chain_ids.txt" > "$W/train_sorted.txt"
cut -d_ -f1 "$W/train_sorted.txt" | sort -u > "$W/train_entries_sorted.txt"
cut -d_ -f1 "$W/hit_targets.txt"   | sort -u > "$W/hit_entries.txt"
n_exact=$(comm -12 "$W/hit_targets.txt" "$W/train_sorted.txt" | wc -l)
n_entry=$(comm -12 "$W/hit_entries.txt" "$W/train_entries_sorted.txt" | wc -l)
echo "[sanity] hit targets: $(wc -l < "$W/hit_targets.txt")"
echo "[match] exact chain-level: $n_exact"
echo "[match] entry-level (permissive => CONSERVATIVE for a leakage audit): $n_entry"
n_match=$n_entry
if [ "$n_match" -eq 0 ]; then
  echo "!! ZERO training hits. Either the target DB is empty or the ID conventions disagree."
  echo "!! This is NOT evidence of no leakage. Failing so it cannot be read as a result."
  exit 3
fi
