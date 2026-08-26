#!/bin/bash
#SBATCH --job-name=fsscale
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

RAW=/orcd/pool/006/chenxiou/proteina/data/pdb_train/raw
FS=/orcd/scratch/orcd/011/chenxiou/proteina/foldseek/bin/foldseek
D=/orcd/scratch/orcd/011/chenxiou/valset_analysis/fsscale
rm -rf "$D"; mkdir -p "$D"

# The earlier diagnostic (21251300) proved ONE directory holding 3 files works. The production
# call passes ~1296 shard directories AT ONCE and produced 0 entries. Those are different
# invocations, so this separates the candidate causes instead of guessing again:
#   real_shard  -- a raw/ shard exactly as it sits on disk (may hold nested dirs or stray files)
#   many_files  -- one directory with many structures (is it a per-directory file-count limit?)
#   few_dirs    -- a handful of directory arguments (does multi-dir input work at all?)
#   many_dirs   -- every shard, i.e. the production call (is it an argument-count/scale limit?)
probe () {
  label=$1; shift
  out="$D/db_$label"
  echo "=================== $label  (args: $#)"
  "$FS" createdb "$@" "$out" > "$D/log_$label.txt" 2>&1
  rc=$?
  n=$(wc -l < "${out}.index" 2>/dev/null || echo 0)
  echo ">>> $label : rc=$rc  entries=$n"
  [ "$n" -eq 0 ] && tail -4 "$D/log_$label.txt"
  return 0
}

echo "### shard 00 contents (is it flat .cif.gz, or something else?)"
ls -la "$RAW/00" | head -5
echo "### file count in shard 00: $(ls "$RAW/00" | wc -l)"
echo "### total shard dirs: $(ls -d "$RAW"/*/ | wc -l)"

probe real_shard "$RAW/00"

mkdir -p "$D/many"
find "$RAW/00" "$RAW/01" -name '*.cif.gz' | head -200 | while read -r f; do cp "$f" "$D/many/"; done
echo "### many_files dir holds $(ls "$D/many" | wc -l) files"
probe many_files "$D/many"

probe few_dirs $(ls -d "$RAW"/0*/ | head -5)
probe many_dirs $(ls -d "$RAW"/*/)

echo "=================== SUMMARY"
for v in real_shard many_files few_dirs many_dirs; do
  echo "$v: $(wc -l < "$D/db_$v.index" 2>/dev/null || echo 0) entries"
done
