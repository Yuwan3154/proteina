#!/bin/bash
#SBATCH --job-name=fsdiag
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

RAW=/orcd/pool/006/chenxiou/proteina/data/pdb_train/raw
FSROOT=/orcd/scratch/orcd/011/chenxiou/proteina
FS=$FSROOT/foldseek/bin/foldseek
D=/orcd/scratch/orcd/011/chenxiou/valset_analysis/fsdiag
rm -rf "$D"; mkdir -p "$D/gz" "$D/plain"

# Four variants, to separate "cannot read gzipped mmCIF" from "cannot take a path list".
# Both are plausible and they need different fixes, so guessing would risk another 254k-entry
# build that yields nothing.
for e in 4jgo 200l 101m; do   # 4jgo is multi-chain: 4jgo_A and 4jgo_B are both in the train split
  p="$RAW/${e:1:2}/${e}.cif.gz"
  [ -f "$p" ] || { echo "missing $p"; continue; }
  cp "$p" "$D/gz/"
  gunzip -c "$p" > "$D/plain/${e}.cif"
done
ls -l "$D/gz" "$D/plain"

ls "$D"/gz/*.cif.gz > "$D/gz_list.txt"
ls "$D"/plain/*.cif  > "$D/plain_list.txt"

probe () {   # $1 = label, $2 = input argument
  out="$D/db_$1"
  echo "=================== $1 : $2"
  "$FS" createdb "$2" "$out" 2>&1 | tail -3
  n=$(wc -l < "${out}.index" 2>/dev/null || echo 0)
  echo ">>> $1 : ${n} entries"
  if [ "$n" -gt 0 ]; then echo ">>> $1 : first ids: $(cut -f2 "${out}.lookup" 2>/dev/null | head -4 | tr '\n' ' ')"; fi
}

probe gz_dir    "$D/gz"
probe gz_list   "$D/gz_list.txt"
probe plain_dir "$D/plain"
probe plain_list "$D/plain_list.txt"

echo "=================== SUMMARY"
for v in gz_dir gz_list plain_dir plain_list; do
  echo "$v: $(wc -l < "$D/db_$v.index" 2>/dev/null || echo 0) entries"
done
echo "Per-chain naming is what the val-vs-train filter depends on -- check the 'first ids' lines"
echo "above show <entry>_<chain>, not bare <entry>."
