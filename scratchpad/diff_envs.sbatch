#!/bin/bash
#SBATCH --job-name=envdiff
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=14:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
# Must precede set -u: Lmod's init references unset variables.
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

OUT=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/envdiff
mkdir -p "$OUT"

module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold
set -u
LISTER='
import importlib.metadata as md
rows = set()
for d in md.distributions():
    n = d.metadata["Name"]
    if n:
        rows.add("%s==%s" % (n, d.version))
print("\n".join(sorted(rows)))
'
python -c "$LISTER" | sort > "$OUT/cu126_conda.txt"
conda deactivate

/home/chenxiou/cue-openfold-env-sm120/.venv/bin/python -c "$LISTER" | sort > "$OUT/cu128_uv.txt"

echo "cu126 conda packages: $(wc -l < "$OUT/cu126_conda.txt")"
echo "cu128 uv    packages: $(wc -l < "$OUT/cu128_uv.txt")"
echo
echo "=== PRESENT IN WORKING cu126 ENV, MISSING ENTIRELY FROM cu128 ==="
comm -23 <(cut -d= -f1 "$OUT/cu126_conda.txt" | tr 'A-Z_' 'a-z-' | sort -u) \
         <(cut -d= -f1 "$OUT/cu128_uv.txt"    | tr 'A-Z_' 'a-z-' | sort -u)
echo
echo "=== VERSION MISMATCHES (name: cu126_working -> cu128_new) ==="
python3 - "$OUT/cu126_conda.txt" "$OUT/cu128_uv.txt" <<'PY'
import sys

def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if "==" in line:
            n, v = line.split("==", 1)
            d[n.lower().replace("_", "-")] = v
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
for name in sorted(set(a) & set(b)):
    if a[name] != b[name]:
        print("  %-34s %-22s -> %s" % (name, a[name], b[name]))
PY
