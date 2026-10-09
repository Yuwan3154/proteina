#!/bin/bash
#SBATCH --job-name=d36tc_index
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/006/chenxiou/d36tc/logs/%x-%j.out
# D36tc S1b (CPU, cue_openfold): the D36tc topology index (34 targets + 31 hits, natives-only, cfft stats) and
# CONTROL A2/A3 on the 3 training chains (index rows from the STORED .pt and from MY .pt vs the training index).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
P=/orcd/pool/006/chenxiou/d36tc
REPO="${REPO:-$S/proteina_d36tc}"
D=$P/data
I=$P/inputs
STATS=$S/synth_index_confind_v1/topology_index_confind_synth.pt
module load miniforge
set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; conda deactivate 2>/dev/null || true; conda activate cue_openfold; set -u
export DATA_PATH=/orcd/pool/006/chenxiou/proteina/data PYTHONPATH=$REPO PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
cd "$REPO"; git log -1 --format="[repo] $REPO HEAD %h"
[ ! -e "$D/topology_index_d36tc.pt" ] || { echo "FATAL: index exists"; exit 78; }

python - "$I/d36tc_rows.csv" "$I/d36tc_index_stems.txt" <<'PY'
import csv, sys
st = [r["id"] for r in csv.DictReader(open(sys.argv[1])) if r["role"] in ("target", "reference")]
assert len(st) == 65 == len(set(st)), f"index stems {len(st)} (want 65 distinct = 34 targets + 31 hits)"
open(sys.argv[2], "w").write("\n".join(st) + "\n"); print(f"index stems {len(st)}")
PY
src=$?
[ "$src" -eq 0 ] || { echo "FATAL: index stem list rc=$src"; exit "$src"; }
python scratchpad/d36tc_build_index.py --processed-dir "$D/processed" --stems "$I/d36tc_index_stems.txt" \
  --stats-from "$STATS" --out "$D/topology_index_d36tc.pt" --workers 8
irc=$?
echo "INDEX_RC=$irc"

# D2b record: helix/strand element count per row vs the 64 cap (user: keep the cap, record exceeders)
python - "$D/topology_index_d36tc.pt" "$P/he_counts.tsv" <<'PY'
import sys, torch
idx = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
out = []
for i, s in enumerate(idx["ids"]):
    a, b = int(idx["runs_offset"][i]), int(idx["runs_offset"][i + 1])
    runs = idx["runs_flat"][a:b].tolist()
    n_he = sum(1 for t, n in runs if t in (1, 2))
    out.append((s, int(idx["he_size"][i]), n_he, sum(n for _, n in runs)))
open(sys.argv[2], "w").write("stem\the_size\tn_helix_strand_runs\tL\n" + "\n".join("\t".join(map(str, r)) for r in out) + "\n")
ex = [r for r in out if r[1] > 64]
print(f"[D2b] rows {len(out)}; empty runs {sum(1 for r in out if r[3] == 0)}; he_size > 64 (cap truncates): {len(ex)} {ex}")
assert len(out) == 65, f"index rows {len(out)}"
PY
hrc=$?
echo "HE_COUNTS_RC=$hrc"
[ "$hrc" -eq 0 ] || { echo "FATAL: he_counts rc=$hrc"; exit "$hrc"; }

# CONTROL A2/A3
python scratchpad/d36tc_build_index.py --processed-dir "$I/controlA_stored_paths.tsv" --stems "$I/controlA_stems.txt" \
  --stats-from "$STATS" --out "$P/controlA_idx_stored.pt" --workers 3 &&
python scratchpad/d36tc_build_index.py --processed-dir "$D/processed" --stems "$I/controlA_stems.txt" \
  --stats-from "$STATS" --out "$P/controlA_idx_mine.pt" --workers 3 &&
python scratchpad/d36tc_controlA.py --part a23 --stems "$I/controlA_stems.txt" --mine "$D/processed" \
  --idx-stored "$P/controlA_idx_stored.pt" --idx-mine "$P/controlA_idx_mine.pt"
arc=$?
echo "CONTROL_A23_RC=$arc"
[ "$irc" -ne 0 ] && exit "$irc"
exit "$arc"
