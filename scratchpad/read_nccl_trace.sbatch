#!/bin/bash
#SBATCH --job-name=fr_read
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err
source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail
module load miniforge
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cue_openfold
set -u

python - <<'PY'
import pickle

DIR = "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/nccl_trace"
per_rank = {}
for r in (0, 1):
    with open(f"{DIR}/rank_{r}", "rb") as fh:
        per_rank[r] = pickle.load(fh)

for r, d in per_rank.items():
    ents = d.get("entries", d) if isinstance(d, dict) else d
    print(f"===== rank {r}: {len(ents)} entries =====")
    for e in ents[-6:]:
        print("  seq=%s  %s  state=%s  in=%s out=%s" % (
            e.get("collective_seq_id"), e.get("profiling_name"), e.get("state"),
            e.get("input_sizes"), e.get("output_sizes")))

# The point of the exercise: find the first sequence id where the two ranks disagree
# on WHICH collective was issued. Identical seq NUMBERS do not imply identical ops.
def by_seq(d):
    ents = d.get("entries", d) if isinstance(d, dict) else d
    out = {}
    for e in ents:
        out.setdefault(e.get("collective_seq_id"), e)
    return out

a, b = by_seq(per_rank[0]), by_seq(per_rank[1])
print("\n===== first divergence =====")
found = False
for s in sorted(set(a) | set(b), key=lambda x: (x is None, x)):
    ea, eb = a.get(s), b.get(s)
    na = ea.get("profiling_name") if ea else None
    nb = eb.get("profiling_name") if eb else None
    sa = ea.get("input_sizes") if ea else None
    sb = eb.get("input_sizes") if eb else None
    if na != nb or sa != sb:
        print(f"  seq {s}: rank0 = {na} {sa}   |   rank1 = {nb} {sb}")
        found = True
        break
if not found:
    print("  ranks agree on every recorded collective -- divergence is NOT in the recorded window")

print("\n===== incomplete collectives (the actual hang) =====")
for r, d in per_rank.items():
    ents = d.get("entries", d) if isinstance(d, dict) else d
    for e in ents:
        if e.get("state") != "completed":
            print("  rank %d seq=%s %s state=%s in=%s" % (
                r, e.get("collective_seq_id"), e.get("profiling_name"),
                e.get("state"), e.get("input_sizes")))
PY
