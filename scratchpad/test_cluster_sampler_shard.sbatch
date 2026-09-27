#!/bin/bash
#SBATCH --job-name=test_cs_shard
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/chenxiou/qlogs/test_cs_shard_%j.out
# CPU 2-process gloo test of the ClusterSampler shard flag in the real training env.
set -uo pipefail
REPO="${REPO:-/orcd/scratch/orcd/011/chenxiou/proteina_dev}"
cd "$REPO"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=
OLD=$(mktemp --suffix=.py)
git show "${OLD_REF:-HEAD~1}":proteinfoundation/utils/cluster_utils.py > "$OLD"
git log -1 --format="[repo] %h" ; echo "[old] ${OLD_REF:-HEAD~1}"
python scratchpad/test_cluster_sampler_shard.py "$OLD"
rc=$?
rm -f "$OLD"
echo "TEST_EXIT=$rc"
exit $rc
