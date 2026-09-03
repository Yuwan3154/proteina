#!/bin/bash
#SBATCH --job-name=tri_diag
#SBATCH --partition=mit_quicktest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/logs/%x-%j.err

source /etc/profile >/dev/null 2>&1 || true
set -uo pipefail

REPO=/orcd/scratch/orcd/011/chenxiou/proteina_cmhier
ENV_DIR=/home/chenxiou/cue-openfold-env-sm120
CKPT_DIR=$REPO/store/tri_full384/checkpoints
PROBE=/orcd/scratch/orcd/011/chenxiou/_headroom_probe
trap 'rm -rf "$PROBE"' EXIT

echo "=== [1] mount identity ==="
stat -f -c '%n  blocks=%b free=%f avail=%a bsize=%S' "$REPO/wandb" "$REPO/store" /orcd/scratch/orcd/011

echo
echo "=== [2] checkpoint load verification (no try/except: a failure must crash this job) ==="
source "$ENV_DIR/.venv/bin/activate"
python - <<'EOPY'
import torch, os
ckpts = {
    "last.ckpt":     "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/store/tri_full384/checkpoints/last.ckpt",
    "last-EMA.ckpt": "/orcd/scratch/orcd/011/chenxiou/proteina_cmhier/store/tri_full384/checkpoints/last-EMA.ckpt",
}
for name, path in ckpts.items():
    obj = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(obj.keys())
    gs = obj.get("global_step")
    ep = obj.get("epoch")
    n_sd = len(obj.get("state_dict", {}))
    n_ema = len(obj.get("ema", {}).get("params", [])) if isinstance(obj.get("ema"), dict) else "n/a"
    has_opt = "optimizer_states" in obj and bool(obj["optimizer_states"])
    print(f"{name}: bytes={os.path.getsize(path)} global_step={gs} epoch={ep} "
          f"n_state_dict={n_sd} n_ema_params={n_ema} optimizer_states={has_opt}")
    print(f"    top-level keys: {keys}")
EOPY

echo
echo "=== [3] scratch write headroom (1G chunks, hard cap 40G, stops at first EDQUOT) ==="
mkdir -p "$PROBE"
WROTE=0
for i in $(seq 1 40); do
    if dd if=/dev/zero of="$PROBE/chunk_$i.bin" bs=1M count=1024 status=none 2>"$PROBE/err_$i"; then
        WROTE=$i
    else
        echo "STOPPED at chunk $i:"; cat "$PROBE/err_$i"; break
    fi
done
echo "HEADROOM_AT_LEAST_GiB=$WROTE (cap 40)"

echo
echo "=== [4] inode probe: 5000 small files ==="
mkdir -p "$PROBE/inodes"
MADE=0
for i in $(seq 1 5000); do
    echo x > "$PROBE/inodes/f_$i" 2>"$PROBE/ierr" || { echo "INODE STOP at $i:"; cat "$PROBE/ierr"; break; }
    MADE=$i
done
echo "SMALL_FILES_CREATED=$MADE (cap 5000)"

echo
echo "=== [5] wandb dir file count ==="
find "$REPO/wandb" -type f 2>/dev/null | wc -l
du -sh "$REPO/wandb" 2>/dev/null

echo "=== DONE ==="
