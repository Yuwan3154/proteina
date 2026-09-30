#!/bin/bash
#SBATCH --job-name=t7_quick_setup
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/chenxiou/qlogs/t7_quick_setup_%j.out
# T7-QUICK (user 2026-09-30): L<=256 fold-clustered held-out subset + chunk lists, and freeze each tri's BEST-by-sampling
# EMA checkpoint (top of its own validation_sampling/contact_precision_at_L_median tracker in last.ckpt).
set -uo pipefail
S=/orcd/scratch/orcd/011/chenxiou
REPO="${REPO:-$S/proteina_t7}"
cd "$REPO"
source /home/chenxiou/cue-openfold-env-sm120/.venv/bin/activate
export PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=
git log -1 --format="[repo] %h"
python scratchpad/t7_subset.py scratchpad/stageA/stageA_run_union.txt $S/stageA/new_nonself/samples.jsonl $S/t7/quick \
    --lmax 256 --k 8 --chunk 25 --procs 8 || exit 3
python - <<'PY' || exit 4
import os, shutil, torch
S = "/orcd/scratch/orcd/011/chenxiou"
runs = {"new": f"{S}/proteina_tri/store/tri_cb8synth_v5/checkpoints", "cfft": f"{S}/proteina_cfft/store/tri_confindsynth_ft/checkpoints"}
for tag, d in runs.items():
    c = torch.load(f"{d}/last.ckpt", map_location="cpu", weights_only=False, mmap=True)
    cbs = [v for k, v in c["callbacks"].items() if "contact_precision_at_L_median" in k]
    assert len(cbs) == 1, f"{tag}: {len(cbs)} P@L trackers"
    bk = {p: float(s) for p, s in cbs[0]["best_k_models"].items()}
    for p, s in sorted(bk.items(), key=lambda t: -t[1]):
        print(f"[{tag}] top-k {s:.4f} {os.path.basename(p)} exists={os.path.exists(p)}")
    best = max(bk, key=bk.get)
    ema = best.replace(".ckpt", "-EMA.ckpt")
    assert os.path.exists(ema), f"{tag}: no EMA sibling {ema}"
    gs = int(torch.load(ema, map_location="cpu", weights_only=False, mmap=True)["global_step"])
    dst = f"{S}/stageA/ckpt/{tag}_tri_best-EMA_step{gs}.ckpt"
    if not os.path.exists(dst):
        shutil.copy2(ema, dst)
    print(f"[{tag}] BEST score {bk[best]:.4f} step {gs} -> {dst}")
PY
ls -la $S/t7/quick | head -20; cat $S/t7/quick/clusters.tsv | head -12
echo T7_QUICK_SETUP_DONE
