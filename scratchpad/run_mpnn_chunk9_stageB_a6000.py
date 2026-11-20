"""4-GPU Stage-B c2c 3D structure folding launcher on A6000_offsite for mpnn_chunk9.

Folds all 20 Stage-A contact map chunks (mpnn_t7_c0..3 first, then mpnn_equal_c0..7, then mpnn_self_c0..7)
using ConFind twin final_twin_s29000_ema_only.ckpt (20 steps, SIGMA_DATA=16.0) across all 4 RTX A6000 GPUs.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from scratchpad.run_mpnn_chunk9_eval_a6000 import T7_CHANGED_CHAINS, boot_ci


def worker_loop(gpu_id, tasks, ckpt, dataset, out_dir, usalign, python_bin):
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["SIGMA_DATA"] = "16.0"
    env["SKIP_CONFIND_PRECOMPUTE"] = "1"
    env["PYTHONPATH"] = REPO
    env.setdefault("DATA_PATH", "/home/jupyter-chenxi/sse1d_data/orcd/pool/006/chenxiou/proteina/data")
    env.setdefault("PACK_PATH", "/home/jupyter-chenxi/sse1d_data/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack")

    sb_dir = os.path.join(out_dir, "stageB")
    os.makedirs(sb_dir, exist_ok=True)

    for label, chains_file, maps_dir in tasks:
        out_jsonl = os.path.join(sb_dir, f"{label}.jsonl")
        if os.path.exists(out_jsonl) and os.path.getsize(out_jsonl) > 0:
            n_stems = len({l.split()[0] for l in open(chains_file) if l.strip()})
            n_rows = sum(1 for l in open(out_jsonl) if l.strip())
            if n_rows == n_stems * 8:
                print(f"[GPU {gpu_id}] SKIP {label} (already complete: {n_rows} rows)", flush=True)
                continue
            os.remove(out_jsonl)

        t0 = time.time()
        print(f"[GPU {gpu_id}] START Stage-B c2c {label} -> {out_jsonl}", flush=True)
        cmd = [
            python_bin,
            os.path.join(HERE, "stageB_c2c_from_maps.py"),
            "--ckpt", ckpt,
            "--weights", "ema",
            "--dataset", dataset,
            "--chains", chains_file,
            "--maps_dir", maps_dir,
            "--out", out_jsonl,
            "--label", label,
            "--usalign", usalign,
            "--record_gt_mismatch",
        ]
        res = subprocess.run(cmd, env=env)
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"[GPU {gpu_id}] FAIL {label} rc={res.returncode} after {dt:.1f}s", flush=True)
            sys.exit(res.returncode)
        rows = [json.loads(l) for l in open(out_jsonl) if l.strip()]
        tms = [float(r["tm"]) for r in rows]
        print(
            f"[GPU {gpu_id}] DONE {label}: {len(rows)} structures in {dt:.1f}s "
            f"| c2c TM mean={np.mean(tms):.4f} median={np.median(tms):.4f}",
            flush=True,
        )


def score_stageB(out_dir):
    sb_dir = os.path.join(out_dir, "stageB")
    lines = []
    for arm, n_chunks in (("t7", 4), ("equal", 8), ("self", 8)):
        per_chain = defaultdict(list)
        for c in range(n_chunks):
            p = os.path.join(sb_dir, f"mpnn_{arm}_c{c}.jsonl")
            if not os.path.exists(p):
                continue
            for l in open(p):
                if l.strip():
                    r = json.loads(l)
                    per_chain[r["stem"]].append(float(r["tm"]))
        stems_all = sorted(per_chain.keys())
        if not stems_all:
            continue
        subsets = [("all", stems_all)]
        if arm == "t7":
            clean = [s for s in stems_all if s not in T7_CHANGED_CHAINS]
            subsets = [("clean_70", clean), ("all_98", stems_all)]
        for sub_name, stems in subsets:
            rng = np.random.default_rng(0)
            means = np.array([np.mean(per_chain[s]) for s in stems])
            all_tm = np.concatenate([per_chain[s] for s in stems])
            mu, lo, hi = boot_ci(means, rng)
            med = float(np.median(means))
            lines.append(
                f"=== Stage-B c2c ARM: {arm} ({sub_name}, chains={len(stems)}, samples={len(all_tm)}) ===\n"
                f"  mean TM = {mu:.4f} [{lo:.4f}, {hi:.4f}]  median_of_means = {med:.4f}\n"
                f"  chains mean TM >= 0.5: {int((means >= 0.5).sum())}/{len(stems)} ({100.0 * (means >= 0.5).mean():.1f}%)\n"
                f"  samples TM >= 0.5:     {int((all_tm >= 0.5).sum())}/{len(all_tm)} ({100.0 * (all_tm >= 0.5).mean():.1f}%)"
            )
    text = "\n\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "stageB_summary.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/mpnn_chunk9_eval")
    ap.add_argument("--ckpt", default="/home/jupyter-chenxi/mpnn_chunk9_eval/ckpt/final_twin_s29000_ema_only.ckpt")
    ap.add_argument("--dataset", default="pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--usalign", default="/home/jupyter-chenxi/.local/bin/USalign")
    ap.add_argument("--python_bin", default="/home/jupyter-chenxi/miniconda3/envs/proteina_sse1d/bin/python")
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        score_stageB(args.out_dir)
        return

    all_tasks = []
    # 1. T7-NEWEST test set (4 chunks, 98 chains)
    for k in range(4):
        all_tasks.append((
            f"mpnn_t7_c{k}",
            os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt"),
            os.path.join(args.out_dir, "dumps", f"mpnn_t7_c{k}"),
        ))
    # 2. toporef_xrun equal arm (8 chunks, 88 chains)
    for k in range(8):
        all_tasks.append((
            f"mpnn_equal_c{k}",
            os.path.join(args.out_dir, "chunks", f"equal_c{k}_q.txt"),
            os.path.join(args.out_dir, "dumps", f"mpnn_equal_c{k}"),
        ))
    # 3. toporef_xrun self arm (8 chunks, 88 chains)
    for k in range(8):
        all_tasks.append((
            f"mpnn_self_c{k}",
            os.path.join(args.out_dir, "chunks", f"self_c{k}_q.txt"),
            os.path.join(args.out_dir, "dumps", f"mpnn_self_c{k}"),
        ))

    per_gpu = [[] for _ in range(args.ngpus)]
    for i, task in enumerate(all_tasks):
        per_gpu[i % args.ngpus].append(task)

    ctx = mp.get_context("spawn")
    procs = []
    for g in range(args.ngpus):
        p = ctx.Process(
            target=worker_loop,
            args=(g, per_gpu[g], args.ckpt, args.dataset, args.out_dir, args.usalign, args.python_bin),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Stage-B GPU worker exited with code {p.exitcode}")

    score_stageB(args.out_dir)


if __name__ == "__main__":
    main()
