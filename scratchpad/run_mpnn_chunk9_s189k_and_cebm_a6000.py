"""Evaluate step=189,375 (chk_best_contact_f1_epoch=00000468_step=000000189375-EMA.ckpt, all-time peak
validation_sampling/contact_f1_median = 0.5523) and ContactEBM (t_eval=0.25, t0=0.30, K=0, minE/pick/final)
on A6000_offsite across all 4 RTX A6000 GPUs, comparing against step=75,069 (s75k) and step=160,148 (s160k).

Phases:
  Phase 1: Stage-A (tri 50-step ODE rollout, 8 samples/chain) + Stage-B c2c folding + ContactEBM
           (single-ckpt s189k 8-sample pool, s160k+s189k 16-sample pool, and s75k+s160k+s189k 24-sample pool)
           on T7-NEWEST (98 chains / 70 clean primary chains).
  Phase 2: Stage-A + Stage-B c2c folding for step=189,375 on toporef_xrun (equal and self, 88 chains).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from scratchpad.run_mpnn_chunk9_eval_a6000 import (
    T7_CHANGED_CHAINS,
    boot_ci,
    gpu_worker as stageA_gpu_worker,
    prepare_inputs,
)
from scratchpad.run_mpnn_chunk9_s160k_and_cebm_a6000 import (
    CEBM_TAGS,
    cebm_worker,
    run_parallel,
)
from scratchpad.run_mpnn_chunk9_stageB_a6000 import worker_loop as stageB_worker_loop


def summarize_s189k(out_dir):
    lines = ["=== MPNN_CHUNK9 STEP=75,069 (s75k) vs STEP=160,148 (s160k) vs STEP=189,375 (s189k) & CONTACTEBM COMPARISON ==="]

    # 1. Stage-A comparison (s75k vs s160k vs s189k) across t7, equal, self
    stageA_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tag, prefix in (("s75k", "mpnn"), ("s160k", "mpnn_s160k"), ("s189k", "mpnn_s189k")):
        for arm, n_c in (("t7", 4), ("equal", 8), ("self", 8)):
            for c in range(n_c):
                d = os.path.join(out_dir, "dumps", f"{prefix}_{arm}_c{c}")
                sp = os.path.join(d, "samples.jsonl")
                if not os.path.exists(sp):
                    continue
                for ln in open(sp):
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    st = r["stem"]
                    m = r["metrics"]
                    stageA_data[arm][tag][st].append(float(m["contact_precision_at_L"]))
                    stageA_data[f"{arm}_f1"][tag][st].append(float(m.get("contact_f1", 0.0)))
                    stageA_data[f"{arm}_pl5"][tag][st].append(float(m.get("contact_precision_at_L5", 0.0)))
                    stageA_data[f"{arm}_lr_pl5"][tag][st].append(float(m.get("contact_long_range_precision_at_L5", 0.0)))

    for arm in ("t7", "equal", "self"):
        if "s75k" not in stageA_data[arm] or "s189k" not in stageA_data[arm]:
            continue
        stems_all = sorted(
            set(stageA_data[arm]["s75k"].keys())
            & set(stageA_data[arm].get("s160k", {}).keys())
            & set(stageA_data[arm]["s189k"].keys())
        )
        if not stems_all:
            continue
        subsets = [("all", stems_all)]
        if arm == "t7":
            clean = [s for s in stems_all if s not in T7_CHANGED_CHAINS]
            subsets = [("clean_70", clean), ("all_98", stems_all)]
        for sub_name, S in subsets:
            lines.append(f"\n--- Stage-A Contact Metrics: {arm} ({sub_name}, n={len(S)} chains) ---")
            for mkey, mlabel in (("", "P@L"), ("_pl5", "P@L/5"), ("_lr_pl5", "LR_P@L/5"), ("_f1", "F1")):
                rng = np.random.default_rng(0)
                a75 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s75k"][s]) for s in S])
                a160 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s160k"][s]) for s in S])
                a189 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s189k"][s]) for s in S])
                mu75, lo75, hi75 = boot_ci(a75, rng)
                mu160, lo160, hi160 = boot_ci(a160, rng)
                mu189, lo189, hi189 = boot_ci(a189, rng)
                d_189_75 = a189 - a75
                mud75, lod75, hid75 = boot_ci(d_189_75, rng)
                d_189_160 = a189 - a160
                mud160, lod160, hid160 = boot_ci(d_189_160, rng)
                lines.append(
                    f"  {mlabel:8s} | s75k: {mu75:.4f} (med {np.median(a75):.4f}) "
                    f"| s160k: {mu160:.4f} (med {np.median(a160):.4f}) "
                    f"| s189k: {mu189:.4f} [{lo189:.4f}, {hi189:.4f}] (med {np.median(a189):.4f}) "
                    f"| s189k-s75k: {mud75:+.4f} [{lod75:+.4f}, {hid75:+.4f}] ({int((d_189_75>0).sum())}/{len(S)}) "
                    f"| s189k-s160k: {mud160:+.4f} [{lod160:+.4f}, {hid160:+.4f}] ({int((d_189_160>0).sum())}/{len(S)})"
                )

    # 2. Stage-B c2c TM comparison on T7-NEWEST
    sb_dir = os.path.join(out_dir, "stageB")
    tm_by_arm = defaultdict(lambda: defaultdict(list))
    for tag, prefix in (("s75k", "mpnn_t7"), ("s160k", "mpnn_s160k_t7"), ("s189k", "mpnn_s189k_t7")):
        for c in range(4):
            p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip():
                        r = json.loads(ln)
                        tm_by_arm[tag][r["stem"]].append(float(r["tm"]))

    if "s75k" in tm_by_arm and "s160k" in tm_by_arm:
        for st in set(tm_by_arm["s75k"].keys()) & set(tm_by_arm["s160k"].keys()):
            if len(tm_by_arm["s75k"][st]) == 8 and len(tm_by_arm["s160k"][st]) == 8:
                tm_by_arm["pool16_mean"][st] = tm_by_arm["s75k"][st] + tm_by_arm["s160k"][st]
    if "s160k" in tm_by_arm and "s189k" in tm_by_arm:
        for st in set(tm_by_arm["s160k"].keys()) & set(tm_by_arm["s189k"].keys()):
            if len(tm_by_arm["s160k"][st]) == 8 and len(tm_by_arm["s189k"][st]) == 8:
                tm_by_arm["pool16_160_189_mean"][st] = tm_by_arm["s160k"][st] + tm_by_arm["s189k"][st]
                if st in tm_by_arm["s75k"] and len(tm_by_arm["s75k"][st]) == 8:
                    tm_by_arm["pool24_mean"][st] = tm_by_arm["s75k"][st] + tm_by_arm["s160k"][st] + tm_by_arm["s189k"][st]

    for cebm_tag, prefix in (
        ("s75k", "mpnn_cebm_t7"),
        ("s160k", "mpnn_s160k_cebm_t7"),
        ("s189k", "mpnn_s189k_cebm_t7"),
        ("pool16", "mpnn_pool16_cebm_t7"),
        ("pool16_160_189", "mpnn_pool16_160_189_cebm_t7"),
        ("pool24", "mpnn_pool24_cebm_t7"),
    ):
        for c in range(4):
            p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip():
                        r = json.loads(ln)
                        sub = CEBM_TAGS[int(r["sample_index"])]
                        tm_by_arm[f"{cebm_tag}_{sub}"][r["stem"]].append(float(r["tm"]))

    t7_all = sorted(tm_by_arm["s75k"].keys())
    if t7_all:
        t7_clean = [s for s in t7_all if s not in T7_CHANGED_CHAINS]
        for sub_name, S in (("clean_70", t7_clean), ("all_98", t7_all)):
            lines.append(f"\n=== T7-NEWEST Stage-B c2c 3D Structure TM-Score ({sub_name}, n={len(S)} chains) ===")
            lines.append(f"{'arm':<26} {'samples':>8} {'TM>=0.5':>9} {'chains mean>=0.5':>17}   mean TM [95% CI]          median   best_of_K")
            means = {}
            for arm_name in (
                "s75k",
                "s160k",
                "s189k",
                "pool16_mean",
                "pool16_160_189_mean",
                "pool24_mean",
                "s75k_cebm_pick",
                "s75k_cebm_minE",
                "s75k_cebm_final",
                "s160k_cebm_pick",
                "s160k_cebm_minE",
                "s160k_cebm_final",
                "s189k_cebm_pick",
                "s189k_cebm_minE",
                "s189k_cebm_final",
                "pool16_cebm_pick",
                "pool16_cebm_minE",
                "pool16_cebm_final",
                "pool16_160_189_cebm_pick",
                "pool16_160_189_cebm_minE",
                "pool16_160_189_cebm_final",
                "pool24_cebm_pick",
                "pool24_cebm_minE",
                "pool24_cebm_final",
            ):
                if arm_name not in tm_by_arm or any(s not in tm_by_arm[arm_name] for s in S):
                    continue
                rng = np.random.default_rng(0)
                m = np.array([np.mean(tm_by_arm[arm_name][s]) for s in S])
                b = np.array([np.max(tm_by_arm[arm_name][s]) for s in S])
                means[arm_name] = m
                allv = np.concatenate([tm_by_arm[arm_name][s] for s in S])
                mu, lo, hi = boot_ci(m, rng)
                lines.append(
                    f"{arm_name:<26} {len(allv):>8} {int((allv >= 0.5).sum()):>9} {int((m >= 0.5).sum()):>17}   "
                    f"{mu:.4f} [{lo:.4f}, {hi:.4f}]   {np.median(m):.4f}   {np.mean(b):.4f} ({int((b>=0.5).sum())}/{len(S)})"
                )
            lines.append("paired difference B - A (per chain)             mean [95% CI]            chains B>A / n")
            for b_, a_ in (
                ("s160k", "s75k"),
                ("s189k", "s75k"),
                ("s189k", "s160k"),
                ("s189k_cebm_pick", "s189k"),
                ("s189k_cebm_minE", "s189k"),
                ("s189k_cebm_pick", "s75k"),
                ("s189k_cebm_minE", "s75k"),
                ("s189k_cebm_pick", "s160k"),
                ("s189k_cebm_minE", "s160k"),
                ("s189k_cebm_pick", "s160k_cebm_pick"),
                ("s189k_cebm_minE", "s160k_cebm_minE"),
                ("pool16_160_189_cebm_pick", "s75k"),
                ("pool16_160_189_cebm_minE", "s75k"),
                ("pool16_160_189_cebm_final", "s75k"),
                ("pool24_cebm_pick", "s75k"),
                ("pool24_cebm_minE", "s75k"),
                ("pool24_cebm_final", "s75k"),
                ("pool24_cebm_pick", "s160k"),
                ("pool24_cebm_minE", "s160k"),
                ("pool24_cebm_final", "s160k"),
            ):
                if b_ in means and a_ in means:
                    rng = np.random.default_rng(0)
                    d = means[b_] - means[a_]
                    mu, lo, hi = boot_ci(d, rng)
                    lines.append(
                        f"  {b_:>24} - {a_:<20} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]   {int((d > 0).sum())}/{len(d)}"
                    )

    # 3. toporef_xrun Stage-B c2c TM comparison (equal and self, s75k vs s160k vs s189k)
    for xarm in ("equal", "self"):
        x_tm = defaultdict(lambda: defaultdict(list))
        for tag, prefix in (("s75k", f"mpnn_{xarm}"), ("s160k", f"mpnn_s160k_{xarm}"), ("s189k", f"mpnn_s189k_{xarm}")):
            for c in range(8):
                p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
                if os.path.exists(p):
                    for ln in open(p):
                        if ln.strip():
                            r = json.loads(ln)
                            x_tm[tag][r["stem"]].append(float(r["tm"]))
        if not x_tm["s75k"]:
            continue
        S = sorted(x_tm["s75k"].keys())
        lines.append(f"\n=== Held-Out Eval Set (toporef_xrun) Stage-B c2c TM: {xarm} (n={len(S)} chains) ===")
        means = {}
        for tag in ("s75k", "s160k", "s189k"):
            if any(s not in x_tm[tag] for s in S):
                continue
            rng = np.random.default_rng(0)
            m = np.array([np.mean(x_tm[tag][s]) for s in S])
            b = np.array([np.max(x_tm[tag][s]) for s in S])
            means[tag] = m
            allv = np.concatenate([x_tm[tag][s] for s in S])
            mu, lo, hi = boot_ci(m, rng)
            lines.append(
                f"  {tag:<10} samples={len(allv):3d}  TM>=0.5={int((allv>=0.5).sum()):3d}  "
                f"chains>=0.5={int((m>=0.5).sum()):2d}/{len(S)}  mean TM={mu:.4f} [{lo:.4f}, {hi:.4f}]  "
                f"med={np.median(m):.4f}  best8={np.mean(b):.4f}"
            )
        for b_, a_ in (("s160k", "s75k"), ("s189k", "s75k"), ("s189k", "s160k")):
            if b_ in means and a_ in means:
                rng = np.random.default_rng(0)
                d = means[b_] - means[a_]
                mu, lo, hi = boot_ci(d, rng)
                lines.append(f"  paired diff {b_:<6} - {a_:<6}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  wins={int((d>0).sum())}/{len(d)}")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "s189k_and_cebm_report.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/mpnn_chunk9_eval")
    ap.add_argument("--config_name", default="training_contact_tri_v2_confind_mpnn_chunk9")
    ap.add_argument(
        "--ckpt_189k",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_f1_epoch=00000468_step=000000189375-EMA.ckpt",
    )
    ap.add_argument("--c2c_ckpt", default="/home/jupyter-chenxi/mpnn_chunk9_eval/ckpt/final_twin_s29000_ema_only.ckpt")
    ap.add_argument("--c2c_dataset", default="pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--pack_path", default="/home/jupyter-chenxi/sse1d_data/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack")
    ap.add_argument("--cebm_repo", default="/home/jupyter-chenxi/ContactEBM")
    ap.add_argument("--cebm_ckpt", default="/home/jupyter-chenxi/contactebm_runs/confind_a1_prod100/epoch82.pt")
    ap.add_argument("--usalign", default="/home/jupyter-chenxi/.local/bin/USalign")
    ap.add_argument("--python_bin", default="/home/jupyter-chenxi/miniconda3/envs/proteina_sse1d/bin/python")
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        summarize_s189k(args.out_dir)
        return

    prepare_inputs(args.out_dir)

    # Phase 1: Step=189,375 Stage-A + Stage-B c2c + ContactEBM on T7-NEWEST
    print("=== PHASE 1: Step=189,375 Stage-A + Stage-B c2c + ContactEBM on T7-NEWEST ===", flush=True)
    s189k_t7_stageA = [[] for _ in range(args.ngpus)]
    s189k_t7_stageB = [[] for _ in range(args.ngpus)]
    cebm_189k_tasks = [[] for _ in range(args.ngpus)]
    c2c_cebm_189k_tasks = [[] for _ in range(args.ngpus)]
    for k in range(4):
        q_file = os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt")
        dump_189 = os.path.join(args.out_dir, "dumps", f"mpnn_s189k_t7_c{k}")
        dump_160 = os.path.join(args.out_dir, "dumps", f"mpnn_s160k_t7_c{k}")
        dump_75 = os.path.join(args.out_dir, "dumps", f"mpnn_t7_c{k}")
        s189k_t7_stageA[k % args.ngpus].append(("t7", k))
        s189k_t7_stageB[k % args.ngpus].append((f"mpnn_s189k_t7_c{k}", q_file, dump_189))

        out_cebm_189 = os.path.join(args.out_dir, "dumps", f"mpnn_s189k_cebm_t7_c{k}")
        out_cebm_pool16_160_189 = os.path.join(args.out_dir, "dumps", f"mpnn_pool16_160_189_cebm_t7_c{k}")
        out_cebm_pool24 = os.path.join(args.out_dir, "dumps", f"mpnn_pool24_cebm_t7_c{k}")
        cebm_189k_tasks[k % args.ngpus].append((f"mpnn_s189k_cebm_t7_c{k}", q_file, [dump_189], out_cebm_189))
        cebm_189k_tasks[k % args.ngpus].append(
            (f"mpnn_pool16_160_189_cebm_t7_c{k}", q_file, [dump_160, dump_189], out_cebm_pool16_160_189)
        )
        cebm_189k_tasks[k % args.ngpus].append(
            (f"mpnn_pool24_cebm_t7_c{k}", q_file, [dump_75, dump_160, dump_189], out_cebm_pool24)
        )
        c2c_cebm_189k_tasks[k % args.ngpus].append((f"mpnn_s189k_cebm_t7_c{k}", q_file, out_cebm_189))
        c2c_cebm_189k_tasks[k % args.ngpus].append(
            (f"mpnn_pool16_160_189_cebm_t7_c{k}", q_file, out_cebm_pool16_160_189)
        )
        c2c_cebm_189k_tasks[k % args.ngpus].append((f"mpnn_pool24_cebm_t7_c{k}", q_file, out_cebm_pool24))

    run_parallel(
        stageA_gpu_worker,
        [(s189k_t7_stageA[g], args.config_name, args.ckpt_189k, args.out_dir, args.seed, "mpnn_s189k") for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(s189k_t7_stageB[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    run_parallel(
        cebm_worker,
        [(cebm_189k_tasks[g], args.pack_path, args.cebm_repo, args.cebm_ckpt) for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(c2c_cebm_189k_tasks[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    summarize_s189k(args.out_dir)

    # Phase 2: Step=189,375 Stage-A + Stage-B c2c on toporef_xrun (equal and self)
    print("=== PHASE 2: Step=189,375 Stage-A + Stage-B c2c on toporef_xrun (equal and self) ===", flush=True)
    s189k_xrun_stageA = [[] for _ in range(args.ngpus)]
    s189k_xrun_stageB = [[] for _ in range(args.ngpus)]
    for idx_t, arm in enumerate(("equal", "self")):
        for k in range(8):
            q_file = os.path.join(args.out_dir, "chunks", f"{arm}_c{k}_q.txt")
            dump_189 = os.path.join(args.out_dir, "dumps", f"mpnn_s189k_{arm}_c{k}")
            g = (idx_t * 8 + k) % args.ngpus
            s189k_xrun_stageA[g].append((arm, k))
            s189k_xrun_stageB[g].append((f"mpnn_s189k_{arm}_c{k}", q_file, dump_189))

    run_parallel(
        stageA_gpu_worker,
        [(s189k_xrun_stageA[g], args.config_name, args.ckpt_189k, args.out_dir, args.seed, "mpnn_s189k") for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(s189k_xrun_stageB[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    summarize_s189k(args.out_dir)


if __name__ == "__main__":
    main()
