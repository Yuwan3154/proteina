"""Evaluate step=160,148 (chk_best_contact_precL_epoch=00000396_step=000000160148-EMA.ckpt) and ContactEBM
(t_eval=0.25, t0=0.30, K=0, minE/pick/final) on A6000_offsite across all 4 RTX A6000 GPUs.

Phases:
  Phase 1: ContactEBM selection & reverse refinement (pick, minE, final) + Stage-B c2c folding on
           step=75,069 T7-NEWEST maps (mpnn_t7_c0..3, 98 chains / 70 clean primary chains).
  Phase 2: Stage-A (tri 50-step ODE rollout, 8 samples/chain) + Stage-B c2c folding + ContactEBM
           (single-ckpt 8-sample pool and 75k+160k 16-sample pool) for step=160,148 on T7-NEWEST,
           testing whether the validation P@L mean (0.5112 -> 0.5169) and F1 (0.4781 -> 0.5306)
           gains carry over to the test set.
  Phase 3: Stage-A + Stage-B c2c folding for step=160,148 on toporef_xrun (equal and self, 88 chains).
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from scratchpad.run_mpnn_chunk9_eval_a6000 import (
    K_SAMPLES,
    T7_CHANGED_CHAINS,
    T7_CHUNKS,
    XRUN_CHUNKS,
    boot_ci,
    elements_of,
    gpu_worker as stageA_gpu_worker,
    p_at_q,
    prepare_inputs,
)
from scratchpad.run_mpnn_chunk9_stageB_a6000 import worker_loop as stageB_worker_loop
from proteinfoundation.datasets.sse_topology import runs_to_spans
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform

T_EVAL = 0.25
T0, K_CORR = 0.3, 0
TRI_SEED = 0
CEBM_MD5 = "2affbf45e2b28fecbc5f4d2a1dbe3b09"
CEBM_TAGS = {0: "cebm_minE", 1: "cebm_pick", 2: "cebm_final"}


def _cebm_pair_mask(rmask):
    m = rmask.bool()
    L = m.shape[1]
    diag = torch.eye(L, dtype=torch.bool, device=m.device)[None]
    return (m[:, :, None] & m[:, None, :] & ~diag).float()


def load_contact_ebm(cebm_ckpt, dev):
    from contact_ebm.model.contact_ebm import ContactEBM
    from contact_ebm.scripts.train import build_diffusion

    ck = torch.load(cebm_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    mc = cfg["model"]
    diffc = cfg["diffusion"]
    diffusion, c_in = build_diffusion(diffc)
    diffusion = diffusion.to(dev)
    model = ContactEBM(
        vocab_size=diffc.get("vocab_size", 2),
        contact_input_dim=c_in,
        arch=mc["arch"],
        token_s=mc["token_s"],
        token_z=mc["token_z"],
        depth=mc["depth"],
        heads=mc["heads"],
        dim_fourier=mc.get("dim_fourier", 256),
        energy_dim=mc.get("energy_dim", 8),
        activation_checkpointing=False,
    ).to(dev)
    state = ck["ema"] if "ema" in ck else ck["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, diffusion


def score_clean(cebm_model, prob_maps, aatype, Lq, t_eval, dev):
    k = prob_maps.shape[0]
    binary_tokens = torch.from_numpy((prob_maps >= 0.5).astype(np.int64)).to(dev)
    aa_batch = aatype[:Lq].long().to(dev)[None].expand(k, Lq)
    rmask = torch.ones((k, Lq), dtype=torch.float32, device=dev)
    with torch.no_grad():
        return cebm_model.score_energy(binary_tokens, aa_batch, rmask, t_eval=float(t_eval)).detach().float().cpu().numpy()


def cebm_reverse_sample_from_t0(cebm_model, cebm_diff, init_binary_map, aatype, Lq, t0, K_corrector, dev, t_eval, seed):
    from contact_ebm.diffusion.schedules import symmetrize_matrix

    torch.manual_seed(seed)
    x0 = torch.from_numpy((init_binary_map >= 0.5).astype(np.int64)).to(dev)[None]
    aa = aatype[:Lq].long().to(dev)[None]
    rmask = torch.ones((1, Lq), dtype=torch.float32, device=dev)
    pair_mask = cebm_model._pair_mask(rmask) if hasattr(cebm_model, "_pair_mask") else _cebm_pair_mask(rmask)
    pm_b = pair_mask.bool()
    diag = torch.arange(Lq, device=dev)

    timesteps = int(getattr(cebm_diff, "timesteps", 50))
    i0 = 0
    for i in range(timesteps):
        _, t_grid = cebm_diff.get_sampling_grid(i, timesteps)
        if t_grid <= t0 + 1e-6:
            i0 = i
            break
    _, t_start = cebm_diff.get_sampling_grid(i0, timesteps)
    t_start_tensor = torch.full((1,), float(t_start), device=dev, dtype=torch.float32)

    with torch.no_grad():
        zt = cebm_diff.forward_sample(x0, t_start_tensor, pair_mask)
        zt = symmetrize_matrix(zt.float()).long()
        zt[:, diag, diag] = 0
        zt = zt * pm_b.long()

    step_times, step_kinds, step_e, step_maps = [], [], [], []

    def _call(t_tensor):
        with torch.enable_grad():
            logits, _, _ = cebm_model.compute_outputs(cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False)
        logits_d = logits.detach()
        b_clean = (torch.sigmoid(logits_d.float())[0, :Lq, :Lq].cpu().numpy() >= 0.5).astype(np.uint8)
        with torch.no_grad():
            e_eval = float(
                cebm_model.score_energy(
                    torch.from_numpy(b_clean.astype(np.int64)).to(dev)[None],
                    aa,
                    rmask,
                    t_eval=float(t_eval),
                )[0].item()
            )
        return logits_d, b_clean, e_eval

    for i in range(i0, timesteps):
        s, t = cebm_diff.get_sampling_grid(i, timesteps)
        t_tensor = torch.full((1,), float(t), device=dev, dtype=torch.float32)
        logits_d, b_clean, e_eval = _call(t_tensor)
        with torch.no_grad():
            zt = cebm_diff.sample_step(zt, logits_d, s, t, pair_mask=pair_mask)
        step_times.append(float(t))
        step_kinds.append("predictor")
        step_e.append(e_eval)
        step_maps.append(b_clean)

    with torch.no_grad():
        z0 = (torch.sigmoid(logits_d.float()) >= 0.5).long() * pm_b.long()
        final_binary = z0[0, :Lq, :Lq].detach().cpu().numpy().astype(np.uint8)
        final_e = float(cebm_model.score_energy(z0[:, :Lq, :Lq].long(), aa, rmask, t_eval=float(t_eval))[0].item())
    return {
        "t_start": float(t_start),
        "step_times": step_times,
        "step_kinds": step_kinds,
        "step_e": step_e,
        "step_maps": step_maps,
        "final_binary": final_binary,
        "final_e": final_e,
    }


def pack_reader(pack):
    z = np.load(pack + ".idx.npz", allow_pickle=True)
    idx = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    fh = open(pack, "rb")

    def residue_type(stem):
        o, l = idx[stem]
        fh.seek(o)
        d = torch.load(io.BytesIO(fh.read(l)), map_location="cpu", weights_only=False)
        rt = d["residue_type"] if isinstance(d, dict) else d.residue_type
        return torch.as_tensor(rt).long()

    return residue_type


def cebm_worker(gpu_id, tasks, pack_path, cebm_repo, cebm_ckpt):
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    sys.path.insert(0, cebm_repo)
    with open(cebm_ckpt, "rb") as fh:
        md5 = hashlib.file_digest(fh, "md5").hexdigest()
    assert md5 == CEBM_MD5, f"ContactEBM ckpt md5 {md5} != {CEBM_MD5}"
    cebm_model, cebm_diff = load_contact_ebm(cebm_ckpt, dev)
    residue_type = pack_reader(pack_path)

    for label, chains_file, dump_dirs, out_dump_dir in tasks:
        if os.path.exists(os.path.join(out_dump_dir, "samples.jsonl")):
            stems_q = sorted({l.split()[0] for l in open(chains_file) if l.strip()})
            n_got = sum(1 for l in open(os.path.join(out_dump_dir, "samples.jsonl")) if l.strip())
            if n_got == len(stems_q) * 3:
                print(f"[GPU {gpu_id}] SKIP ContactEBM {label} (already complete: {n_got} maps)", flush=True)
                continue

        t0_task = time.time()
        print(f"[GPU {gpu_id}] START ContactEBM {label} from {dump_dirs} -> {out_dump_dir}", flush=True)
        rows = defaultdict(list)
        for d_idx, d in enumerate(dump_dirs):
            for ln in open(os.path.join(d, "samples.jsonl")):
                if ln.strip():
                    r = json.loads(ln)
                    rows[r["stem"]].append((d_idx, r, d))

        stems = sorted({l.split()[0] for l in open(chains_file) if l.strip()})
        os.makedirs(out_dump_dir, exist_ok=True)
        spath = os.path.join(out_dump_dir, "samples.jsonl")
        if os.path.exists(spath):
            os.remove(spath)
        mpath = os.path.join(out_dump_dir, "cebm.jsonl")
        if os.path.exists(mpath):
            os.remove(mpath)

        for stem in stems:
            got = sorted(rows[stem], key=lambda x: (x[0], x[1]["sample_index"]))
            maps, labels, ref_ids, gt, L = [], [], set(), None, None
            for d_idx, r, d in got:
                z = np.load(os.path.join(d, r["file"]))
                L = int(z["L"]) if L is None else L
                assert int(z["L"]) == L
                maps.append(z["contact_prob"].astype(np.float32))
                labels.append(f"src{d_idx}#{r['sample_index']}")
                ref_ids.add(r["ref_id"])
                if gt is None:
                    gt = z["contact_gt"]
            ref_id = sorted(ref_ids)[0]
            aat = residue_type(stem)
            assert int(aat.shape[0]) == L, f"{stem}: pack aatype {aat.shape[0]} != L {L}"
            pool = np.stack(maps)
            e25 = np.concatenate([
                score_clean(cebm_model, pool[i:i + 8], aat, L, T_EVAL, dev)
                for i in range(0, len(pool), 8)
            ])
            pick = int(np.argmin(e25))
            init_bmap = (pool[pick] >= 0.5).astype(np.uint8)
            res = cebm_reverse_sample_from_t0(
                cebm_model, cebm_diff, init_bmap, aat, L, t0=T0, K_corrector=K_CORR, dev=dev, t_eval=T_EVAL, seed=TRI_SEED
            )
            pred = [i for i, k in enumerate(res["step_kinds"]) if k == "predictor"]
            second_half = pred[len(pred) // 2:]
            min_e_idx = min(second_half, key=lambda i: res["step_e"][i])
            mine = res["step_maps"][min_e_idx].astype(np.float32)
            np.fill_diagonal(mine, 0.0)
            outs = (("minE", mine), ("pick", pool[pick]), ("final", res["final_binary"].astype(np.float32)))
            with open(spath, "a") as fh:
                for k, (tag, mp_arr) in enumerate(outs):
                    f = f"{stem}_s{k:02d}.npz"
                    np.savez_compressed(
                        os.path.join(out_dump_dir, f),
                        contact_prob=mp_arr,
                        contact_gt=gt,
                        L=np.int32(L),
                        stem=stem,
                        sample_index=np.int32(k),
                        ref_id=ref_id,
                        cebm_output=tag,
                    )
                    fh.write(
                        json.dumps({
                            "stem": stem,
                            "sample_index": k,
                            "L": L,
                            "ref_id": ref_id,
                            "file": f,
                            "metrics": got[pick][1].get("metrics", {}),
                            "cebm_output": tag,
                        })
                        + "\n"
                    )
            with open(mpath, "a") as fh:
                fh.write(
                    json.dumps({
                        "stem": stem,
                        "L": L,
                        "ref_id": ref_id,
                        "pick_idx": pick,
                        "pick_label": labels[pick],
                        "pick_e25": float(e25[pick]),
                        "minE_e25": float(res["step_e"][min_e_idx]),
                        "final_e25": float(res["final_e"]),
                    })
                    + "\n"
                )
        print(f"[GPU {gpu_id}] DONE ContactEBM {label} ({len(stems)} chains) in {time.time() - t0_task:.1f}s", flush=True)


def run_parallel(worker_fn, per_gpu_args):
    ctx = mp.get_context("spawn")
    procs = []
    for g, args_tuple in enumerate(per_gpu_args):
        p = ctx.Process(target=worker_fn, args=(g, *args_tuple))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {p.exitcode}")


def summarize_all(out_dir, index_path):
    tf = TopologyReferenceTransform(
        index_path=index_path,
        reference_source="synthetic",
        reference_mode="mpnn_chunk9",
        chunk_size=9,
        dssp_def="tmalign_smooth",
        sse_types=(1, 2),
        element_types=(1,),
        elem_features=True,
    )
    tf._ensure_loaded()
    tf_he = TopologyReferenceTransform(
        index_path="/home/jupyter-chenxi/sse1d_idx/tmalign_smooth/topology_index_confind_synth_tmalign_smooth.pt",
        reference_source="synthetic",
        dssp_def="tmalign_smooth",
        sse_types=(1, 2),
        element_types=(1, 2),
    )
    tf_he._ensure_loaded()

    lines = ["=== MPNN_CHUNK9 STEP=75,069 vs STEP=160,148 & CONTACTEBM COMPARISON ==="]

    # 1. Stage-A comparison (75k vs 160k) across t7, equal, self
    stageA_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tag, prefix in (("s75k", "mpnn"), ("s160k", "mpnn_s160k")):
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
        if "s75k" not in stageA_data[arm] or "s160k" not in stageA_data[arm]:
            continue
        stems_all = sorted(set(stageA_data[arm]["s75k"].keys()) & set(stageA_data[arm]["s160k"].keys()))
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
                mu75, lo75, hi75 = boot_ci(a75, rng)
                mu160, lo160, hi160 = boot_ci(a160, rng)
                diff = a160 - a75
                mud, lod, hid = boot_ci(diff, rng)
                wins = int((diff > 0).sum())
                lines.append(
                    f"  {mlabel:8s} | s75k: {mu75:.4f} [{lo75:.4f}, {hi75:.4f}] (med {np.median(a75):.4f}) "
                    f"| s160k: {mu160:.4f} [{lo160:.4f}, {hi160:.4f}] (med {np.median(a160):.4f}) "
                    f"| s160k - s75k: {mud:+.4f} [{lod:+.4f}, {hid:+.4f}] (160k>75k: {wins}/{len(S)})"
                )

    # 2. Stage-B c2c TM comparison (s75k, s160k, cebm_75k, cebm_160k, cebm_pool16)
    sb_dir = os.path.join(out_dir, "stageB")
    tm_by_arm = defaultdict(lambda: defaultdict(list))
    for tag, prefix in (("s75k", "mpnn_t7"), ("s160k", "mpnn_s160k_t7")):
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

    for cebm_tag, prefix in (("s75k", "mpnn_cebm_t7"), ("s160k", "mpnn_s160k_cebm_t7"), ("pool16", "mpnn_pool16_cebm_t7")):
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
            lines.append(f"{'arm':<22} {'samples':>8} {'TM>=0.5':>9} {'chains mean>=0.5':>17}   mean TM [95% CI]          median")
            means = {}
            for arm_name in (
                "s75k",
                "s160k",
                "pool16_mean",
                "s75k_cebm_pick",
                "s75k_cebm_minE",
                "s75k_cebm_final",
                "s160k_cebm_pick",
                "s160k_cebm_minE",
                "s160k_cebm_final",
                "pool16_cebm_pick",
                "pool16_cebm_minE",
                "pool16_cebm_final",
            ):
                if arm_name not in tm_by_arm or any(s not in tm_by_arm[arm_name] for s in S):
                    continue
                rng = np.random.default_rng(0)
                m = np.array([np.mean(tm_by_arm[arm_name][s]) for s in S])
                means[arm_name] = m
                allv = np.concatenate([tm_by_arm[arm_name][s] for s in S])
                mu, lo, hi = boot_ci(m, rng)
                lines.append(
                    f"{arm_name:<22} {len(allv):>8} {int((allv >= 0.5).sum()):>9} {int((m >= 0.5).sum()):>17}   "
                    f"{mu:.4f} [{lo:.4f}, {hi:.4f}]   {np.median(m):.4f}"
                )
            lines.append("paired difference B - A (per chain)   mean [95% CI]            chains B>A / n")
            for b_, a_ in (
                ("s160k", "s75k"),
                ("s75k_cebm_pick", "s75k"),
                ("s75k_cebm_minE", "s75k"),
                ("s75k_cebm_minE", "s75k_cebm_pick"),
                ("s160k_cebm_pick", "s160k"),
                ("s160k_cebm_minE", "s160k"),
                ("s160k_cebm_minE", "s75k_cebm_minE"),
                ("pool16_cebm_pick", "s75k"),
                ("pool16_cebm_minE", "s75k"),
            ):
                if b_ in means and a_ in means:
                    rng = np.random.default_rng(0)
                    d = means[b_] - means[a_]
                    mu, lo, hi = boot_ci(d, rng)
                    lines.append(
                        f"  {b_:>18} - {a_:<18} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]   {int((d > 0).sum())}/{len(d)}"
                    )

    # 3. toporef_xrun Stage-B c2c TM comparison (equal and self, s75k vs s160k)
    for xarm in ("equal", "self"):
        x_tm = defaultdict(lambda: defaultdict(list))
        for tag, prefix in (("s75k", f"mpnn_{xarm}"), ("s160k", f"mpnn_s160k_{xarm}")):
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
        for tag in ("s75k", "s160k"):
            if any(s not in x_tm[tag] for s in S):
                continue
            rng = np.random.default_rng(0)
            m = np.array([np.mean(x_tm[tag][s]) for s in S])
            means[tag] = m
            allv = np.concatenate([x_tm[tag][s] for s in S])
            mu, lo, hi = boot_ci(m, rng)
            lines.append(
                f"  {tag:<10} samples={len(allv):3d}  TM>=0.5={int((allv>=0.5).sum()):3d}  "
                f"chains>=0.5={int((m>=0.5).sum()):2d}/{len(S)}  mean TM={mu:.4f} [{lo:.4f}, {hi:.4f}]  med={np.median(m):.4f}"
            )
        if "s75k" in means and "s160k" in means:
            rng = np.random.default_rng(0)
            d = means["s160k"] - means["s75k"]
            mu, lo, hi = boot_ci(d, rng)
            lines.append(f"  paired diff s160k - s75k: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  wins={int((d>0).sum())}/{len(d)}")

    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "s160k_and_cebm_report.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/mpnn_chunk9_eval")
    ap.add_argument("--config_name", default="training_contact_tri_v2_confind_mpnn_chunk9")
    ap.add_argument(
        "--ckpt_160k",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_precL_epoch=00000396_step=000000160148-EMA.ckpt",
    )
    ap.add_argument("--c2c_ckpt", default="/home/jupyter-chenxi/mpnn_chunk9_eval/ckpt/final_twin_s29000_ema_only.ckpt")
    ap.add_argument("--c2c_dataset", default="pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--pack_path", default="/home/jupyter-chenxi/sse1d_data/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack")
    ap.add_argument("--cebm_repo", default="/home/jupyter-chenxi/ContactEBM")
    ap.add_argument("--cebm_ckpt", default="/home/jupyter-chenxi/contactebm_runs/confind_a1_prod100/epoch82.pt")
    ap.add_argument("--index", default="/home/jupyter-chenxi/sse1d_idx/mpnn_chunk9/topology_index_confind_synth_mpnn_chunk9.pt")
    ap.add_argument("--usalign", default="/home/jupyter-chenxi/.local/bin/USalign")
    ap.add_argument("--python_bin", default="/home/jupyter-chenxi/miniconda3/envs/proteina_sse1d/bin/python")
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        summarize_all(args.out_dir, args.index)
        return

    prepare_inputs(args.out_dir)

    # Phase 1: ContactEBM (pick, minE, final) + Stage-B c2c on step=75,069 T7-NEWEST maps
    print("=== PHASE 1: ContactEBM + Stage-B c2c on step=75,069 T7-NEWEST ===", flush=True)
    cebm_75k_tasks = [[] for _ in range(args.ngpus)]
    c2c_cebm_75k_tasks = [[] for _ in range(args.ngpus)]
    for k in range(4):
        q_file = os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt")
        src_dump = os.path.join(args.out_dir, "dumps", f"mpnn_t7_c{k}")
        out_dump = os.path.join(args.out_dir, "dumps", f"mpnn_cebm_t7_c{k}")
        cebm_75k_tasks[k % args.ngpus].append((f"mpnn_cebm_t7_c{k}", q_file, [src_dump], out_dump))
        c2c_cebm_75k_tasks[k % args.ngpus].append((f"mpnn_cebm_t7_c{k}", q_file, out_dump))

    run_parallel(
        cebm_worker,
        [(cebm_75k_tasks[g], args.pack_path, args.cebm_repo, args.cebm_ckpt) for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(c2c_cebm_75k_tasks[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    summarize_all(args.out_dir, args.index)

    # Phase 2: Step=160,148 Stage-A + Stage-B c2c + ContactEBM on T7-NEWEST
    print("=== PHASE 2: Step=160,148 Stage-A + Stage-B c2c + ContactEBM on T7-NEWEST ===", flush=True)
    s160k_t7_stageA = [[] for _ in range(args.ngpus)]
    s160k_t7_stageB = [[] for _ in range(args.ngpus)]
    cebm_160k_tasks = [[] for _ in range(args.ngpus)]
    c2c_cebm_160k_tasks = [[] for _ in range(args.ngpus)]
    for k in range(4):
        q_file = os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt")
        dump_160 = os.path.join(args.out_dir, "dumps", f"mpnn_s160k_t7_c{k}")
        dump_75 = os.path.join(args.out_dir, "dumps", f"mpnn_t7_c{k}")
        s160k_t7_stageA[k % args.ngpus].append(("t7", k))
        s160k_t7_stageB[k % args.ngpus].append((f"mpnn_s160k_t7_c{k}", q_file, dump_160))

        out_cebm_160 = os.path.join(args.out_dir, "dumps", f"mpnn_s160k_cebm_t7_c{k}")
        out_cebm_pool16 = os.path.join(args.out_dir, "dumps", f"mpnn_pool16_cebm_t7_c{k}")
        cebm_160k_tasks[k % args.ngpus].append((f"mpnn_s160k_cebm_t7_c{k}", q_file, [dump_160], out_cebm_160))
        cebm_160k_tasks[k % args.ngpus].append((f"mpnn_pool16_cebm_t7_c{k}", q_file, [dump_75, dump_160], out_cebm_pool16))
        c2c_cebm_160k_tasks[k % args.ngpus].append((f"mpnn_s160k_cebm_t7_c{k}", q_file, out_cebm_160))
        c2c_cebm_160k_tasks[k % args.ngpus].append((f"mpnn_pool16_cebm_t7_c{k}", q_file, out_cebm_pool16))

    run_parallel(
        stageA_gpu_worker,
        [(s160k_t7_stageA[g], args.config_name, args.ckpt_160k, args.out_dir, args.seed, "mpnn_s160k") for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(s160k_t7_stageB[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    run_parallel(
        cebm_worker,
        [(cebm_160k_tasks[g], args.pack_path, args.cebm_repo, args.cebm_ckpt) for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(c2c_cebm_160k_tasks[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    summarize_all(args.out_dir, args.index)

    # Phase 3: Step=160,148 Stage-A + Stage-B c2c on toporef_xrun (equal and self)
    print("=== PHASE 3: Step=160,148 Stage-A + Stage-B c2c on toporef_xrun (equal and self) ===", flush=True)
    s160k_xrun_stageA = [[] for _ in range(args.ngpus)]
    s160k_xrun_stageB = [[] for _ in range(args.ngpus)]
    for idx_t, arm in enumerate(("equal", "self")):
        for k in range(8):
            q_file = os.path.join(args.out_dir, "chunks", f"{arm}_c{k}_q.txt")
            dump_160 = os.path.join(args.out_dir, "dumps", f"mpnn_s160k_{arm}_c{k}")
            g = (idx_t * 8 + k) % args.ngpus
            s160k_xrun_stageA[g].append((arm, k))
            s160k_xrun_stageB[g].append((f"mpnn_s160k_{arm}_c{k}", q_file, dump_160))

    run_parallel(
        stageA_gpu_worker,
        [(s160k_xrun_stageA[g], args.config_name, args.ckpt_160k, args.out_dir, args.seed, "mpnn_s160k") for g in range(args.ngpus)],
    )
    run_parallel(
        stageB_worker_loop,
        [(s160k_xrun_stageB[g], args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin) for g in range(args.ngpus)],
    )
    summarize_all(args.out_dir, args.index)


if __name__ == "__main__":
    main()
