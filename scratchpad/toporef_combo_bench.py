"""Combined & hyperparameter-sweep inference experiments for TopoRef + ContactEBM + c2c.

Tests combinations of our three pillars on the 10 bad_afdb-22 benchmark targets (lad_r250 and tmpl):
  Pillar 1 (Template Featurization):
    - `h_bond` (PyDSSP)
    - `tmalign` (USalign CA-based `make_sec`)
    - `tmalign_smooth` (`make_sec` + 3-pass `smooth`)
    - `tmalign_smooth_minlen` (`tmalign_smooth` + dropping tiny fragments H<3 or E<2)
    - `sse_ensemble` (pool k=8 samples from `h_bond` + k=8 from `tmalign` + k=8 from `tmalign_smooth` = 24 candidates,
      ranked by ContactEBM at t_eval)
  Pillar 2 (TopoRef Sampling & ContactEBM Selection Hyperparameters):
    - Sample pool size: k=8 vs k=16 vs k=24 (multi-SSE ensemble)
    - ContactEBM ranking evaluation time sweep: t_eval in {0.05, 0.15, 0.20, 0.25, 0.30, 0.35} and multi-t average
      E_multi = mean(E(0.15), E(0.25), E(0.35))
    - Contact binarization cutoff sweep before c2c: tau in {0.35, 0.40, 0.45, 0.50, 0.55, 0.60}
  Pillar 3 (Joint Cycling + ContactEBM Reverse / Langevin Refinement):
    - Regime C1 (`tmalign` / `sse_ensemble` -> ContactEBM pick -> 1b local refinement `t0=0.20..0.30, K=0`)
    - Regime C2 (`tmalign` / `sse_ensemble` -> ContactEBM pick -> 1b multi-trajectory energy-gated portfolio
      `{(0.2, K=0), (0.3, K=0), (0.4, K=10), (0.5, K=20)}` + trajectory minimum-energy step selection)
    - Regime C3 (`tmalign` r=0 -> c2c -> `tmalign_smooth` r=1 cycling (energy-gated) -> 1b ContactEBM refinement)
"""

import argparse
import glob
import json
import os
import sys
import time

import lightning as L
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from proteinfoundation.datasets.sse_topology import DSSP_HELIX, DSSP_STRAND, STRUCTURAL_PAIR_FEATURES
from proteinfoundation.nn.af3_diffusion import SIGMA_DATA
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils import precompute_synthetic_topology_index as builder
from proteinfoundation.utils.c2c_dump import write_atom14_pdb
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from synth_tri_bench import (
    C2C_SEED_BASE,
    REF_EXPECT,
    TRI_SEED,
    check_index_matches_builder,
    load_c2c,
    load_predictor,
    load_tri,
)
from toporef_regimes_bench import (
    build_natural_aa_pair_stats,
    build_toporef_transform,
    c2c_rollout_atom14,
    cebm_reverse_sample_from_t0,
    contact_metrics_vs_gt,
    coords14_to_atom37,
    filter_targets,
    generate_tri_with_trajectory,
    load_contact_ebm,
    load_structure_any,
    native_for,
    score_maps_with_cebm,
    template_reference_with_dssp,
    tm_pair_seqdep,
    tm_pair_seqindep,
    tm_vs_native,
    unpack_upper_triangle,
)


def filter_dssp_minlen(dssp_vec: torch.Tensor, min_helix: int = 3, min_strand: int = 2) -> torch.Tensor:
    """Zero out (set to loop=0) helix runs shorter than min_helix and strand runs shorter than min_strand."""
    out = dssp_vec.clone()
    vals = out.tolist()
    i = 0
    n = len(vals)
    while i < n:
        j = i
        while j < n and vals[j] == vals[i]:
            j += 1
        length = j - i
        if vals[i] == DSSP_HELIX and length < min_helix:
            out[i:j] = 0
        elif vals[i] == DSSP_STRAND and length < min_strand:
            out[i:j] = 0
        i = j
    return out


def template_reference_custom(
    full, amask, aatype, predictor, transform, length, pro_donor, dssp_mode="tmalign", cm_t_override=None
):
    """Support dssp_mode in {'h_bond', 'tmalign', 'tmalign_smooth', 'tmalign_minlen'}."""
    base_dssp = "tmalign_smooth" if dssp_mode == "tmalign_minlen" else dssp_mode
    Lt = int(amask.shape[0])
    donor = torch.from_numpy(aatype.cpu().numpy() != builder.PRO_IDX)[None] if pro_donor else None
    dssp_t = compute_dssp_target(
        full[None],
        amask[:, builder.CA_IDX][None],
        coord_mask=amask[None],
        coord_layout="atom37",
        donor_mask=donor,
        dssp_def=base_dssp,
    )[0]
    if dssp_mode == "tmalign_minlen":
        dssp_t = filter_dssp_minlen(dssp_t, min_helix=4, min_strand=2)
    if cm_t_override is not None:
        cm_t = torch.as_tensor(cm_t_override, dtype=torch.float32).cpu()
    else:
        x, mask = builder._f2c_inputs(full[None], amask)
        probs = predictor.predict_batch(x, mask[None])[:, :Lt, :Lt].float().cpu()
        cm_t = (probs[0] >= builder.CONFIND_THRESHOLD).float()
    row = builder.topology_row(cm_t, full, amask, dssp_t, REF_EXPECT["min_len"])
    if row is None:
        ref = transform.assemble_reference(
            [], torch.zeros(0, 0), torch.zeros(0, 0, len(STRUCTURAL_PAIR_FEATURES)), length, augment=False
        )
        return ref, 0, dssp_t, cm_t
    runs, ref_bytes, n_el, feat_bytes, _stats, _elem = row
    runs = [(int(t), int(n)) for t, n in runs]
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    if n_el > 0:
        he_contact = torch.frombuffer(bytearray(ref_bytes), dtype=torch.uint8).reshape(n_el, n_el).float()
        structural = torch.frombuffer(bytearray(feat_bytes), dtype=torch.float16).reshape(n_el, n_el, n_struct).float()
    else:
        he_contact = torch.zeros(0, 0)
        structural = torch.zeros(0, 0, n_struct)
    ref = transform.assemble_reference(runs, he_contact, structural, length, augment=False)
    return ref, n_el, dssp_t, cm_t


def score_maps_multi_t(cebm_model, cebm_diff, prob_maps, aatype, Lq, t_list, dev, cutoff=0.5):
    """Score maps at multiple t_eval values and return dict {t: energies_clean [k]}."""
    k = prob_maps.shape[0]
    binary_tokens = torch.from_numpy((prob_maps >= cutoff).astype(np.int64)).to(dev)
    aa_batch = aatype[:Lq].long().to(dev)[None].expand(k, Lq)
    rmask = torch.ones((k, Lq), dtype=torch.float32, device=dev)
    out = {}
    with torch.no_grad():
        for t_ev in t_list:
            e_cl = cebm_model.score_energy(binary_tokens, aa_batch, rmask, t_eval=float(t_ev)).detach().float().cpu().numpy()
            out[float(t_ev)] = e_cl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d36tc-dir", default="/orcd/pool/006/chenxiou/d36tc")
    ap.add_argument("--natives-dir", default="/home/gridsan/cou/ProteinEBMalign_data/synth_ladder/natives")
    ap.add_argument("--sc-eval-dir", default="/home/gridsan/cou/tri_synth_bench/analysis/eval")
    ap.add_argument("--sc-pairs-dir", default="/home/gridsan/cou/ProteinEBMalign_data/synth_tri")
    ap.add_argument("--sc-pt-dir", default="/home/gridsan/cou/tri_synth_bench/inputs/c1b/stored_pt")
    ap.add_argument("--tri-ckpt", default="/home/gridsan/cou/tri_synth_bench/inputs/cfft_tri_last-EMA_step60505.ckpt")
    ap.add_argument("--c2c-ckpt", default="/home/gridsan/cou/tri_synth_bench/inputs/final_twin_s29000.ckpt")
    ap.add_argument("--index", default="/home/gridsan/cou/tri_synth_bench/inputs/topology_index_confind_synth.pt")
    ap.add_argument("--f2c-ckpt-dir", default="/home/gridsan/cou/Frame2ConFind/runs/f2s_v384_smoothap-pearson")
    ap.add_argument("--f2c-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--cebm-repo", default="/home/gridsan/cou/ContactEBM")
    ap.add_argument("--cebm-ckpt", default="/home/gridsan/cou/c2cbench/yard/ckpt/confind_a1_prod100_epoch82.pt")
    ap.add_argument("--usalign", default="/home/gridsan/cou/USalign/USalign")
    ap.add_argument("--prev-out", default="/home/gridsan/cou/tri_synth_bench/toporef_regimes",
                    help="Reuse existing r0 / sse_tmalign maps where available to save time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tm-threshold", type=float, default=0.5)
    ap.add_argument("--arm-override", choices=("lad_r250", "tmpl", "both"), default="lad_r250")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pad-to", type=int, default=384)
    ap.add_argument("--t-eval", type=float, default=0.25)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    filt = filter_targets(args)
    rows_by_pid = {r["pid"]: r for r in filt["rows"]}
    selected_pids_all = list(filt["selected_pids"])
    selected_pids = selected_pids_all[args.shard :: args.n_shards]
    print(f"[combo shard {args.shard}/{args.n_shards}] targets: {selected_pids}", flush=True)
    if not selected_pids:
        return 0

    assert os.environ.get("SIGMA_DATA") is not None and SIGMA_DATA == 16.0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_toporef_transform(args.index)
    transform._ensure_loaded()
    pro_donor = check_index_matches_builder(transform._index)
    predictor = load_predictor(args.f2c_ckpt_dir, args.f2c_dtype)
    model, cfg_exp = load_tri(args.tri_ckpt, os.path.join(args.out, f".store_tri_shard{args.shard}"))
    model = model.to(dev).eval()
    c2c = load_c2c(args.c2c_ckpt).to(dev).eval()
    cebm_model, cebm_diff, _ = load_contact_ebm(args.cebm_repo, args.cebm_ckpt, dev)
    topo_keys = model._topology_keys()

    all_pids_sorted = sorted(rows_by_pid.keys())
    ci_of = {p: i for i, p in enumerate(all_pids_sorted)}
    arms = ["lad_r250", "tmpl"] if args.arm_override == "both" else [args.arm_override]
    t_sweep_list = (0.05, 0.15, 0.20, 0.25, 0.30, 0.35)

    for arm in arms:
        for pid in selected_pids:
            t_start_target = time.time()
            rinfo = rows_by_pid[pid]
            q_src = rinfo["qpath"]
            t_src = (
                rinfo["tpath_tmpl"]
                if arm == "tmpl"
                else (rinfo.get("tpath_lad_r250", rinfo["tpath_self"]) if arm == "lad_r250" else rinfo["tpath_self"])
            )
            full_q, amask_q, aatype_q, cm_nat = load_structure_any(q_src, predictor=predictor)
            full_t0, amask_t0, aatype_t0, cm_tmpl0 = load_structure_any(t_src, predictor=predictor)
            Lq, Lt0 = int(amask_q.shape[0]), int(amask_t0.shape[0])
            native_cif, native_ch = native_for(args.natives_dir, pid, qpath=rinfo.get("qpath"))
            c2c_seed = C2C_SEED_BASE + ci_of[pid]
            n_pad = Lq if args.pad_to == 0 else args.pad_to

            pair_dir = os.path.join(args.out, arm, pid)
            os.makedirs(pair_dir, exist_ok=True)
            tmpl0_pdb = os.path.join(pair_dir, "template_r0.pdb")
            builder.write_ca_pdb(full_t0[:, builder.CA_IDX].numpy(), amask_t0[:, builder.CA_IDX].numpy(), tmpl0_pdb)

            mask_q = torch.zeros(args.k, n_pad, dtype=torch.bool, device=dev)
            mask_q[:, :Lq] = True
            res_q = torch.zeros(args.k, n_pad, dtype=torch.long, device=dev)
            res_q[:, :Lq] = aatype_q.to(dev)

            # ── 1. Multi-SSE Featurization Pool (h_bond, tmalign, tmalign_smooth, tmalign_minlen) + k=16 tmalign ──
            sse_modes = ("h_bond", "tmalign", "tmalign_smooth", "tmalign_minlen")
            mode_data = {}
            for dmode in sse_modes:
                ref_d, n_el_d, dssp_vec, _ = template_reference_custom(
                    full_t0, amask_t0, aatype_t0, predictor, transform, Lq, pro_donor, dssp_mode=dmode, cm_t_override=cm_tmpl0
                )
                topo_d = {kk: Proteina._stack_topology([ref_d[kk]] * args.k).to(dev) for kk in topo_keys}
                L.seed_everything(TRI_SEED)
                res_d, _, _ = generate_tri_with_trajectory(model, cfg_exp, topo_d, res_q, mask_q, args.k, Lq)
                prob_d = torch.sigmoid(res_d["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
                e_by_t = score_maps_multi_t(cebm_model, cebm_diff, prob_d, aatype_q, Lq, t_sweep_list, dev)

                tms_d = []
                c14s_d = []
                pdbs_d = []
                for si in range(args.k):
                    c14_si = c2c_rollout_atom14(c2c, prob_d[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_si = os.path.join(pair_dir, f"pool_{dmode}_s{si}.pdb")
                    write_atom14_pdb(pdb_si, c14_si, aatype_q, torch.ones(Lq))
                    tms_d.append(tm_vs_native(args.usalign, native_cif, native_ch, pdb_si))
                    c14s_d.append(c14_si)
                    pdbs_d.append(pdb_si)

                pick_25 = int(np.argmin(e_by_t[0.25]))
                e_multi = (e_by_t[0.15] + e_by_t[0.25] + e_by_t[0.35]) / 3.0
                pick_multi = int(np.argmin(e_multi))
                mode_data[dmode] = {
                    "n_el": int(n_el_d),
                    "prob": prob_d,
                    "e_by_t": e_by_t,
                    "e_multi": e_multi,
                    "tms": tms_d,
                    "c14s": c14s_d,
                    "pdbs": pdbs_d,
                    "mean8_tm": float(np.mean(tms_d)),
                    "best8_tm": float(np.max(tms_d)),
                    "pick25_idx": pick_25,
                    "pick25_tm": float(tms_d[pick_25]),
                    "pick25_e": float(e_by_t[0.25][pick_25]),
                    "pick_multi_tm": float(tms_d[pick_multi]),
                }
                print(
                    f"[P1 mode] {pid} ({arm}) {dmode:<15}: n_el={n_el_d:2d} | mean8={np.mean(tms_d):.4f} "
                    f"best8={np.max(tms_d):.4f} pick(t=0.25)={tms_d[pick_25]:.4f} pick(multi_t)={tms_d[pick_multi]:.4f} "
                    f"E25={e_by_t[0.25][pick_25]:.2f}",
                    flush=True,
                )

            # Extra k=8 draw (seed TRI_SEED+100) for `tmalign` to test k=16 pool size
            ref_tm, _, _, _ = template_reference_custom(
                full_t0, amask_t0, aatype_t0, predictor, transform, Lq, pro_donor, dssp_mode="tmalign", cm_t_override=cm_tmpl0
            )
            topo_tm = {kk: Proteina._stack_topology([ref_tm[kk]] * args.k).to(dev) for kk in topo_keys}
            L.seed_everything(TRI_SEED + 100)
            res_tm2, _, _ = generate_tri_with_trajectory(model, cfg_exp, topo_tm, res_q, mask_q, args.k, Lq)
            prob_tm2 = torch.sigmoid(res_tm2["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
            e_tm2_by_t = score_maps_multi_t(cebm_model, cebm_diff, prob_tm2, aatype_q, Lq, t_sweep_list, dev)
            tms_tm2 = []
            c14s_tm2 = []
            for si in range(args.k):
                c14_si = c2c_rollout_atom14(c2c, prob_tm2[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                pdb_si = os.path.join(pair_dir, f"pool_tmalign_b2_s{si}.pdb")
                write_atom14_pdb(pdb_si, c14_si, aatype_q, torch.ones(Lq))
                tms_tm2.append(tm_vs_native(args.usalign, native_cif, native_ch, pdb_si))
                c14s_tm2.append(c14_si)

            tm16_tms = mode_data["tmalign"]["tms"] + tms_tm2
            tm16_probs = np.concatenate([mode_data["tmalign"]["prob"], prob_tm2], axis=0)
            tm16_c14s = mode_data["tmalign"]["c14s"] + c14s_tm2
            tm16_e25 = np.concatenate([mode_data["tmalign"]["e_by_t"][0.25], e_tm2_by_t[0.25]], axis=0)
            tm16_emulti = np.concatenate(
                [mode_data["tmalign"]["e_multi"], (e_tm2_by_t[0.15] + e_tm2_by_t[0.25] + e_tm2_by_t[0.35]) / 3.0], axis=0
            )
            pick_tm16_25 = int(np.argmin(tm16_e25))
            pick_tm16_multi = int(np.argmin(tm16_emulti))
            print(
                f"[P1 tmalign_k16] {pid} ({arm}): mean16={np.mean(tm16_tms):.4f} best16={np.max(tm16_tms):.4f} "
                f"pick(t=0.25)={tm16_tms[pick_tm16_25]:.4f} pick(multi_t)={tm16_tms[pick_tm16_multi]:.4f}",
                flush=True,
            )

            # Multi-SSE Ensemble (pool h_bond + tmalign + tmalign_smooth = 24 candidates, pick lowest ContactEBM energy)
            ens_modes = ("h_bond", "tmalign", "tmalign_smooth")
            ens_tms = sum((mode_data[m]["tms"] for m in ens_modes), [])
            ens_probs = np.concatenate([mode_data[m]["prob"] for m in ens_modes], axis=0)
            ens_c14s = sum((mode_data[m]["c14s"] for m in ens_modes), [])
            ens_e25 = np.concatenate([mode_data[m]["e_by_t"][0.25] for m in ens_modes], axis=0)
            ens_emulti = np.concatenate([mode_data[m]["e_multi"] for m in ens_modes], axis=0)
            pick_ens_25 = int(np.argmin(ens_e25))
            pick_ens_multi = int(np.argmin(ens_emulti))
            pick_ens_src = ens_modes[pick_ens_25 // args.k]
            print(
                f"[P1 sse_ens24] {pid} ({arm}): mean24={np.mean(ens_tms):.4f} best24={np.max(ens_tms):.4f} "
                f"pick(t=0.25)={ens_tms[pick_ens_25]:.4f} (from {pick_ens_src}) pick(multi_t)={ens_tms[pick_ens_multi]:.4f}",
                flush=True,
            )

            # ── 2. Hyperparameter Sweep A: Contact Probability Cutoff tau before c2c ──
            # Note: c2c takes continuous probabilities or thresholded maps. Test thresholding prob at tau vs continuous prob!
            base_prob = mode_data["tmalign"]["prob"][mode_data["tmalign"]["pick25_idx"]]
            tau_results = {}
            for tau_mode in ("continuous", "tau_0.35", "tau_0.40", "tau_0.50", "tau_0.60"):
                if tau_mode == "continuous":
                    p_in = base_prob
                else:
                    tau_val = float(tau_mode.split("_")[1])
                    p_in = (base_prob >= tau_val).astype(np.float32)
                c14_tau = c2c_rollout_atom14(c2c, p_in, aatype_q, Lq, n_pad, c2c_seed, dev)
                pdb_tau = os.path.join(pair_dir, f"tau_{tau_mode}.pdb")
                write_atom14_pdb(pdb_tau, c14_tau, aatype_q, torch.ones(Lq))
                tau_results[tau_mode] = float(tm_vs_native(args.usalign, native_cif, native_ch, pdb_tau))
            print(
                f"[P2 tau_sweep] {pid} ({arm}) tmalign pick: "
                + " ".join(f"{k}={v:.4f}" for k, v in tau_results.items()),
                flush=True,
            )

            # ── 3. Combined Regime C1 & C2: `tmalign` + `sse_ens24` -> ContactEBM Reverse / Langevin Refinement ──
            # We test starting from BOTH `tmalign` pick(t=0.25) and `sse_ens24` pick(t=0.25) across:
            #   - (t0=0.2, K=0): ultra-conservative local denoising
            #   - (t0=0.3, K=0): conservative local denoising
            #   - (t0=0.3, K=10): moderate local predictor-corrector
            #   - (t0=0.4, K=10): medium basin-hop
            #   - (t0=0.5, K=20): deep basin-hop
            # Plus evaluating the trajectory's MINIMUM-ENERGY predictor step (`minE_pred_tm`) in addition to `mid` and `final`!
            combo_langevin = {}
            for start_label, init_prob, init_tm, init_e25, init_c14 in [
                (
                    "tmalign_pick",
                    mode_data["tmalign"]["prob"][mode_data["tmalign"]["pick25_idx"]],
                    mode_data["tmalign"]["pick25_tm"],
                    mode_data["tmalign"]["pick25_e"],
                    mode_data["tmalign"]["c14s"][mode_data["tmalign"]["pick25_idx"]],
                ),
                (
                    "sse_ens24_pick",
                    ens_probs[pick_ens_25],
                    float(ens_tms[pick_ens_25]),
                    float(ens_e25[pick_ens_25]),
                    ens_c14s[pick_ens_25],
                ),
            ]:
                init_bmap = (init_prob >= 0.5).astype(np.uint8)
                traj_list = []
                for t0, K_corr in ((0.2, 0), (0.3, 0), (0.3, 10), (0.4, 10), (0.5, 20)):
                    res_l = cebm_reverse_sample_from_t0(
                        cebm_model, cebm_diff, init_bmap, aatype_q, Lq, t0=t0, K_corrector=K_corr, dev=dev, t_eval=args.t_eval, seed=TRI_SEED
                    )
                    unpacked = unpack_upper_triangle(res_l["packed_clean_steps"], Lq)
                    pred_indices = [idx for idx, kd in enumerate(res_l["step_kinds"]) if kd == "predictor"]
                    mid_idx = pred_indices[len(pred_indices) // 2]
                    # Minimum-energy predictor step along the trajectory (in the second half of the trajectory so t <= t0/2)
                    second_half_preds = pred_indices[len(pred_indices) // 2 :]
                    min_e_idx = min(second_half_preds, key=lambda idx: float(res_l["step_energies_005"][idx]))

                    c14_mid = c2c_rollout_atom14(c2c, unpacked[mid_idx], aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_mid = os.path.join(pair_dir, f"c2_{start_label}_t{int(t0*100):03d}_K{K_corr}_mid.pdb")
                    write_atom14_pdb(pdb_mid, c14_mid, aatype_q, torch.ones(Lq))
                    tm_mid = tm_vs_native(args.usalign, native_cif, native_ch, pdb_mid)

                    c14_mine = c2c_rollout_atom14(c2c, unpacked[min_e_idx], aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_mine = os.path.join(pair_dir, f"c2_{start_label}_t{int(t0*100):03d}_K{K_corr}_minE.pdb")
                    write_atom14_pdb(pdb_mine, c14_mine, aatype_q, torch.ones(Lq))
                    tm_mine = tm_vs_native(args.usalign, native_cif, native_ch, pdb_mine)

                    c14_fin = c2c_rollout_atom14(c2c, res_l["final_binary"].astype(np.float32), aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_fin = os.path.join(pair_dir, f"c2_{start_label}_t{int(t0*100):03d}_K{K_corr}_fin.pdb")
                    write_atom14_pdb(pdb_fin, c14_fin, aatype_q, torch.ones(Lq))
                    tm_fin = tm_vs_native(args.usalign, native_cif, native_ch, pdb_fin)

                    # Soft probability map feeding into c2c at final step
                    c14_fin_soft = c2c_rollout_atom14(c2c, res_l["final_prob"].astype(np.float32), aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_fin_soft = os.path.join(pair_dir, f"c2_{start_label}_t{int(t0*100):03d}_K{K_corr}_finsoft.pdb")
                    write_atom14_pdb(pdb_fin_soft, c14_fin_soft, aatype_q, torch.ones(Lq))
                    tm_fin_soft = tm_vs_native(args.usalign, native_cif, native_ch, pdb_fin_soft)

                    row_tr = {
                        "t0": t0,
                        "K": K_corr,
                        "init_tm": float(init_tm),
                        "init_e25": float(init_e25),
                        "mid_tm": float(tm_mid),
                        "mid_e25": float(res_l["step_energies_005"][mid_idx]),
                        "minE_t": float(res_l["step_times"][min_e_idx]),
                        "minE_e25": float(res_l["step_energies_005"][min_e_idx]),
                        "minE_tm": float(tm_mine),
                        "final_e25": float(res_l["final_e005"]),
                        "final_tm": float(tm_fin),
                        "final_soft_tm": float(tm_fin_soft),
                    }
                    traj_list.append(row_tr)
                    print(
                        f"[C2 {start_label}] {pid} ({arm}) t0={t0:.1f} K={K_corr:2d}: E25 {init_e25:.1f}->{res_l['final_e005']:.1f} "
                        f"(minE={row_tr['minE_e25']:.1f}@t={row_tr['minE_t']:.2f}) | "
                        f"TM init={init_tm:.4f} mid={tm_mid:.4f} minE={tm_mine:.4f} fin={tm_fin:.4f} fin_soft={tm_fin_soft:.4f}",
                        flush=True,
                    )
                combo_langevin[start_label] = traj_list

            # ── 4. Combined Regime C3: `tmalign` Round-0 -> Energy-Gated Round-1 Cycling -> ContactEBM Refinement ──
            # Take the Round-0 `tmalign` pick structure (and `sse_ens24` pick structure), re-featurize with `tmalign_smooth`,
            # sample k=8 Round-1 maps, energy-gate against Round-0, and run (t0=0.3, K=0) & (t0=0.5, K=20) on the winner!
            c3_results = {}
            for c3_label, r0_c14, r0_prob_pick, r0_tm_pick, r0_e25_pick in [
                (
                    "tmalign_cyc1",
                    mode_data["tmalign"]["c14s"][mode_data["tmalign"]["pick25_idx"]],
                    mode_data["tmalign"]["prob"][mode_data["tmalign"]["pick25_idx"]],
                    mode_data["tmalign"]["pick25_tm"],
                    mode_data["tmalign"]["pick25_e"],
                ),
                (
                    "sse_ens24_cyc1",
                    ens_c14s[pick_ens_25],
                    ens_probs[pick_ens_25],
                    float(ens_tms[pick_ens_25]),
                    float(ens_e25[pick_ens_25]),
                ),
            ]:
                full_r1, amask_r1 = coords14_to_atom37(r0_c14, aatype_q)
                ref_r1, n_el_r1, _, _ = template_reference_custom(
                    full_r1, amask_r1, aatype_q, predictor, transform, Lq, pro_donor, dssp_mode="tmalign_smooth"
                )
                topo_r1 = {kk: Proteina._stack_topology([ref_r1[kk]] * args.k).to(dev) for kk in topo_keys}
                L.seed_everything(TRI_SEED + 1)
                res_r1, _, _ = generate_tri_with_trajectory(model, cfg_exp, topo_r1, res_q, mask_q, args.k, Lq)
                prob_r1 = torch.sigmoid(res_r1["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
                e_r1_25, _ = score_maps_with_cebm(cebm_model, cebm_diff, prob_r1, aatype_q, Lq, args.t_eval, dev, seed=TRI_SEED + 1)

                tms_r1 = []
                for si in range(args.k):
                    c14_si = c2c_rollout_atom14(c2c, prob_r1[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                    pdb_si = os.path.join(pair_dir, f"c3_{c3_label}_r1_s{si}.pdb")
                    write_atom14_pdb(pdb_si, c14_si, aatype_q, torch.ones(Lq))
                    tms_r1.append(tm_vs_native(args.usalign, native_cif, native_ch, pdb_si))
                pick_r1 = int(np.argmin(e_r1_25))
                r1_tm_pick = float(tms_r1[pick_r1])
                r1_e25_pick = float(e_r1_25[pick_r1])

                # Energy gate between r0 and r1
                if r1_e25_pick < r0_e25_pick:
                    gated_tm = r1_tm_pick
                    gated_e25 = r1_e25_pick
                    gated_prob = prob_r1[pick_r1]
                    gated_round = 1
                else:
                    gated_tm = float(r0_tm_pick)
                    gated_e25 = float(r0_e25_pick)
                    gated_prob = r0_prob_pick
                    gated_round = 0

                # Run (0.3, K=0) and (0.5, K=20) on the energy-gated winner map
                gated_bmap = (gated_prob >= 0.5).astype(np.uint8)
                res_30 = cebm_reverse_sample_from_t0(
                    cebm_model, cebm_diff, gated_bmap, aatype_q, Lq, t0=0.3, K_corrector=0, dev=dev, t_eval=args.t_eval, seed=TRI_SEED
                )
                c14_30 = c2c_rollout_atom14(c2c, res_30["final_binary"].astype(np.float32), aatype_q, Lq, n_pad, c2c_seed, dev)
                pdb_30 = os.path.join(pair_dir, f"c3_{c3_label}_gated_t030_K0.pdb")
                write_atom14_pdb(pdb_30, c14_30, aatype_q, torch.ones(Lq))
                tm_30 = float(tm_vs_native(args.usalign, native_cif, native_ch, pdb_30))

                res_50 = cebm_reverse_sample_from_t0(
                    cebm_model, cebm_diff, gated_bmap, aatype_q, Lq, t0=0.5, K_corrector=20, dev=dev, t_eval=args.t_eval, seed=TRI_SEED
                )
                c14_50 = c2c_rollout_atom14(c2c, res_50["final_binary"].astype(np.float32), aatype_q, Lq, n_pad, c2c_seed, dev)
                pdb_50 = os.path.join(pair_dir, f"c3_{c3_label}_gated_t050_K20.pdb")
                write_atom14_pdb(pdb_50, c14_50, aatype_q, torch.ones(Lq))
                tm_50 = float(tm_vs_native(args.usalign, native_cif, native_ch, pdb_50))

                # Energy-gated choice across {gated_r01, (0.3, K=0), (0.5, K=20)}
                cands = [
                    ("gated_r01", gated_e25, gated_tm),
                    ("t030_K0", float(res_30["final_e005"]), tm_30),
                    ("t050_K20", float(res_50["final_e005"]), tm_50),
                ]
                best_cand = min(cands, key=lambda x: x[1])

                c3_results[c3_label] = {
                    "r0_tm": float(r0_tm_pick),
                    "r0_e25": float(r0_e25_pick),
                    "r1_mean8_tm": float(np.mean(tms_r1)),
                    "r1_best8_tm": float(np.max(tms_r1)),
                    "r1_pick_tm": r1_tm_pick,
                    "r1_pick_e25": r1_e25_pick,
                    "gated_round": gated_round,
                    "gated_tm": gated_tm,
                    "gated_e25": gated_e25,
                    "t030_K0_tm": tm_30,
                    "t030_K0_e25": float(res_30["final_e005"]),
                    "t050_K20_tm": tm_50,
                    "t050_K20_e25": float(res_50["final_e005"]),
                    "portfolio_minE_src": best_cand[0],
                    "portfolio_minE_e25": best_cand[1],
                    "portfolio_minE_tm": best_cand[2],
                }
                print(
                    f"[C3 {c3_label}] {pid} ({arm}): r0={r0_tm_pick:.4f}(E={r0_e25_pick:.1f}) -> "
                    f"r1={r1_tm_pick:.4f}(E={r1_e25_pick:.1f}, mean8={np.mean(tms_r1):.4f}) | "
                    f"gated(r={gated_round})={gated_tm:.4f} | +0.3/K0={tm_30:.4f} +0.5/K20={tm_50:.4f} | "
                    f"minE_portfolio({best_cand[0]})={best_cand[2]:.4f}",
                    flush=True,
                )

            summary_out = {
                "pid": pid,
                "arm": arm,
                "Lq": Lq,
                "Lt0": Lt0,
                "modes": {
                    m: {k: v for k, v in mode_data[m].items() if k not in ("prob", "c14s", "pdbs", "e_by_t", "e_multi")}
                    for m in sse_modes
                },
                "tmalign_k16": {
                    "mean16_tm": float(np.mean(tm16_tms)),
                    "best16_tm": float(np.max(tm16_tms)),
                    "pick25_tm": float(tm16_tms[pick_tm16_25]),
                    "pick_multi_tm": float(tm16_tms[pick_tm16_multi]),
                },
                "sse_ens24": {
                    "mean24_tm": float(np.mean(ens_tms)),
                    "best24_tm": float(np.max(ens_tms)),
                    "pick25_tm": float(ens_tms[pick_ens_25]),
                    "pick25_src": pick_ens_src,
                    "pick_multi_tm": float(ens_tms[pick_ens_multi]),
                },
                "tau_sweep": tau_results,
                "combo_langevin": combo_langevin,
                "c3_joint": c3_results,
                "elapsed_sec": round(time.time() - t_start_target, 1),
            }
            with open(os.path.join(pair_dir, "combo_results.json"), "w") as fh:
                json.dump(summary_out, fh, indent=2)
            print(f"[DONE] {pid} ({arm}) in {summary_out['elapsed_sec']:.1f}s", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
