"""TopoRef inference regimes benchmark (Tasks 1.0, 1a, 1b, and Task 2 evaluation).

Modes:
  filter        Step A0: filter targets where default tri+c2c mean TM >= 0.5 (preferring non-self Foldseek
                templates in d36tc_A_tmpl if >= 4 targets qualify, else falling back to self-reference d36tc_A_self).
  traj_aa       Task 1.0: pairwise amino-acid contact compatibility log-odds S_ab and JS divergence across the
                TopoRef sampling trajectory (t in [0, 1]) vs natural reference, native map, and template map.
  cycling       Task 1a: 5-round iterative cycling: TopoRef (k=8) -> ContactEBM (t_eval=0.05) best map -> c2c ->
                re-featurize predicted structure as next round's template (both h_bond and tmalign_smooth DSSP).
  cebm_langevin Task 1b: TopoRef (k=8) -> ContactEBM (t_eval=0.05) best map -> forward-noise to t0 in {0.3, 0.4, 0.5}
                -> reverse sampling with K in {0, 20} corrector steps -> bit-packed uint8 clean maps at every step ->
                c2c folding and Energy-vs-TM correlation.
  sse_tmalign   Task 2: compare dssp_def in {h_bond, tmalign, tmalign_smooth} on template featurization + TopoRef + c2c.
  all           Run filter -> traj_aa -> cycling -> cebm_langevin -> sse_tmalign sequentially on the filtered targets.
"""

import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from contact_ebm.diffusion.schedules import symmetrize_matrix
from contact_ebm.model.contact_ebm import ContactEBM
from contact_ebm.scripts.train import build_diffusion
from Frame2ConFind.inference.api import Frame2ConFindPredictor
from gen_c2c_structures import MODEL_CFG
from proteinfoundation.datasets.atom_features import atom14_features
from proteinfoundation.datasets.cath_utils import load_cath_mapping
from proteinfoundation.datasets.sse_topology import DSSP_HELIX, DSSP_STRAND, STRUCTURAL_PAIR_FEATURES
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
from proteinfoundation.nn.af3_diffusion import C2C_INFERENCE_STEPS, SIGMA_DATA
from proteinfoundation.openfold_stub.np import residue_constants as rc
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils import precompute_synthetic_topology_index as builder
from proteinfoundation.utils.c2c_dump import write_atom14_pdb
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from synth_tri_bench import (
    C2C_SEED_BASE,
    EXP_CFG,
    REF_AUG_OFF,
    REF_EXPECT,
    TRI_SEED,
    build_transform,
    check_index_matches_builder,
    load_c2c,
    load_predictor,
    load_tri,
    structure_atom37,
)

BAD_AFDB_22 = (
    "6VQH_f", "6VYR_9", "6Y4F_B", "6Z9I_B", "7A0S_I", "7A0S_U", "7B7E_B", "7K0P_A",
    "7K9L_C", "7N0I_E", "7P3H_G", "7QEP_L1", "7R5K_A", "7S2R_B", "7S5L_E", "7T7Q_B",
    "7UEB_E", "7W6B_B", "7XHS_A", "7YR6_E", "8F3K_A", "8GZU_42",
)


def build_toporef_transform(index_path):
    """TopologyReferenceTransform with the cfft dataset config's values and every augmentation off."""
    with hydra.initialize_config_dir(os.path.join(REPO, "configs", "experiment_config"), version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=EXP_CFG)
    ds_dir = os.path.join(REPO, "configs", "datasets_config", cfg_exp.dataset_config_subdir)
    with hydra.initialize_config_dir(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    ents = [t for t in cfg_data.datamodule.transforms if t._target_.endswith("TopologyReferenceTransform")]
    assert len(ents) == 1, f"{cfg_exp.dataset}: {len(ents)} TopologyReferenceTransform entries"
    kw = OmegaConf.to_container(ents[0], resolve=False)
    kw.pop("index_path")
    kw.pop("_target_")
    for k, v in REF_EXPECT.items():
        assert kw[k] == v, f"dataset config {k}={kw[k]!r}, expected {v!r}"
    kw.update(REF_AUG_OFF)
    return TopologyReferenceTransform(index_path=index_path, **kw)


def native_for(natives_dir, pid):
    entry, chain = pid.rsplit("_", 1)
    native = os.path.join(natives_dir, entry + ".cif")
    assert os.path.exists(native), f"{pid}: missing native cif {native}"
    return native, chain


def tm_vs_native(usalign, native, chain, pred_pdb):
    r = subprocess.run(
        [usalign, native, "-chain1", chain, pred_pdb, "-TMscore", "5", "-outfmt", "2"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 7 and not ln.startswith("#"):
            return float(p[2])
    raise RuntimeError(f"no USalign row for {pred_pdb}: {r.stdout[:300]} {r.stderr[:300]}")


def tm_pair_seqdep(usalign, pdb1, pdb2):
    r = subprocess.run(
        [usalign, pdb1, pdb2, "-TMscore", "5", "-outfmt", "2"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 7 and not ln.startswith("#"):
            return float(p[2])
    raise RuntimeError(f"no USalign row for {pdb1} vs {pdb2}")


def tm_pair_seqindep(usalign, pdb_q, pdb_t):
    r = subprocess.run(
        [usalign, pdb_q, pdb_t, "-TMscore", "0", "-outfmt", "2"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 7 and not ln.startswith("#"):
            return float(p[2])
    raise RuntimeError(f"no USalign row for {pdb_q} vs {pdb_t}")


def graph_to_atom37(g):
    coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    amask = (g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float() > 0.5).bool()
    full = torch.where(amask[..., None], coords, torch.zeros_like(coords))
    aatype = g.residue_type.long()
    return full, amask, aatype


def coords14_to_atom37(coords14, aatype):
    """Convert c2c [L, 14, 3] atom14 coordinates to (full [L, 37, 3], amask [L, 37] bool)."""
    L = int(coords14.shape[0])
    full = torch.zeros(L, 37, 3, dtype=torch.float32)
    amask = torch.zeros(L, 37, dtype=torch.bool)
    c14 = torch.as_tensor(coords14, dtype=torch.float32).cpu()
    aa = aatype.long().cpu()
    for i in range(L):
        ai = int(aa[i].item())
        aa3 = rc.restype_1to3.get(rc.restypes[ai], "UNK") if ai < len(rc.restypes) else "UNK"
        names = rc.restype_name_to_atom14_names.get(aa3, [""] * 14)
        for j, nm in enumerate(names):
            if nm and nm in rc.atom_order:
                slot37 = rc.atom_order[nm]
                xyz = c14[i, j]
                if bool(torch.isfinite(xyz).all()):
                    full[i, slot37] = xyz
                    amask[i, slot37] = True
    return full, amask


def template_reference_with_dssp(
    full, amask, aatype, predictor, transform, length, pro_donor, dssp_def="h_bond", cm_t_override=None
):
    """Build topology reference dict with configurable dssp_def ('h_bond', 'tmalign', 'tmalign_smooth')."""
    Lt = int(amask.shape[0])
    donor = torch.from_numpy(aatype.cpu().numpy() != builder.PRO_IDX)[None] if pro_donor else None
    dssp_t = compute_dssp_target(
        full[None],
        amask[:, builder.CA_IDX][None],
        coord_mask=amask[None],
        coord_layout="atom37",
        donor_mask=donor,
        dssp_def=dssp_def,
    )
    assert dssp_t is not None, f"template DSSP is None for dssp_def={dssp_def}"
    if cm_t_override is not None:
        cm_t = torch.as_tensor(cm_t_override, dtype=torch.float32).cpu()
    else:
        x, mask = builder._f2c_inputs(full[None], amask)
        probs = predictor.predict_batch(x, mask[None])[:, :Lt, :Lt].float().cpu()
        cm_t = (probs[0] >= builder.CONFIND_THRESHOLD).float()
    row = builder.topology_row(cm_t, full, amask, dssp_t[0], REF_EXPECT["min_len"])
    if row is None:
        # Fallback to empty reference if no H/E residue is assigned
        ref = transform.assemble_reference([], torch.zeros(0, 0), torch.zeros(0, 0, len(STRUCTURAL_PAIR_FEATURES)), length, augment=False)
        elem_t = torch.full((Lt,), builder.ALIGN_NONE, dtype=torch.int64)
        return ref, elem_t, 0, False, dssp_t[0], cm_t
    runs, ref_bytes, n_el, feat_bytes, _stats, elem = row
    runs = [(int(t), int(n)) for t, n in runs]
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    if n_el > 0:
        he_contact = torch.frombuffer(bytearray(ref_bytes), dtype=torch.uint8).reshape(n_el, n_el).float()
        structural = torch.frombuffer(bytearray(feat_bytes), dtype=torch.float16).reshape(n_el, n_el, n_struct).float()
    else:
        he_contact = torch.zeros(0, 0)
        structural = torch.zeros(0, 0, n_struct)
    ref = transform.assemble_reference(runs, he_contact, structural, length, augment=False)
    n_he = int(ref["topology_he_tokens"].numel())
    elem_t = torch.from_numpy(elem.astype(np.int64))
    elem_t = torch.where(elem_t < n_he, elem_t, torch.full_like(elem_t, builder.ALIGN_NONE))
    n_tok = sum(1 for t, n in runs if t in transform.alphabet.types and n >= transform.alphabet.min_len)
    truncated = n_el > transform.max_topology_he_len or n_tok > transform.max_topology_len
    return ref, elem_t, n_el, truncated, dssp_t[0], cm_t


def c2c_rollout_atom14(c2c, prob, aatype, L, Lp, seed, dev):
    """Run ContactToCoord rollout and return full atom14 coordinates [L, 14, 3] in Angstrom."""
    mask = torch.zeros(1, Lp, device=dev)
    mask[0, :L] = 1.0
    aa = torch.zeros(1, Lp, dtype=torch.long, device=dev)
    aa[0, :L] = aatype.to(dev)
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aa, mask)
    contacts = torch.zeros(1, Lp, Lp, device=dev)
    contacts[0, :L, :L] = torch.as_tensor(prob, dtype=torch.float32, device=dev)
    with torch.no_grad():
        s, z, _ = c2c.model.encode(contacts, aa, mask)
        torch.manual_seed(seed)
        coords = c2c.model.rollout(s, z, mask, ref_feats, ref_pos, a2t, amask, ruid, n_steps=C2C_INFERENCE_STEPS)
    return coords.reshape(-1, Lp, 14, 3)[0, :L, :, :].float().cpu().numpy()


def pack_upper_triangle(binary_maps, L):
    """Pack [N, L, L] uint8/bool 0/1 maps into upper-triangle bit-packed uint8 [N, n_bytes]."""
    iu = np.triu_indices(L, k=1)
    flat = np.asarray(binary_maps, dtype=np.uint8)[:, iu[0], iu[1]]
    return np.packbits(flat, axis=1)


def unpack_upper_triangle(packed, L):
    """Unpack [N, n_bytes] bit-packed upper triangle back to symmetric [N, L, L] float32 maps."""
    iu = np.triu_indices(L, k=1)
    n_pairs = len(iu[0])
    bits = np.unpackbits(packed, axis=1)[:, :n_pairs].astype(np.float32)
    N = packed.shape[0]
    out = np.zeros((N, L, L), dtype=np.float32)
    out[:, iu[0], iu[1]] = bits
    out[:, iu[1], iu[0]] = bits
    return out


# ───────────────────────── Task 1.0: Pairwise AA Contact Compatibility ─────────────────────────
def build_natural_aa_pair_stats(processed_dir, stems, min_sep=6, pseudocount=1.0):
    """Compute natural 20x20 contact pairwise amino-acid distribution P_nat(a,b) and log-odds matrix S_ab."""
    contact_counts = np.full((20, 20), pseudocount, dtype=np.float64)
    total_counts = np.full((20, 20), pseudocount * 20.0, dtype=np.float64)
    n_chains = 0
    for st in stems:
        pt = os.path.join(processed_dir, f"{st}.pt")
        if not os.path.exists(pt):
            continue
        g = torch.load(pt, map_location="cpu", weights_only=False)
        cm_raw = getattr(g, "contact_map_confind", None)
        aa = getattr(g, "residue_type", None)
        if cm_raw is None or aa is None:
            continue
        L = int(aa.shape[0])
        if tuple(cm_raw.shape) != (L, L) or L <= min_sep:
            continue
        cm = (cm_raw.float().numpy() >= builder.CONFIND_THRESHOLD).astype(np.float64)
        aa_np = aa.long().numpy()
        iu = np.triu_indices(L, k=min_sep)
        a_i = aa_np[iu[0]]
        a_j = aa_np[iu[1]]
        valid = (a_i >= 0) & (a_i < 20) & (a_j >= 0) & (a_j < 20)
        a_i, a_j, c_ij = a_i[valid], a_j[valid], cm[iu[0][valid], iu[1][valid]]
        np.add.at(total_counts, (a_i, a_j), 1.0)
        np.add.at(total_counts, (a_j, a_i), 1.0)
        np.add.at(contact_counts, (a_i, a_j), c_ij)
        np.add.at(contact_counts, (a_j, a_i), c_ij)
        n_chains += 1
    p_nat_pair = contact_counts / contact_counts.sum()
    p_bg_pair = total_counts / total_counts.sum()
    p_contact_given_aa = contact_counts / total_counts
    p_contact_overall = contact_counts.sum() / total_counts.sum()
    log_odds = np.log(p_contact_given_aa / p_contact_overall)
    return {
        "n_chains": n_chains,
        "p_nat_pair": p_nat_pair.astype(np.float32),
        "p_bg_pair": p_bg_pair.astype(np.float32),
        "log_odds": log_odds.astype(np.float32),
        "p_contact_overall": float(p_contact_overall),
    }


def score_map_aa_compatibility(prob_map, aatype, aa_stats, min_sep=6):
    """Compute soft/hard AA-pair log-odds compatibility score and JS divergence vs P_nat(a,b)."""
    L = int(aatype.shape[0])
    iu = np.triu_indices(L, k=min_sep)
    aa_np = aatype.long().cpu().numpy()
    a_i = aa_np[iu[0]].clip(0, 19)
    a_j = aa_np[iu[1]].clip(0, 19)
    s_ij = aa_stats["log_odds"][a_i, a_j]
    p_ij = np.asarray(prob_map, dtype=np.float64)[iu[0], iu[1]]
    b_ij = (p_ij >= 0.5).astype(np.float64)

    soft_score = float((p_ij * s_ij).sum() / max(p_ij.sum(), 1e-8))
    hard_score = float((b_ij * s_ij).sum() / max(b_ij.sum(), 1.0))

    # Empirical 20x20 contact AA-pair distribution of this map (soft-weighted and hard-thresholded)
    emp_counts = np.full((20, 20), 1e-6, dtype=np.float64)
    np.add.at(emp_counts, (a_i, a_j), p_ij)
    np.add.at(emp_counts, (a_j, a_i), p_ij)
    p_emp = emp_counts / emp_counts.sum()
    p_nat = aa_stats["p_nat_pair"].astype(np.float64)
    m = 0.5 * (p_emp + p_nat)
    kl_emp = np.sum(p_emp * np.log(np.clip(p_emp / np.clip(m, 1e-12, None), 1e-12, None)))
    kl_nat = np.sum(p_nat * np.log(np.clip(p_nat / np.clip(m, 1e-12, None), 1e-12, None)))
    jsd = float(0.5 * (kl_emp + kl_nat))
    return {
        "soft_log_odds": soft_score,
        "hard_log_odds": hard_score,
        "jsd_vs_nat": jsd,
        "n_pred_contacts": float(b_ij.sum()),
    }


def contact_metrics_vs_gt(prob_map, gt_map, min_sep=6):
    """Compute P@L, F1, precision, and recall on |i - j| >= min_sep."""
    L = int(gt_map.shape[0])
    iu = np.triu_indices(L, k=min_sep)
    p = np.asarray(prob_map, dtype=np.float64)[iu[0], iu[1]]
    g = (np.asarray(gt_map, dtype=np.float64)[iu[0], iu[1]] >= 0.5).astype(np.float64)
    b = (p >= 0.5).astype(np.float64)
    tp = float((b * g).sum())
    fp = float((b * (1.0 - g)).sum())
    fn = float(((1.0 - b) * g).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-8)
    k_top = min(L, len(p))
    top_idx = np.argpartition(-p, k_top - 1)[:k_top] if k_top > 0 else np.array([], dtype=int)
    p_at_l = float(g[top_idx].mean()) if k_top > 0 else 0.0
    return {"p_at_l": p_at_l, "f1": f1, "prec": prec, "rec": rec}


def generate_tri_with_trajectory(model, cfg_exp, topology, residue_type, mask, k, Lq):
    """Run model.generate while capturing per-step clean contact probabilities and align logits."""
    vs = cfg_exp.validation_sampling
    cath_code_indices = cath_code_indices_mask = None
    if cfg_exp.training.get("fold_cond", False):
        cath_code_dir = cfg_exp.model.nn.get("cath_code_dir")
        if cath_code_dir is not None:
            _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
            cath_code_indices = torch.zeros((k, 1, 3), device=model.device, dtype=torch.long)
            cath_code_indices[:, 0, 0], cath_code_indices[:, 0, 1], cath_code_indices[:, 0, 2] = nC, nA, nT
            cath_code_indices_mask = torch.zeros((k, 1), device=model.device, dtype=torch.bool)

    step_records = []
    rec = {"n": 0, "first": None, "last": None}

    def _hook(_mod, inp, out):
        if isinstance(out, dict) and "contact_map_logits" in out:
            nn_in = inp[0]
            t_val = float(nn_in["t"][0].detach().cpu().item())
            p_clean = torch.sigmoid(out["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
            step_records.append((t_val, p_clean))
        if isinstance(out, dict) and "align_logits" in out:
            cur = {kk: out[kk].detach().float().cpu() for kk in ("align_logits", "align_none_logits")}
            rec["first"] = cur if rec["first"] is None else rec["first"]
            rec["last"] = cur
            rec["n"] += 1

    hooks = [
        m.register_forward_hook(_hook)
        for m in {id(m): m for m in (model.nn, getattr(model, "nn_sc", None)) if m is not None}.values()
    ]
    with torch.no_grad():
        result = model.generate(
            nsamples=k,
            n=mask.shape[1],
            dt=float(vs.get("dt", 0.005)),
            self_cond=cfg_exp.training.self_cond,
            cath_code=None,
            cath_code_indices=cath_code_indices,
            cath_code_indices_mask=cath_code_indices_mask,
            residue_type=residue_type,
            guidance_weight=1.0,
            autoguidance_ratio=0.0,
            dtype=torch.float32,
            schedule_mode=vs.get("schedule_mode", "uniform"),
            schedule_p=float(vs.get("schedule_p", 1.0)),
            sampling_mode=vs.get("sampling_mode", "sc"),
            sc_scale_noise=float(vs.get("sc_scale_noise", 0.45)),
            sc_scale_score=float(vs.get("sc_scale_score", 1.0)),
            gt_mode="us",
            gt_p=1.0,
            gt_clamp_val=None,
            mask=mask,
            x_motif=None,
            fixed_sequence_mask=None,
            fixed_structure_mask=None,
            topology=topology,
            zero_sin_pos_emb=bool(cfg_exp.training.get("zero_sin_pos_emb", False)),
        )
    for h in hooks:
        h.remove()
    return result, rec, step_records


# ───────────────────────── ContactEBM Loader, Scorer & Reverse Sampler ─────────────────────────
def load_contact_ebm(cebm_repo, cebm_ckpt, dev):
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
    print(f"[cebm] loaded {cebm_ckpt} arch={mc['arch']} step={ck.get('step')} ema={'ema' in ck}", flush=True)
    return model, diffusion, cfg


def score_maps_with_cebm(cebm_model, cebm_diff, prob_maps, aatype, Lq, t_eval, dev, seed=0):
    """Score k contact maps [k, Lq, Lq] with ContactEBM at t_eval (both clean input and forward-noised z_t)."""
    k = prob_maps.shape[0]
    binary_tokens = torch.from_numpy((prob_maps >= 0.5).astype(np.int64)).to(dev)
    aa_batch = aatype[:Lq].long().to(dev)[None].expand(k, Lq)
    rmask = torch.ones((k, Lq), dtype=torch.float32, device=dev)
    pair_mask = cebm_model._pair_mask(rmask)
    t_tensor = torch.full((k,), float(t_eval), device=dev, dtype=torch.float32)
    with torch.no_grad():
        e_clean = cebm_model.score_energy(binary_tokens, aa_batch, rmask, t_eval=float(t_eval)).detach().float().cpu().numpy()
        torch.manual_seed(seed)
        zt = cebm_diff.forward_sample(binary_tokens, t_tensor, pair_mask)
        e_noised = cebm_model.energy(cebm_model._onehot(zt), aa_batch, rmask, t_tensor).detach().float().cpu().numpy()
    return e_clean, e_noised


def cebm_reverse_sample_from_t0(cebm_model, cebm_diff, init_binary_map, aatype, Lq, t0, K_corrector, dev, seed=0):
    """Forward-noise init_binary_map [Lq, Lq] to t0 and run reverse sampling (with K_corrector steps/interval).

    Saves EVERY network call's clean contact prediction (sigmoid(logits) >= 0.5) as bit-packed uint8 upper triangle.
    """
    torch.manual_seed(seed)
    x0 = torch.from_numpy((init_binary_map >= 0.5).astype(np.int64)).to(dev)[None]
    aa = aatype[:Lq].long().to(dev)[None]
    rmask = torch.ones((1, Lq), dtype=torch.float32, device=dev)
    pair_mask = cebm_model._pair_mask(rmask)
    pm_b = pair_mask.bool()
    diag = torch.arange(Lq, device=dev)

    timesteps = int(cebm_diff.timesteps)
    # Find grid index i0 where grid time t <= t0
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

        step_times = []
        step_kinds = []
        step_energies_t = []
        step_energies_005 = []
        step_clean_maps = []

        def _renoise(z_s, s_val, t_val):
            s_t = torch.full((1,), float(s_val), device=dev)
            t_t = torch.full((1,), float(t_val), device=dev)
            alpha_t = cebm_diff.schedule.alpha(t_t).view(1, 1, 1)
            alpha_s = cebm_diff.schedule.alpha(s_t).view(1, 1, 1)
            keep_prob = (alpha_t / alpha_s.clamp(min=1e-6)).clamp(0.0, 1.0)
            keep = torch.bernoulli(keep_prob.expand_as(z_s)).bool()
            if hasattr(cebm_diff, "mask_id"):
                fresh = torch.full_like(z_s, cebm_diff.mask_id)
            else:
                fresh = cebm_diff.prior_sample(z_s.shape, device=dev)
            out = torch.where(keep, z_s, fresh)
            out = symmetrize_matrix(out.float()).long()
            out[:, diag, diag] = 0
            return out * pm_b.long()

        for i in range(i0, timesteps):
            s, t = cebm_diff.get_sampling_grid(i, timesteps)
            dt = t - s
            t_tensor = torch.full((1,), float(t), device=dev, dtype=torch.float32)
            if K_corrector > 0 and s > 0:
                for kk in range(K_corrector):
                    logits, energy_t, _ = cebm_model.compute_outputs(
                        cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False
                    )
                    p_clean = torch.sigmoid(logits.detach().float())[0, :Lq, :Lq].cpu().numpy()
                    b_clean = (p_clean >= 0.5).astype(np.uint8)
                    e005 = float(
                        cebm_model.score_energy(
                            torch.from_numpy(b_clean.astype(np.int64)).to(dev)[None], aa, rmask, t_eval=0.05
                        )[0].item()
                    )
                    step_times.append(float(t))
                    step_kinds.append(f"corrector_{kk}")
                    step_energies_t.append(float(energy_t[0].item()))
                    step_energies_005.append(e005)
                    step_clean_maps.append(b_clean)

                    zc = cebm_diff.sample_step(zt, logits.detach(), s, t, pair_mask=pair_mask)
                    zt = _renoise(zc, s, t)

            logits, energy_t, _ = cebm_model.compute_outputs(
                cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False
            )
            p_clean = torch.sigmoid(logits.detach().float())[0, :Lq, :Lq].cpu().numpy()
            b_clean = (p_clean >= 0.5).astype(np.uint8)
            e005 = float(
                cebm_model.score_energy(
                    torch.from_numpy(b_clean.astype(np.int64)).to(dev)[None], aa, rmask, t_eval=0.05
                )[0].item()
            )
            step_times.append(float(t))
            step_kinds.append("predictor")
            step_energies_t.append(float(energy_t[0].item()))
            step_energies_005.append(e005)
            step_clean_maps.append(b_clean)

            zt = cebm_diff.sample_step(zt, logits.detach(), s, t, pair_mask=pair_mask)

        z0 = cebm_diff.decode(zt, logits.detach(), pair_mask=pair_mask)
        final_binary = z0[0, :Lq, :Lq].detach().cpu().numpy().astype(np.uint8)
        final_prob = torch.sigmoid(logits.detach().float())[0, :Lq, :Lq].cpu().numpy()
        final_e005 = float(
            cebm_model.score_energy(z0[:, :Lq, :Lq].long(), aa, rmask, t_eval=0.05)[0].item()
        )

    packed_clean = pack_upper_triangle(np.stack(step_clean_maps, axis=0), Lq)
    return {
        "t0": float(t0),
        "t_start": float(t_start),
        "K": int(K_corrector),
        "n_calls": len(step_times),
        "step_times": np.asarray(step_times, dtype=np.float32),
        "step_kinds": np.asarray(step_kinds),
        "step_energies_t": np.asarray(step_energies_t, dtype=np.float32),
        "step_energies_005": np.asarray(step_energies_005, dtype=np.float32),
        "packed_clean_steps": packed_clean,
        "final_binary": final_binary,
        "final_prob": final_prob.astype(np.float16),
        "final_e005": final_e005,
    }


# ───────────────────────── Step A0: Target Filtering ─────────────────────────
def filter_targets(args):
    """Filter bad_afdb / D36tc Group A targets with default tri+c2c mean TM >= 0.5."""
    group_a_file = os.path.join(args.d36tc_dir, "inputs", "groupA.txt")
    tmap_file = os.path.join(args.d36tc_dir, "inputs", "d36tc_template_map.json")
    proc_dir = os.path.join(args.d36tc_dir, "data", "processed")
    c2c_dir = os.path.join(args.d36tc_dir, "c2c")
    c2c_pdb_dir = os.path.join(args.d36tc_dir, "c2c_pdb")

    stems = [ln.strip() for ln in open(group_a_file) if ln.strip()]
    tmap = json.load(open(tmap_file))

    def _load_arm_tms(arm):
        label = f"d36tc_A_{arm}"
        jpath = os.path.join(c2c_dir, f"B_{label}.jsonl")
        by_pid = collections.defaultdict(list)
        if os.path.exists(jpath):
            for ln in open(jpath):
                if ln.strip():
                    r = json.loads(ln)
                    if "tm" in r and r["tm"] is not None:
                        by_pid[r["stem"]].append(float(r["tm"]))
        if len(by_pid) < len(stems):
            for st in stems:
                if len(by_pid.get(st, [])) == 8:
                    continue
                pdbs = sorted(glob.glob(os.path.join(c2c_pdb_dir, label, "*", st, "*_gen.pdb")))
                if len(pdbs) == 8:
                    native, ch = native_for(args.natives_dir, st)
                    by_pid[st] = [tm_vs_native(args.usalign, native, ch, p) for p in pdbs]
        return by_pid

    tms_tmpl = _load_arm_tms("tmpl")
    tms_self = _load_arm_tms("self")
    tms_nat = _load_arm_tms("native")

    rows = []
    for st in stems:
        pt = os.path.join(proc_dir, f"{st}.pt")
        g = torch.load(pt, map_location="cpu", weights_only=False)
        Lq = int(g.coords.shape[0])
        ref_stem = tmap.get(st, st)
        gt = torch.load(os.path.join(proc_dir, f"{ref_stem}.pt"), map_location="cpu", weights_only=False)
        Lt = int(gt.coords.shape[0])
        vt = tms_tmpl.get(st, [])
        vs = tms_self.get(st, [])
        vn = tms_nat.get(st, [])
        rows.append({
            "pid": st,
            "in_bad_afdb22": st in BAD_AFDB_22,
            "Lq": Lq,
            "ref_stem": ref_stem,
            "Lt": Lt,
            "tmpl_mean8": float(np.mean(vt)) if vt else float("nan"),
            "tmpl_median8": float(np.median(vt)) if vt else float("nan"),
            "tmpl_best8": float(np.max(vt)) if vt else float("nan"),
            "self_mean8": float(np.mean(vs)) if vs else float("nan"),
            "self_median8": float(np.median(vs)) if vs else float("nan"),
            "self_best8": float(np.max(vs)) if vs else float("nan"),
            "native_mean8": float(np.mean(vn)) if vn else float("nan"),
        })

    if args.only_bad_afdb22:
        cand_rows = [r for r in rows if r["in_bad_afdb22"]]
    else:
        cand_rows = list(rows)

    tmpl_pass_all30 = [r for r in rows if r["tmpl_mean8"] >= args.tm_threshold]
    self_pass_all30 = [r for r in rows if r["self_mean8"] >= args.tm_threshold]
    tmpl_pass_22 = [r for r in rows if r["in_bad_afdb22"] and r["tmpl_mean8"] >= args.tm_threshold]
    self_pass_22 = [r for r in rows if r["in_bad_afdb22"] and r["self_mean8"] >= args.tm_threshold]

    tmpl_pass = [r for r in cand_rows if r["tmpl_mean8"] >= args.tm_threshold]
    self_pass = [r for r in cand_rows if r["self_mean8"] >= args.tm_threshold]

    # User rule: prefer non-self Foldseek templates if >= 4 targets remain; else fall back to self-reference
    if len(tmpl_pass) >= 4:
        selected_arm = "tmpl"
        selected = tmpl_pass
    else:
        selected_arm = "self"
        selected = self_pass

    print("\n=== STEP A0: TARGET FILTERING SUMMARY ===")
    print(f"Threshold: default tri+c2c mean TM >= {args.tm_threshold:.2f}")
    print(f"All Group A (n={len(rows)}, L<=384 with Foldseek hit):")
    print(f"  non-self Foldseek template (tmpl_mean8 >= {args.tm_threshold:.2f}): {len(tmpl_pass_all30)} / {len(rows)} targets: {[r['pid'] for r in tmpl_pass_all30]}")
    print(f"  self-reference template    (self_mean8 >= {args.tm_threshold:.2f}): {len(self_pass_all30)} / {len(rows)} targets: {[r['pid'] for r in self_pass_all30]}")
    print(f"Subset in bad_afdb-22 (n={sum(r['in_bad_afdb22'] for r in rows)} of 22 in Group A):")
    print(f"  non-self Foldseek template (tmpl_mean8 >= {args.tm_threshold:.2f}): {len(tmpl_pass_22)} targets: {[r['pid'] for r in tmpl_pass_22]}")
    print(f"  self-reference template    (self_mean8 >= {args.tm_threshold:.2f}): {len(self_pass_22)} targets: {[r['pid'] for r in self_pass_22]}")
    print(f"Selected template regime (rule: non-self if >= 4 else self): arm={selected_arm!r}, n={len(selected)} targets: {[r['pid'] for r in selected]}")
    print(f"\n{'pid':<10} {'in22':<5} {'Lq':>4} {'ref_stem':<10} {'Lt':>4} | {'tmpl_mean':>9} {'tmpl_med':>8} {'tmpl_best':>9} | {'self_mean':>9} {'self_med':>8} {'self_best':>9} | {'nat_mean':>8}")
    for r in rows:
        mark = "*" if any(s["pid"] == r["pid"] for s in selected) else " "
        print(
            f"{r['pid'] + mark:<10} {str(r['in_bad_afdb22']):<5} {r['Lq']:>4d} {r['ref_stem']:<10} {r['Lt']:>4d} | "
            f"{r['tmpl_mean8']:>9.4f} {r['tmpl_median8']:>8.4f} {r['tmpl_best8']:>9.4f} | "
            f"{r['self_mean8']:>9.4f} {r['self_median8']:>8.4f} {r['self_best8']:>9.4f} | "
            f"{r['native_mean8']:>8.4f}"
        )

    os.makedirs(args.out, exist_ok=True)
    filter_out = {
        "tm_threshold": args.tm_threshold,
        "only_bad_afdb22": bool(args.only_bad_afdb22),
        "selected_arm": selected_arm,
        "selected_pids": [r["pid"] for r in selected],
        "tmpl_pass_all30": [r["pid"] for r in tmpl_pass_all30],
        "self_pass_all30": [r["pid"] for r in self_pass_all30],
        "tmpl_pass_22": [r["pid"] for r in tmpl_pass_22],
        "self_pass_22": [r["pid"] for r in self_pass_22],
        "rows": rows,
    }
    with open(os.path.join(args.out, "filter_summary.json"), "w") as fh:
        json.dump(filter_out, fh, indent=2)
    return filter_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("filter", "traj_aa", "cycling", "cebm_langevin", "sse_tmalign", "all"), default="all")
    ap.add_argument("--d36tc-dir", default="/orcd/pool/006/chenxiou/d36tc")
    ap.add_argument("--natives-dir", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    ap.add_argument("--tri-ckpt", default="/orcd/scratch/orcd/011/chenxiou/stageA/ckpt/cfft_tri_last-EMA_step60505.ckpt")
    ap.add_argument("--c2c-ckpt", default="/orcd/scratch/orcd/011/chenxiou/c2c_store/c2c_confind_tbeta/final_twin_s29000.ckpt")
    ap.add_argument("--index", default="/orcd/pool/006/chenxiou/d36tc/data/topology_index_d36tc.pt")
    ap.add_argument("--f2c-ckpt-dir", default="/orcd/scratch/orcd/011/chenxiou/Frame2ConFind/runs/f2s_v384_smoothap-pearson")
    ap.add_argument("--f2c-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--cebm-repo", default="/orcd/scratch/orcd/011/chenxiou/ContactEBM")
    ap.add_argument("--cebm-ckpt", default="/orcd/scratch/orcd/011/chenxiou/ContactEBM/runs/confind_a1_prod100/checkpoints/model_step_100000.pt")
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tm-threshold", type=float, default=0.5)
    ap.add_argument("--only-bad-afdb22", action="store_true", help="Restrict candidate pool strictly to the 22 bad_afdb-22 targets")
    ap.add_argument("--arm-override", choices=("auto", "tmpl", "self", "both"), default="both",
                    help="Which template arm(s) to run on the filtered targets ('both' runs non-self tmpl AND self)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pad-to", type=int, default=384)
    ap.add_argument("--n-cycles", type=int, default=5)
    ap.add_argument("--t-eval", type=float, default=0.05)
    ap.add_argument("--max-targets", type=int, default=0, help="0 = all filtered targets")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    filt = filter_targets(args)
    if args.mode == "filter":
        return 0

    assert os.environ.get("SIGMA_DATA") is not None and SIGMA_DATA == 16.0, f"SIGMA_DATA={SIGMA_DATA} (env must be 16)"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc_dir = os.path.join(args.d36tc_dir, "data", "processed")
    all_pt_stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(proc_dir, "*.pt")))
    aa_stats = build_natural_aa_pair_stats(proc_dir, all_pt_stems)
    np.savez_compressed(os.path.join(args.out, "natural_aa_pair_stats.npz"), **aa_stats)
    print(f"[aa_stats] built 20x20 natural ConFind AA-pair matrix from {aa_stats['n_chains']} chains "
          f"(overall contact rate |i-j|>=6: {aa_stats['p_contact_overall']:.4f})", flush=True)

    transform = build_toporef_transform(args.index)
    transform._ensure_loaded()
    pro_donor = check_index_matches_builder(transform._index)
    predictor = load_predictor(args.f2c_ckpt_dir, args.f2c_dtype) if args.f2c_ckpt_dir else None
    assert predictor is not None or args.mode not in ("cycling", "all"), "--f2c-ckpt-dir required for cycling"
    model, cfg_exp = load_tri(args.tri_ckpt, os.path.join(args.out, ".store_tri"))
    model = model.to(dev).eval()
    c2c = load_c2c(args.c2c_ckpt).to(dev).eval()
    cebm_model, cebm_diff, _ = load_contact_ebm(args.cebm_repo, args.cebm_ckpt, dev)
    topo_keys = model._topology_keys()

    rows_by_pid = {r["pid"]: r for r in filt["rows"]}
    selected_pids = filt["selected_pids"]
    if args.max_targets > 0:
        selected_pids = selected_pids[: args.max_targets]

    arms_to_run = [filt["selected_arm"]] if args.arm_override == "auto" else (
        ["tmpl", "self"] if args.arm_override == "both" else [args.arm_override]
    )
    all_pids_sorted = sorted(rows_by_pid.keys())
    ci_of = {p: i for i, p in enumerate(all_pids_sorted)}

    for arm in arms_to_run:
        print(f"\n==================== RUNNING ARM: {arm} (n={len(selected_pids)} targets) ====================", flush=True)
        for pid in selected_pids:
            rinfo = rows_by_pid[pid]
            ref_stem = rinfo["ref_stem"] if arm == "tmpl" else pid
            gq = torch.load(os.path.join(proc_dir, f"{pid}.pt"), map_location="cpu", weights_only=False)
            gt = torch.load(os.path.join(proc_dir, f"{ref_stem}.pt"), map_location="cpu", weights_only=False)
            full_q, amask_q, aatype_q = graph_to_atom37(gq)
            full_t0, amask_t0, aatype_t0 = graph_to_atom37(gt)
            Lq, Lt0 = int(amask_q.shape[0]), int(amask_t0.shape[0])
            cm_nat = builder.confind_native_contacts(gq, Lq).numpy()
            cm_tmpl0 = builder.confind_native_contacts(gt, Lt0).numpy()
            native_cif, native_ch = native_for(args.natives_dir, pid)
            c2c_seed = C2C_SEED_BASE + ci_of[pid]
            n_pad = Lq if args.pad_to == 0 else args.pad_to

            pair_dir = os.path.join(args.out, arm, pid)
            os.makedirs(pair_dir, exist_ok=True)
            tmpl0_pdb = os.path.join(pair_dir, "template_r0.pdb")
            builder.write_ca_pdb(full_t0[:, builder.CA_IDX].numpy(), amask_t0[:, builder.CA_IDX].numpy(), tmpl0_pdb)
            nat_ca_pdb = os.path.join(pair_dir, "native_ca.pdb")
            builder.write_ca_pdb(full_q[:, builder.CA_IDX].numpy(), amask_q[:, builder.CA_IDX].numpy(), nat_ca_pdb)
            tm_tmpl0_vs_nat = tm_pair_seqindep(args.usalign, nat_ca_pdb, tmpl0_pdb)

            # ── Round 0 & Task 1.0 (Trajectory AA-pair compatibility) ──
            ref0, elem_t0, n_el0, trunc0, dssp_t0, _ = template_reference_with_dssp(
                full_t0,
                amask_t0,
                aatype_t0,
                predictor,
                transform,
                Lq,
                pro_donor,
                dssp_def="h_bond",
                cm_t_override=cm_tmpl0 if predictor is None else None,
            )
            topology0 = {kk: Proteina._stack_topology([ref0[kk]] * args.k).to(dev) for kk in topo_keys}
            mask_q = torch.zeros(args.k, n_pad, dtype=torch.bool, device=dev)
            mask_q[:, :Lq] = True
            res_q = torch.zeros(args.k, n_pad, dtype=torch.long, device=dev)
            res_q[:, :Lq] = aatype_q.to(dev)

            L.seed_everything(TRI_SEED)
            res0, rec0, step_records0 = generate_tri_with_trajectory(model, cfg_exp, topology0, res_q, mask_q, args.k, Lq)
            prob0 = torch.sigmoid(res0["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
            e_clean0, e_noised0 = score_maps_with_cebm(cebm_model, cebm_diff, prob0, aatype_q, Lq, args.t_eval, dev, seed=TRI_SEED)

            if args.mode in ("traj_aa", "all"):
                nat_aa_compat = score_map_aa_compatibility(cm_nat, aatype_q, aa_stats)
                tmpl_aa_compat = score_map_aa_compatibility(cm_tmpl0, aatype_t0, aa_stats)
                L_min = min(Lq, Lt0)
                tmpl_on_query_compat = score_map_aa_compatibility(cm_tmpl0[:L_min, :L_min], aatype_q[:L_min], aa_stats)
                traj_rows = []
                for step_idx, (t_val, p_step) in enumerate(step_records0):
                    mean_p = p_step.mean(axis=0)
                    sc_mean = score_map_aa_compatibility(mean_p, aatype_q, aa_stats)
                    sc_per_k = [score_map_aa_compatibility(p_step[ii], aatype_q, aa_stats) for ii in range(args.k)]
                    cm_vs_nat = [contact_metrics_vs_gt(p_step[ii], cm_nat) for ii in range(args.k)]
                    traj_rows.append({
                        "step_idx": step_idx,
                        "t": round(t_val, 4),
                        "mean_map_soft_log_odds": sc_mean["soft_log_odds"],
                        "mean_map_jsd_vs_nat": sc_mean["jsd_vs_nat"],
                        "sample_soft_log_odds_mean": float(np.mean([x["soft_log_odds"] for x in sc_per_k])),
                        "sample_hard_log_odds_mean": float(np.mean([x["hard_log_odds"] for x in sc_per_k])),
                        "sample_jsd_vs_nat_mean": float(np.mean([x["jsd_vs_nat"] for x in sc_per_k])),
                        "p_at_l_vs_nat_mean": float(np.mean([x["p_at_l"] for x in cm_vs_nat])),
                        "f1_vs_nat_mean": float(np.mean([x["f1"] for x in cm_vs_nat])),
                    })
                traj_summary = {
                    "pid": pid,
                    "arm": arm,
                    "ref_stem": ref_stem,
                    "Lq": Lq,
                    "Lt0": Lt0,
                    "tm_tmpl0_vs_nat": tm_tmpl0_vs_nat,
                    "native_aa_compat": nat_aa_compat,
                    "template_own_aa_compat": tmpl_aa_compat,
                    "template_on_query_aa_compat": tmpl_on_query_compat,
                    "t_first": traj_rows[0] if traj_rows else None,
                    "t_mid": traj_rows[len(traj_rows) // 2] if traj_rows else None,
                    "t_last": traj_rows[-1] if traj_rows else None,
                    "trajectory": traj_rows,
                }
                with open(os.path.join(pair_dir, "traj_aa.json"), "w") as fh:
                    json.dump(traj_summary, fh, indent=2)
                print(
                    f"[1.0 traj_aa] {pid} ({arm}): nat_log_odds={nat_aa_compat['soft_log_odds']:.4f} "
                    f"tmpl_on_q={tmpl_on_query_compat['soft_log_odds']:.4f} | "
                    f"traj t={traj_rows[0]['t']:.2f}->{traj_rows[-1]['t']:.2f}: "
                    f"soft_log_odds {traj_rows[0]['sample_soft_log_odds_mean']:.4f}->{traj_rows[-1]['sample_soft_log_odds_mean']:.4f}, "
                    f"JSD {traj_rows[0]['sample_jsd_vs_nat_mean']:.4f}->{traj_rows[-1]['sample_jsd_vs_nat_mean']:.4f}, "
                    f"P@L {traj_rows[0]['p_at_l_vs_nat_mean']:.3f}->{traj_rows[-1]['p_at_l_vs_nat_mean']:.3f}",
                    flush=True,
                )

            # Fold all k=8 Round-0 maps with c2c to get full Round-0 distribution + ContactEBM correlation
            r0_pdbs = []
            r0_coords14 = []
            r0_tms = []
            r0_tms_vs_tmpl0 = []
            for si in range(args.k):
                c14 = c2c_rollout_atom14(c2c, prob0[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                pdb_si = os.path.join(pair_dir, f"r0_s{si}.pdb")
                write_atom14_pdb(pdb_si, c14, aatype_q, torch.ones(Lq))
                tm_nat = tm_vs_native(args.usalign, native_cif, native_ch, pdb_si)
                tm_t0 = tm_pair_seqindep(args.usalign, pdb_si, tmpl0_pdb)
                r0_pdbs.append(pdb_si)
                r0_coords14.append(c14)
                r0_tms.append(tm_nat)
                r0_tms_vs_tmpl0.append(tm_t0)

            best_idx0 = int(np.argmin(e_clean0))
            print(
                f"[r0 summary] {pid} ({arm}): mean8_TM={np.mean(r0_tms):.4f} best8_TM={np.max(r0_tms):.4f} "
                f"cebm_pick(idx={best_idx0}, E={e_clean0[best_idx0]:.3f})_TM={r0_tms[best_idx0]:.4f} "
                f"vs_tmpl0_TM={r0_tms_vs_tmpl0[best_idx0]:.4f}",
                flush=True,
            )

            # ── Task 1a: 5-Round Cycling (both h_bond and tmalign_smooth DSSP) ──
            if args.mode in ("cycling", "all"):
                for cyc_dssp in ("h_bond", "tmalign_smooth"):
                    cyc_rounds = [{
                        "round": 0,
                        "dssp_def": cyc_dssp,
                        "n_el": int(n_el0),
                        "pick_idx": best_idx0,
                        "pick_energy_clean_005": float(e_clean0[best_idx0]),
                        "pick_energy_noised_005": float(e_noised0[best_idx0]),
                        "pick_tm_vs_native": float(r0_tms[best_idx0]),
                        "pick_tm_vs_tmpl0": float(r0_tms_vs_tmpl0[best_idx0]),
                        "pick_tm_vs_prev": float(r0_tms_vs_tmpl0[best_idx0]),
                        "mean8_tm_vs_native": float(np.mean(r0_tms)),
                        "best8_tm_vs_native": float(np.max(r0_tms)),
                        "all8_tms": [float(x) for x in r0_tms],
                        "all8_energies_clean_005": [float(x) for x in e_clean0],
                        "pick_p_at_l_vs_nat": contact_metrics_vs_gt(prob0[best_idx0], cm_nat)["p_at_l"],
                        "pick_aa_soft_log_odds": score_map_aa_compatibility(prob0[best_idx0], aatype_q, aa_stats)["soft_log_odds"],
                    }]
                    cur_c14 = r0_coords14[best_idx0]
                    prev_pdb = r0_pdbs[best_idx0]
                    for r_idx in range(1, args.n_cycles):
                        full_tr, amask_tr = coords14_to_atom37(cur_c14, aatype_q)
                        ref_r, elem_tr, n_el_r, trunc_r, _, _ = template_reference_with_dssp(
                            full_tr, amask_tr, aatype_q, predictor, transform, Lq, pro_donor, dssp_def=cyc_dssp
                        )
                        topo_r = {kk: Proteina._stack_topology([ref_r[kk]] * args.k).to(dev) for kk in topo_keys}
                        L.seed_everything(TRI_SEED + r_idx)
                        res_r, _, _ = generate_tri_with_trajectory(model, cfg_exp, topo_r, res_q, mask_q, args.k, Lq)
                        prob_r = torch.sigmoid(res_r["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
                        e_cl_r, e_ns_r = score_maps_with_cebm(
                            cebm_model, cebm_diff, prob_r, aatype_q, Lq, args.t_eval, dev, seed=TRI_SEED + r_idx
                        )
                        tms_r = []
                        c14s_r = []
                        pdbs_r = []
                        for si in range(args.k):
                            c14_si = c2c_rollout_atom14(c2c, prob_r[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                            pdb_si = os.path.join(pair_dir, f"cyc_{cyc_dssp}_r{r_idx}_s{si}.pdb")
                            write_atom14_pdb(pdb_si, c14_si, aatype_q, torch.ones(Lq))
                            tms_r.append(tm_vs_native(args.usalign, native_cif, native_ch, pdb_si))
                            c14s_r.append(c14_si)
                            pdbs_r.append(pdb_si)
                        pick_r = int(np.argmin(e_cl_r))
                        pick_pdb = pdbs_r[pick_r]
                        tm_vs_t0 = tm_pair_seqindep(args.usalign, pick_pdb, tmpl0_pdb)
                        tm_vs_prev = tm_pair_seqdep(args.usalign, pick_pdb, prev_pdb)
                        cyc_rounds.append({
                            "round": r_idx,
                            "dssp_def": cyc_dssp,
                            "n_el": int(n_el_r),
                            "pick_idx": pick_r,
                            "pick_energy_clean_005": float(e_cl_r[pick_r]),
                            "pick_energy_noised_005": float(e_ns_r[pick_r]),
                            "pick_tm_vs_native": float(tms_r[pick_r]),
                            "pick_tm_vs_tmpl0": float(tm_vs_t0),
                            "pick_tm_vs_prev": float(tm_vs_prev),
                            "mean8_tm_vs_native": float(np.mean(tms_r)),
                            "best8_tm_vs_native": float(np.max(tms_r)),
                            "all8_tms": [float(x) for x in tms_r],
                            "all8_energies_clean_005": [float(x) for x in e_cl_r],
                            "pick_p_at_l_vs_nat": contact_metrics_vs_gt(prob_r[pick_r], cm_nat)["p_at_l"],
                            "pick_aa_soft_log_odds": score_map_aa_compatibility(prob_r[pick_r], aatype_q, aa_stats)["soft_log_odds"],
                        })
                        cur_c14 = c14s_r[pick_r]
                        prev_pdb = pick_pdb
                        print(
                            f"[1a cycling {cyc_dssp}] {pid} ({arm}) r={r_idx}: n_el={n_el_r} "
                            f"pick_TM={tms_r[pick_r]:.4f} (mean8={np.mean(tms_r):.4f}, best8={np.max(tms_r):.4f}) "
                            f"TM_vs_tmpl0={tm_vs_t0:.4f} TM_vs_prev={tm_vs_prev:.4f} E={e_cl_r[pick_r]:.3f}",
                            flush=True,
                        )
                    with open(os.path.join(pair_dir, f"cycling_{cyc_dssp}.json"), "w") as fh:
                        json.dump({"pid": pid, "arm": arm, "ref_stem": ref_stem, "dssp_def": cyc_dssp, "rounds": cyc_rounds}, fh, indent=2)

            # ── Task 1b: ContactEBM Reverse / Langevin Sampling (t0 in {0.3, 0.4, 0.5}, K in {0, 20}) ──
            if args.mode in ("cebm_langevin", "all"):
                init_bmap = (prob0[best_idx0] >= 0.5).astype(np.uint8)
                langevin_rows = []
                for t0 in (0.3, 0.4, 0.5):
                    for K_corr in (0, 20):
                        t_l0 = time.time()
                        res_lang = cebm_reverse_sample_from_t0(
                            cebm_model, cebm_diff, init_bmap, aatype_q, Lq, t0=t0, K_corrector=K_corr, dev=dev, seed=TRI_SEED
                        )
                        # Save every step's bit-packed uint8 clean contact map prediction
                        npz_path = os.path.join(pair_dir, f"cebm_t{int(t0*100):03d}_K{K_corr}.npz")
                        np.savez_compressed(
                            npz_path,
                            pid=pid,
                            arm=arm,
                            L=np.int32(Lq),
                            t0=np.float32(t0),
                            K=np.int32(K_corr),
                            step_times=res_lang["step_times"],
                            step_kinds=res_lang["step_kinds"],
                            step_energies_t=res_lang["step_energies_t"],
                            step_energies_005=res_lang["step_energies_005"],
                            packed_clean_steps=res_lang["packed_clean_steps"],
                            final_binary=np.packbits(res_lang["final_binary"]),
                        )
                        # Unpack intermediate predictor steps (start, midpoint, final) and fold with c2c
                        unpacked_all = unpack_upper_triangle(res_lang["packed_clean_steps"], Lq)
                        pred_indices = [idx for idx, kd in enumerate(res_lang["step_kinds"]) if kd == "predictor"]
                        mid_pred_idx = pred_indices[len(pred_indices) // 2]
                        last_pred_idx = pred_indices[-1]

                        c14_mid = c2c_rollout_atom14(c2c, unpacked_all[mid_pred_idx], aatype_q, Lq, n_pad, c2c_seed, dev)
                        pdb_mid = os.path.join(pair_dir, f"cebm_t{int(t0*100):03d}_K{K_corr}_mid.pdb")
                        write_atom14_pdb(pdb_mid, c14_mid, aatype_q, torch.ones(Lq))
                        tm_mid = tm_vs_native(args.usalign, native_cif, native_ch, pdb_mid)

                        c14_fin = c2c_rollout_atom14(c2c, res_lang["final_binary"].astype(np.float32), aatype_q, Lq, n_pad, c2c_seed, dev)
                        pdb_fin = os.path.join(pair_dir, f"cebm_t{int(t0*100):03d}_K{K_corr}_final.pdb")
                        write_atom14_pdb(pdb_fin, c14_fin, aatype_q, torch.ones(Lq))
                        tm_fin = tm_vs_native(args.usalign, native_cif, native_ch, pdb_fin)
                        tm_fin_vs_t0 = tm_pair_seqindep(args.usalign, pdb_fin, tmpl0_pdb)

                        cm_fin_vs_nat = contact_metrics_vs_gt(res_lang["final_binary"], cm_nat)
                        aa_fin = score_map_aa_compatibility(res_lang["final_binary"], aatype_q, aa_stats)
                        sec_l = time.time() - t_l0
                        row_l = {
                            "pid": pid,
                            "arm": arm,
                            "t0": t0,
                            "K": K_corr,
                            "n_calls": res_lang["n_calls"],
                            "init_tm": float(r0_tms[best_idx0]),
                            "init_e005": float(e_clean0[best_idx0]),
                            "mid_t": float(res_lang["step_times"][mid_pred_idx]),
                            "mid_e005": float(res_lang["step_energies_005"][mid_pred_idx]),
                            "mid_tm": float(tm_mid),
                            "final_e005": float(res_lang["final_e005"]),
                            "final_tm": float(tm_fin),
                            "final_tm_vs_tmpl0": float(tm_fin_vs_t0),
                            "delta_tm_vs_init": float(tm_fin - r0_tms[best_idx0]),
                            "final_f1_vs_nat": cm_fin_vs_nat["f1"],
                            "final_prec_vs_nat": cm_fin_vs_nat["prec"],
                            "final_rec_vs_nat": cm_fin_vs_nat["rec"],
                            "final_aa_hard_log_odds": aa_fin["hard_log_odds"],
                            "seconds": round(sec_l, 2),
                        }
                        langevin_rows.append(row_l)
                        print(
                            f"[1b cebm] {pid} ({arm}) t0={t0:.1f} K={K_corr:2d}: calls={res_lang['n_calls']:4d} "
                            f"E005 {e_clean0[best_idx0]:.2f}->{res_lang['final_e005']:.2f} | "
                            f"TM init={r0_tms[best_idx0]:.4f} mid={tm_mid:.4f} final={tm_fin:.4f} (dTM={tm_fin - r0_tms[best_idx0]:+.4f}) "
                            f"F1={cm_fin_vs_nat['f1']:.3f} ({sec_l:.1f}s)",
                            flush=True,
                        )
                with open(os.path.join(pair_dir, "cebm_langevin.json"), "w") as fh:
                    json.dump({"pid": pid, "arm": arm, "ref_stem": ref_stem, "runs": langevin_rows}, fh, indent=2)

            # ── Task 2: Compare dssp_def in {h_bond, tmalign, tmalign_smooth} ──
            if args.mode in ("sse_tmalign", "all"):
                sse_rows = []
                for ddef in ("h_bond", "tmalign", "tmalign_smooth"):
                    ref_d, elem_d, n_el_d, trunc_d, dssp_vec, _ = template_reference_with_dssp(
                        full_t0,
                        amask_t0,
                        aatype_t0,
                        predictor,
                        transform,
                        Lq,
                        pro_donor,
                        dssp_def=ddef,
                        cm_t_override=cm_tmpl0 if predictor is None else None,
                    )
                    n_h = int((dssp_vec == DSSP_HELIX).sum().item())
                    n_e = int((dssp_vec == DSSP_STRAND).sum().item())
                    n_l = int((dssp_vec == 0).sum().item())
                    if ddef == "h_bond":
                        prob_d = prob0
                        tms_d = r0_tms
                        e_cl_d = e_clean0
                    else:
                        topo_d = {kk: Proteina._stack_topology([ref_d[kk]] * args.k).to(dev) for kk in topo_keys}
                        L.seed_everything(TRI_SEED)
                        res_d, _, _ = generate_tri_with_trajectory(model, cfg_exp, topo_d, res_q, mask_q, args.k, Lq)
                        prob_d = torch.sigmoid(res_d["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
                        e_cl_d, _ = score_maps_with_cebm(
                            cebm_model, cebm_diff, prob_d, aatype_q, Lq, args.t_eval, dev, seed=TRI_SEED
                        )
                        tms_d = []
                        for si in range(args.k):
                            c14_si = c2c_rollout_atom14(c2c, prob_d[si], aatype_q, Lq, n_pad, c2c_seed, dev)
                            pdb_si = os.path.join(pair_dir, f"sse_{ddef}_s{si}.pdb")
                            write_atom14_pdb(pdb_si, c14_si, aatype_q, torch.ones(Lq))
                            tms_d.append(tm_vs_native(args.usalign, native_cif, native_ch, pdb_si))
                    pick_d = int(np.argmin(e_cl_d))
                    sse_rows.append({
                        "pid": pid,
                        "arm": arm,
                        "dssp_def": ddef,
                        "n_el": int(n_el_d),
                        "n_helix_res": n_h,
                        "n_strand_res": n_e,
                        "n_loop_res": n_l,
                        "he_frac": float((n_h + n_e) / max(Lt0, 1)),
                        "mean8_tm": float(np.mean(tms_d)),
                        "median8_tm": float(np.median(tms_d)),
                        "best8_tm": float(np.max(tms_d)),
                        "cebm_pick_tm": float(tms_d[pick_d]),
                        "all8_tms": [float(x) for x in tms_d],
                    })
                    print(
                        f"[2 sse_tmalign] {pid} ({arm}) {ddef:<14}: n_el={n_el_d:2d} H={n_h:3d} E={n_e:3d} "
                        f"(he_frac={(n_h+n_e)/max(Lt0,1):.2f}) | mean8_TM={np.mean(tms_d):.4f} "
                        f"best8_TM={np.max(tms_d):.4f} cebm_pick_TM={tms_d[pick_d]:.4f}",
                        flush=True,
                    )
                with open(os.path.join(pair_dir, "sse_tmalign.json"), "w") as fh:
                    json.dump({"pid": pid, "arm": arm, "ref_stem": ref_stem, "variants": sse_rows}, fh, indent=2)

    print_final_report(args.out, selected_pids, arms_to_run, args.mode)
    print("\n=== ALL REQUESTED MODES COMPLETED ===", flush=True)
    return 0


def _pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(xs, ys):
    if len(xs) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(xs, dtype=np.float64))).astype(np.float64)
    ry = np.argsort(np.argsort(np.asarray(ys, dtype=np.float64))).astype(np.float64)
    return _pearson(rx, ry)


def print_final_report(out_dir, selected_pids, arms, mode):
    for arm in arms:
        print(f"\n==================== FINAL AGGREGATE REPORT (arm={arm}, n={len(selected_pids)}) ====================", flush=True)
        if mode in ("traj_aa", "all"):
            print("\n--- Task 1.0: Pairwise AA Contact Compatibility Across TopoRef Trajectory ---")
            print(f"{'pid':<10} | {'nat_LO':>7} {'tmplQ_LO':>8} | {'t_start_LO':>10} {'t_mid_LO':>8} {'t_end_LO':>8} | {'JSD_start':>9} {'JSD_end':>7} | {'P@L_end':>7}")
            for pid in selected_pids:
                p = os.path.join(out_dir, arm, pid, "traj_aa.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                tf, tm, tl = d["t_first"], d["t_mid"], d["t_last"]
                print(
                    f"{pid:<10} | {d['native_aa_compat']['soft_log_odds']:>7.4f} "
                    f"{d['template_on_query_aa_compat']['soft_log_odds']:>8.4f} | "
                    f"{tf['sample_soft_log_odds_mean']:>10.4f} {tm['sample_soft_log_odds_mean']:>8.4f} {tl['sample_soft_log_odds_mean']:>8.4f} | "
                    f"{tf['sample_jsd_vs_nat_mean']:>9.4f} {tl['sample_jsd_vs_nat_mean']:>7.4f} | {tl['p_at_l_vs_nat_mean']:>7.3f}"
                )

        if mode in ("cycling", "all"):
            for cyc_dssp in ("h_bond", "tmalign_smooth"):
                print(f"\n--- Task 1a: 5-Round Cycling (dssp_def={cyc_dssp}) ---")
                print(f"{'pid':<10} | {'r0_TM':>6} {'r1_TM':>6} {'r2_TM':>6} {'r3_TM':>6} {'r4_TM':>6} | {'r4-r0':>6} | {'r4_vs_t0':>8} {'r4_vs_r3':>8}")
                all_e, all_tm = [], []
                for pid in selected_pids:
                    p = os.path.join(out_dir, arm, pid, f"cycling_{cyc_dssp}.json")
                    if not os.path.exists(p):
                        continue
                    d = json.load(open(p))
                    rds = d["rounds"]
                    tms = [r["pick_tm_vs_native"] for r in rds]
                    for r in rds:
                        all_e.extend(r["all8_energies_clean_005"])
                        all_tm.extend(r["all8_tms"])
                    t_str = " ".join(f"{x:>6.3f}" for x in tms[:5])
                    print(
                        f"{pid:<10} | {t_str} | {tms[-1] - tms[0]:>+6.3f} | "
                        f"{rds[-1]['pick_tm_vs_tmpl0']:>8.3f} {rds[-1]['pick_tm_vs_prev']:>8.3f}"
                    )
                if all_e:
                    print(f"  [ContactEBM E(t=0.05) vs TM across all cycling maps (n={len(all_e)})]: "
                          f"Pearson r={_pearson(all_e, all_tm):+.4f}, Spearman rho={_spearman(all_e, all_tm):+.4f}")

        if mode in ("cebm_langevin", "all"):
            print("\n--- Task 1b: ContactEBM Reverse / Langevin Sampling (t0 in {0.3, 0.4, 0.5}, K in {0, 20}) ---")
            print(f"{'pid':<10} {'t0':>4} {'K':>3} | {'init_TM':>7} {'mid_TM':>7} {'fin_TM':>7} {'dTM':>7} | {'init_E':>7} {'fin_E':>7} | {'fin_F1':>6}")
            lang_e, lang_tm = [], []
            for pid in selected_pids:
                p = os.path.join(out_dir, arm, pid, "cebm_langevin.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                for r in d["runs"]:
                    lang_e.extend([r["init_e005"], r["mid_e005"], r["final_e005"]])
                    lang_tm.extend([r["init_tm"], r["mid_tm"], r["final_tm"]])
                    print(
                        f"{pid:<10} {r['t0']:>4.1f} {r['K']:>3d} | "
                        f"{r['init_tm']:>7.4f} {r['mid_tm']:>7.4f} {r['final_tm']:>7.4f} {r['delta_tm_vs_init']:>+7.4f} | "
                        f"{r['init_e005']:>7.2f} {r['final_e005']:>7.2f} | {r['final_f1_vs_nat']:>6.3f}"
                    )
            if lang_e:
                print(f"  [ContactEBM E(t=0.05) vs TM across Langevin trajectory checkpoints (n={len(lang_e)})]: "
                      f"Pearson r={_pearson(lang_e, lang_tm):+.4f}, Spearman rho={_spearman(lang_e, lang_tm):+.4f}")

        if mode in ("sse_tmalign", "all"):
            print("\n--- Task 2: TMalign/USalign CA-Based SSE Classifier vs PyDSSP ---")
            print(f"{'pid':<10} {'dssp_def':<15} | {'n_el':>4} {'H_res':>5} {'E_res':>5} {'he_frac':>7} | {'mean8_TM':>8} {'best8_TM':>8} {'cebm_TM':>8}")
            for pid in selected_pids:
                p = os.path.join(out_dir, arm, pid, "sse_tmalign.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                for v in d["variants"]:
                    print(
                        f"{pid:<10} {v['dssp_def']:<15} | {v['n_el']:>4d} {v['n_helix_res']:>5d} {v['n_strand_res']:>5d} {v['he_frac']:>7.3f} | "
                        f"{v['mean8_tm']:>8.4f} {v['best8_tm']:>8.4f} {v['cebm_pick_tm']:>8.4f}"
                    )


if __name__ == "__main__":
    sys.exit(main())
