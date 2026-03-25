#!/usr/bin/env python3
"""
ProteinEBM Scorer (single-pass energy at t=0.1)

This script scores Proteina-generated decoy PDBs with ProteinEBM and writes results to:
  <inference_output_dir>/proteinebm_analysis/

It is designed to be used as a subprocess from the existing evaluation pipeline.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from ml_collections import ConfigDict

from protein_ebm.data.protein_utils import residues_to_features
from protein_ebm.model.boltz_utils import center_random_augmentation
from protein_ebm.model.ebm import ProteinEBM
from protein_ebm.model.r3_diffuser import R3Diffuser

from proteinfoundation.af2rank_evaluation.usalign_tabular import parse_usalign_pair_outfmt2


_USALIGN_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _resolve_tm_workers(explicit: Optional[int]) -> int:
    """Clamp CPU worker count for parallel USalign (1-64); mirrors proteina_diversity.resolve_num_workers."""
    if explicit is not None:
        return max(1, min(64, int(explicit)))
    n = os.cpu_count() or 1
    return max(1, min(64, n))


# ---------------------------------------------------------------------------
# Diagnostic helpers (mirrored from test_proteinebm_single.py)
# ---------------------------------------------------------------------------

def _tensor_stats(name: str, t: torch.Tensor) -> str:
    """Return a one-line stats string for a tensor."""
    arr = t.detach().float().cpu().numpy().ravel()
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if len(finite):
        return (f"shape={list(t.shape)} dtype={t.dtype} "
                f"min={finite.min():.4g} max={finite.max():.4g} mean={finite.mean():.4g} "
                f"nan={n_nan} inf={n_inf}")
    return f"shape={list(t.shape)} dtype={t.dtype} ALL NaN/Inf  nan={n_nan} inf={n_inf}"


def _check_model_weights(model: torch.nn.Module) -> None:
    """Print a one-line model weight NaN/Inf summary; list any bad parameters."""
    n_params = n_nan = n_inf = 0
    for pname, p in model.named_parameters():
        n_params += 1
        arr = p.detach().float().cpu().numpy().ravel()
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        if has_nan or has_inf:
            n_nan += has_nan
            n_inf += has_inf
            print(f"  PROBLEM param: {pname}  nan={has_nan} inf={has_inf}  shape={list(p.shape)}", flush=True)
    print(f"Model weight check: {n_params} params,  nan_params={n_nan}  inf_params={n_inf}", flush=True)


def load_proteinebm_model(config_path: str, checkpoint_path: str, device: torch.device) -> Tuple[ProteinEBM, ConfigDict]:
    config_path = os.path.abspath(os.path.expanduser(config_path))
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config)

    diffuser = R3Diffuser(config.diffuser)
    model = ProteinEBM(config.model, diffuser).to(device)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model")}
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def _pdb_to_chain_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("decoy", pdb_path)
    chain = list(structure.get_chains())[0]
    return [r for r in chain.get_residues() if is_aa(r)]


def _extract_ca_coords(structure_path: str, chain_id: str | None) -> np.ndarray:
    """
    Extract CA coordinates from a PDB or CIF file for a specific chain.

    - If chain_id is None and there is exactly 1 chain, use it.
    - If chain_id is provided but not present:
        - if there is exactly 1 chain, use it
        - else raise with available chain IDs
    """
    is_cif = structure_path.lower().endswith(".cif")
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure("ref", structure_path)
    model = list(structure.get_models())[0]
    chains = list(model.get_chains())
    chain_ids = [c.id for c in chains]

    # Heuristic chain mappings (copied from AF2Rank scorer logic) for cases where
    # the dataset's "A/B/C" chain IDs don't match author-provided CIF chain IDs.
    chain_mappings: dict[str, list[str]] = {
        "A": ["E", "N", "X", "1", "a"],
        "B": ["F", "O", "Y", "2", "b"],
        "C": ["G", "P", "Z", "3", "c"],
    }

    if chain_id is None:
        if len(chains) != 1:
            raise ValueError(f"chain_id not provided and structure has multiple chains: {chain_ids}")
        chain = chains[0]
    else:
        if chain_id in chain_ids:
            chain = next(c for c in chains if c.id == chain_id)
        else:
            # Try common chain remappings (A->E, etc.)
            mapped = None
            for alt in chain_mappings.get(chain_id, []):
                if alt in chain_ids:
                    mapped = alt
                    break
            if mapped is not None:
                chain = next(c for c in chains if c.id == mapped)
            elif len(chains) == 1:
                chain = chains[0]
            else:
                raise ValueError(f"Chain '{chain_id}' not found in {structure_path}. Available chains: {chain_ids}")

    coords: list[np.ndarray] = []
    for res in chain.get_residues():
        if not is_aa(res):
            continue
        if "CA" not in res:
            continue
        coords.append(res["CA"].get_coord())

    if not coords:
        raise ValueError(f"No CA atoms found in {structure_path} (chain={chain.id})")

    return np.asarray(coords, dtype=np.float32)


def _parse_usalign_output(lines: list[str]) -> Dict[str, float]:
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    out = {"rms": 0.0, "tms": 0.0, "gdt": 0.0}
    for line in lines:
        line = line.rstrip()
        if line.startswith("RMSD"):
            out["rms"] = parse_float(line)
        elif line.startswith("TM-score"):
            out["tms"] = parse_float(line)
        elif line.startswith("GDT-TS-score"):
            out["gdt"] = parse_float(line)
    return out


def tmscore_ca_coords(
    x: np.ndarray,
    y: np.ndarray,
    tmscore_exe: str = "USalign",
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Compute TM-score / RMSD / GDT-TS for two CA-traces using USalign (preferred) or TMscore.
    Requires the executable to be available in PATH.

    If env is set, it is merged into the subprocess environment (e.g. OMP_NUM_THREADS=1
    when many USalign processes run in parallel).
    """
    exe = shutil.which(tmscore_exe) or shutil.which("USalign") or shutil.which("TMscore")
    if exe is None:
        raise FileNotFoundError("Neither USalign nor TMscore found in PATH")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f2:
        for f, z in zip([f1, f2], [x, y]):
            for k, c in enumerate(z):
                f.write(
                    "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                    % (k + 1, "CA", "ALA", "A", k + 1, float(c[0]), float(c[1]), float(c[2]), 1, 0)
                )
            f.flush()

    cmd = [exe, f1.name, f2.name]
    if os.path.basename(exe) == "USalign":
        cmd += ["-TMscore", "5", "-outfmt", "2"]

    subprocess_env = None
    if env is not None:
        subprocess_env = os.environ.copy()
        subprocess_env.update(env)

    output = subprocess.check_output(cmd, text=True, env=subprocess_env)
    os.unlink(f1.name)
    os.unlink(f2.name)
    if os.path.basename(exe) == "USalign":
        return parse_usalign_pair_outfmt2(output)
    return _parse_usalign_output(output.splitlines())


def tmscore_pdb_paths(
    path_a: str,
    path_b: str,
    tmscore_exe: str = "USalign",
    env: Optional[Dict[str, str]] = None,
    chain1: Optional[str] = None,
    chain2: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run USalign (or TMscore) on two existing structure files (PDB/mmCIF per USalign auto-detect).
    Same -TMscore 5 mode as tmscore_ca_coords when the executable is USalign.
    Uses -outfmt 2 for USalign; GDT-TS is not in tabular output (set to NaN).

    Optional chain1/chain2: passed as USalign -chain1 / -chain2 when set.
    """
    exe = shutil.which(tmscore_exe) or shutil.which("USalign") or shutil.which("TMscore")
    if exe is None:
        raise FileNotFoundError("Neither USalign nor TMscore found in PATH")
    cmd = [exe, path_a, path_b]
    if os.path.basename(exe) == "USalign":
        if chain1 is not None:
            cmd += ["-chain1", chain1]
        if chain2 is not None:
            cmd += ["-chain2", chain2]
        cmd += ["-TMscore", "5", "-outfmt", "2"]
    subprocess_env = None
    if env is not None:
        subprocess_env = os.environ.copy()
        subprocess_env.update(env)
    output = subprocess.check_output(cmd, text=True, env=subprocess_env)
    if os.path.basename(exe) == "USalign":
        return parse_usalign_pair_outfmt2(output)
    return _parse_usalign_output(output.splitlines())


def _spearmanr(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman correlation without SciPy.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have same shape")
    if x_arr.size < 2:
        return float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)

        # average ranks for ties
        sorted_a = a[order]
        i = 0
        while i < len(sorted_a):
            j = i + 1
            while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    rx = rankdata(x_arr)
    ry = rankdata(y_arr)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def plot_proteinebm_results(results: List[Dict[str, object]], output_dir: str, protein_id: str) -> Dict[str, str]:
    """
    Write per-protein diagnostic plots (analogous to AF2Rank folder PNGs).

    Produces:
      - proteinebm_<protein_id>_score_vs_true_quality.png  (score=-energy vs tm_ref_template)
      - proteinebm_<protein_id>_energy_hist.png
      - proteinebm_<protein_id>_tm_hist.png
    """
    energies = np.asarray([float(r["energy"]) for r in results], dtype=np.float64)
    tm = np.asarray([float(r["tm_ref_template"]) for r in results], dtype=np.float64)
    score = -energies

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: score vs true quality
    fig = plt.figure(figsize=(8, 6), dpi=120)
    plt.scatter(tm, score, s=18, alpha=0.5)
    rho = _spearmanr(tm.tolist(), score.tolist()) if len(tm) > 1 else float("nan")
    plt.title(f"ProteinEBM: score(-energy) vs TM(ref, decoy)\n{protein_id} | Spearman ρ={rho:.3f}")
    plt.xlabel("TM-score (Reference vs Decoy) [tm_ref_template]")
    plt.ylabel("ProteinEBM score (-energy, higher is better)")
    plt.grid(True, alpha=0.3)
    score_plot = os.path.join(output_dir, f"proteinebm_{protein_id}_score_vs_true_quality.png")
    plt.tight_layout()
    plt.savefig(score_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: energy histogram
    valid_energies = energies[~np.isnan(energies)]
    fig = plt.figure(figsize=(8, 6), dpi=120)
    if len(valid_energies) > 0:
        plt.hist(valid_energies, bins=30, alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid energies (all NaN)", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"ProteinEBM energy histogram\n{protein_id}")
    plt.xlabel("Energy (lower is better)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    energy_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_energy_hist.png")
    plt.tight_layout()
    plt.savefig(energy_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: TM histogram
    valid_tm = tm[~np.isnan(tm)]
    fig = plt.figure(figsize=(8, 6), dpi=120)
    if len(valid_tm) > 0:
        plt.hist(valid_tm, bins=30, alpha=0.8)
    else:
        plt.text(0.5, 0.5, "No valid TM scores (all NaN)", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title(f"TM(ref, decoy) histogram\n{protein_id}")
    plt.xlabel("TM-score (Reference vs Decoy)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    tm_hist = os.path.join(output_dir, f"proteinebm_{protein_id}_tm_hist.png")
    plt.tight_layout()
    plt.savefig(tm_hist, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "score_vs_true_quality": os.path.abspath(score_plot),
        "energy_hist": os.path.abspath(energy_hist),
        "tm_hist": os.path.abspath(tm_hist),
    }


def build_input_feats_from_pdb(
    pdb_path: str,
    model: ProteinEBM,
    t: float,
    template_self_condition: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    residues = _pdb_to_chain_residues(pdb_path)
    atom_positions, atom_mask, aatype, residue_idx = residues_to_features(residues, strict=True)

    atom_positions = atom_positions.to(device)
    atom_mask = atom_mask.to(device)
    aatype = aatype.to(device)
    residue_idx = residue_idx.to(device)

    n_res = atom_positions.shape[0]
    residue_mask = torch.ones((1, n_res), dtype=torch.float32, device=device)
    chain_encoding = torch.zeros((1, n_res), dtype=torch.long, device=device)
    external_contacts = torch.zeros((1, n_res), dtype=torch.long, device=device)

    if model.diffuse_sidechain:
        # Flatten atom37 -> augment -> reshape to [B, N, 37*3]
        flat_coords = atom_positions.reshape(1, n_res * 37, 3)
        flat_mask = atom_mask.reshape(1, n_res * 37)
        aug_coords = center_random_augmentation(flat_coords, flat_mask)
        aug_coords = aug_coords * flat_mask[..., None]
        coords = aug_coords.reshape(1, n_res, 37 * 3)
        atom_mask_input = atom_mask.unsqueeze(0)  # [1, N, 37]
    else:
        ca_coords = atom_positions[:, 1, :].unsqueeze(0)  # [1, N, 3]
        ca_mask = atom_mask[:, 1].unsqueeze(0)  # [1, N]
        coords = center_random_augmentation(ca_coords, ca_mask)
        atom_mask_input = torch.ones((1, n_res), dtype=torch.float32, device=device)

    # Match ProteinEBM `score_decoys.py` behavior for ensemble_size=0:
    # - r_noisy is the clean coords (augmented) even when t>0
    # - selfcond_coords is the template coords when enabled
    r_noisy = coords
    selfcond_coords = coords if template_self_condition else torch.zeros_like(coords)

    input_feats = {
        "aatype": aatype.unsqueeze(0),  # [1, N]
        "mask": residue_mask,  # [1, N]
        "residue_idx": residue_idx.unsqueeze(0),  # [1, N]
        "chain_encoding": chain_encoding,  # [1, N]
        "external_contacts": external_contacts,  # [1, N]
        "selfcond_coords": selfcond_coords,  # [1, N, 3] or [1, N, 37*3]
        "r_noisy": r_noisy,  # [1, N, 3] or [1, N, 37*3]
        "atom_mask": atom_mask_input,  # [1, N] or [1, N, 37]
        "t": torch.tensor([t], dtype=torch.float32, device=device),  # [1]
    }
    return input_feats


def score_decoy_pdb(
    pdb_path: str,
    model: ProteinEBM,
    t: float,
    template_self_condition: bool,
    device: torch.device,
) -> float:
    input_feats = build_input_feats_from_pdb(
        pdb_path=pdb_path,
        model=model,
        t=t,
        template_self_condition=template_self_condition,
        device=device,
    )

    with torch.no_grad():
        out = model.compute_energy(input_feats)
    return float(out["energy"].detach().cpu().item())


def score_decoy_pdbs_batched(
    pdb_paths: List[str],
    model: ProteinEBM,
    t: float,
    template_self_condition: bool,
    device: torch.device,
    batch_size: int = 32,
    batch_size_ref: "List[int] | None" = None,
    verbose: bool = False,
) -> Dict[int, float]:
    """Score multiple decoy PDBs using batched inference.

    All PDBs should have the same residue count (true for decoys of one protein).
    Falls back to single-structure scoring on OOM.

    Args:
        batch_size_ref: Optional 1-element list holding the current batch size.
            Mutated in-place (decreased) on OOM. Passing the same list across
            successive proteins ensures the batch size never grows back — correct
            because memory consumption is monotonic with protein length.

    Returns:
        Dict mapping pdb_path index -> energy.
    """
    # Phase 1: Build per-PDB features (sequential PDB parsing)
    feats_list: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    results: Dict[int, float] = {}
    for i, pdb_path in enumerate(pdb_paths):
        try:
            feats = build_input_feats_from_pdb(pdb_path, model, t, template_self_condition, device)
            feats_list.append((i, feats))
        except Exception as e:
            print(f"  Warning: failed to build features for {pdb_path}: {e}", flush=True)

    if not feats_list:
        return results

    # Print input feature stats for the first PDB of this protein (verbose only)
    if verbose:
        first_feats = feats_list[0][1]
        print(f"  Input tensor stats (first PDB, {len(feats_list)} structures total):", flush=True)
        for key, val in first_feats.items():
            print(f"    {key}: {_tensor_stats(key, val)}", flush=True)

    # Phase 2: Batched inference — all decoys share the same residue count
    current_batch_size = batch_size_ref[0] if batch_size_ref is not None else batch_size
    remaining = list(feats_list)
    nan_hook_fired = [False]  # fire forward-hook NaN trace at most once per call

    while remaining:
        chunk = remaining[:current_batch_size]
        remaining = remaining[current_batch_size:]

        if current_batch_size == 1:
            # Single-structure fallback
            for idx, feats in chunk:
                try:
                    with torch.no_grad():
                        out = model.compute_energy(feats)
                    results[idx] = float(out["energy"].detach().cpu().item())
                except Exception as e:
                    print(f"  Warning: scoring failed for index {idx}: {e}", flush=True)
            continue

        try:
            indices = [idx for idx, _ in chunk]
            batched = {}
            for key in chunk[0][1]:
                batched[key] = torch.cat([feats[key] for _, feats in chunk], dim=0)

            with torch.no_grad():
                out = model.compute_energy(batched)

            energies = out["energy"].detach().cpu()
            for j, idx in enumerate(indices):
                results[idx] = float(energies[j].item())

            # --- Diagnostic: print energy stats for this batch (verbose only) ---
            e_arr = energies.float().numpy()
            n_nan_e = int(np.isnan(e_arr).sum())
            if verbose:
                finite_e = e_arr[~np.isnan(e_arr)]
                if len(finite_e):
                    print(f"  Batch energies: size={len(e_arr)} "
                          f"min={finite_e.min():.4g} max={finite_e.max():.4g} "
                          f"mean={finite_e.mean():.4g} nan={n_nan_e}", flush=True)
                else:
                    print(f"  Batch energies: size={len(e_arr)} ALL NaN", flush=True)

            # --- Diagnostic: forward-hook NaN trace (verbose only, fires at most once) ---
            if verbose and n_nan_e > 0 and not nan_hook_fired[0]:
                nan_hook_fired[0] = True
                print("  Tracing NaN source with forward hooks (single-structure re-run)...", flush=True)
                nan_detected: List[str] = []

                def _make_hook(mname: str):
                    def _hook(module, inp, output):
                        if nan_detected:
                            return
                        outs = output if isinstance(output, (list, tuple)) else [output]
                        for o in outs:
                            if isinstance(o, torch.Tensor) and (o.isnan().any() or o.isinf().any()):
                                nan_detected.append(mname)
                                print(f"  !!! First NaN/Inf in module: {mname}", flush=True)
                                print(f"      output: {_tensor_stats(mname, o)}", flush=True)
                    return _hook

                hooks = [m.register_forward_hook(_make_hook(n)) for n, m in model.named_modules()]
                try:
                    # Re-run only the first NaN element as a batch-of-1
                    nan_batch_pos = int(np.where(np.isnan(e_arr))[0][0])
                    single = {k: v[nan_batch_pos:nan_batch_pos + 1] for k, v in batched.items()}
                    print(f"  Input tensor stats for first-NaN element (pos {nan_batch_pos}):", flush=True)
                    for key, val in single.items():
                        print(f"    {key}: {_tensor_stats(key, val)}", flush=True)
                    with torch.no_grad():
                        model.compute_energy(single)
                finally:
                    for h in hooks:
                        h.remove()
                if not nan_detected:
                    print("  (no NaN/Inf caught by hooks — NaN may originate in inputs above)", flush=True)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            n_res = feats_list[0][1]["aatype"].shape[1] if feats_list else "?"
            print(f"  OOM at n_res={n_res}: reducing batch size to {current_batch_size}", flush=True)
            if batch_size_ref is not None:
                batch_size_ref[0] = current_batch_size
            remaining = chunk + remaining  # re-queue failed chunk

    return results


def run_proteinebm_scoring_for_protein(
    protein_id: str,
    inference_output_dir: str,
    output_dir: str,
    reference_cif: str | None,
    reference_chain: str | None,
    proteinebm_config: str,
    proteinebm_checkpoint: str,
    t: float,
    template_self_condition: bool,
    batch_size: int = 32,
    model: "ProteinEBM | None" = None,
    batch_size_ref: "List[int] | None" = None,
    verbose: bool = False,
    num_workers: Optional[int] = None,
) -> Tuple[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, _ = load_proteinebm_model(
            config_path=proteinebm_config,
            checkpoint_path=proteinebm_checkpoint,
            device=device,
        )

    inference_dir = Path(inference_output_dir)
    pdb_files = sorted(inference_dir.glob(f"{protein_id}_*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No decoy PDB files found under {inference_output_dir} matching {protein_id}_*.pdb")

    os.makedirs(output_dir, exist_ok=True)
    scores_csv_path = os.path.join(output_dir, f"proteinebm_scores_{protein_id}.csv")
    summary_path = os.path.join(output_dir, f"proteinebm_summary_{protein_id}.json")

    del num_workers
    ref_cif_abs: Optional[str] = None
    if reference_cif is not None:
        ref_cif_abs = os.path.abspath(os.path.expanduser(reference_cif))

    start_time = time.time()

    # Batched energy scoring
    pdb_path_strs = [str(p) for p in pdb_files]
    energy_map = score_decoy_pdbs_batched(
        pdb_paths=pdb_path_strs,
        model=model,
        t=t,
        template_self_condition=template_self_condition,
        device=device,
        batch_size=batch_size,
        batch_size_ref=batch_size_ref,
        verbose=verbose,
    )

    results: List[Dict[str, object]] = []
    for i, pdb_path in enumerate(pdb_files):
        if i not in energy_map:
            continue
        energy = energy_map[i]
        results.append(
            {
                "protein_id": protein_id,
                "structure_file": pdb_path.name,
                "structure_path": str(pdb_path),
                "t": t,
                "energy": energy,
            }
        )

    # Write CSV
    with open(scores_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["protein_id", "structure_file", "structure_path", "t", "energy"],
        )
        writer.writeheader()
        writer.writerows(results)

    n_nan = sum(1 for r in results if np.isnan(float(r["energy"])))
    if n_nan > 0:
        print(f"  WARNING: {n_nan}/{len(results)} energies are NaN for {protein_id}!", flush=True)

    runtime_s = time.time() - start_time
    summary = {
        "protein_id": protein_id,
        "total_structures": len(results),
        "successful_scores": len(results),
        "runtime_seconds": runtime_s,
        "t": t,
        "reference_structure": os.path.abspath(os.path.expanduser(reference_cif)) if reference_cif else None,
        "chain": reference_chain,
        "proteinebm_config": os.path.abspath(os.path.expanduser(proteinebm_config)),
        "proteinebm_checkpoint": os.path.abspath(os.path.expanduser(proteinebm_checkpoint)),
        "template_self_condition": template_self_condition,
        "scores_csv": os.path.abspath(scores_csv_path),
        "status": "scored_energy_only",
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return scores_csv_path, summary_path


def _get_protein_length_from_pdb(inference_output_dir: str, protein_id: str) -> int:
    """Fast protein length estimate by counting CA atoms in the first available PDB."""
    pdb_files = sorted(Path(inference_output_dir).glob(f"{protein_id}_*.pdb"))
    if not pdb_files:
        return 0
    count = 0
    with open(pdb_files[0]) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="ProteinEBM scoring for Proteina decoys (single-pass energy)")
    # Single-protein mode
    parser.add_argument("--protein_id", default=None, help="Protein identifier (e.g. 1a2y_C). Mutually exclusive with --proteins_json.")
    parser.add_argument("--reference_cif", default=None, help="Path to native/reference CIF (for TMscore ground-truth). Optional; if omitted, TM-score metrics are skipped.")
    parser.add_argument("--chain", default=None, help="Chain ID in reference CIF (default: taken from protein_id if possible)")
    parser.add_argument("--inference_output_dir", default=None, help="Directory containing decoy PDB files")
    parser.add_argument("--output_dir", default=None, help="Directory to write proteinebm_analysis outputs")
    # Multi-protein mode
    parser.add_argument(
        "--proteins_json", default=None,
        help=(
            "Path to a JSON file containing a list of protein configs to process with one "
            "model load. Each entry: {protein_id, inference_output_dir, output_dir, "
            "reference_cif (optional), reference_chain (optional)}. "
            "Mutually exclusive with --protein_id."
        ),
    )
    # Shared args
    parser.add_argument("--proteinebm_config", required=True, help="Path to ProteinEBM base_pretrain.yaml")
    parser.add_argument("--proteinebm_checkpoint", required=True, help="Path to ProteinEBM checkpoint (.pt)")
    parser.add_argument("--t", type=float, default=0.1, help="Diffusion time t for scoring (default: 0.1)")
    parser.add_argument(
        "--proteinebm_template_self_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use template coordinates for self-conditioning input (default: True)",
    )
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for ProteinEBM inference (default: 32). Auto-reduces on OOM.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed diagnostics: model weight check, input tensor stats, "
                             "per-batch energy stats, and forward-hook NaN tracing.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Parallel CPU workers for USalign TM-score vs reference (default: clamp(os.cpu_count()), 1–64).",
    )

    args = parser.parse_args()

    if args.proteins_json and args.protein_id:
        print("Error: --proteins_json and --protein_id are mutually exclusive", flush=True)
        sys.exit(1)
    if not args.proteins_json and not args.protein_id:
        print("Error: one of --protein_id or --proteins_json is required", flush=True)
        sys.exit(1)

    if args.proteins_json:
        # ── Multi-protein mode: load model once, process all proteins sorted by length ──
        with open(args.proteins_json) as f:
            protein_configs: List[Dict] = json.load(f)

        print(f"Multi-protein mode: {len(protein_configs)} proteins, batch_size={args.batch_size}", flush=True)

        # Sort by sequence length (ascending) so similar-length proteins are adjacent
        # and the batch_size_cache is most effective.
        protein_configs.sort(
            key=lambda c: _get_protein_length_from_pdb(c["inference_output_dir"], c["protein_id"])
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.verbose and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            mem_free, mem_total = torch.cuda.mem_get_info(device)
            print(f"GPU: {props.name}  VRAM free/total: {mem_free/1e9:.1f}/{mem_total/1e9:.1f} GB", flush=True)
        model, _ = load_proteinebm_model(
            config_path=args.proteinebm_config,
            checkpoint_path=args.proteinebm_checkpoint,
            device=device,
        )
        if args.verbose:
            print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
            _check_model_weights(model)
        # Single mutable batch size — only ever decreases as proteins get longer
        batch_size_ref: List[int] = [args.batch_size]

        for i, cfg in enumerate(protein_configs):
            protein_id = cfg["protein_id"]
            print(f"[{i+1}/{len(protein_configs)}] Scoring {protein_id} (batch_size={batch_size_ref[0]})", flush=True)
            try:
                run_proteinebm_scoring_for_protein(
                    protein_id=protein_id,
                    inference_output_dir=cfg["inference_output_dir"],
                    output_dir=cfg["output_dir"],
                    reference_cif=cfg.get("reference_cif"),
                    reference_chain=cfg.get("reference_chain"),
                    proteinebm_config=args.proteinebm_config,
                    proteinebm_checkpoint=args.proteinebm_checkpoint,
                    t=args.t,
                    template_self_condition=args.proteinebm_template_self_condition,
                    batch_size=args.batch_size,
                    model=model,
                    batch_size_ref=batch_size_ref,
                    verbose=args.verbose,
                    num_workers=args.num_workers,
                )
                print(f"  Done: {protein_id}", flush=True)
            except Exception as e:
                print(f"  ERROR scoring {protein_id}: {e}", flush=True)
        print(f"Multi-protein scoring complete. final_batch_size={batch_size_ref[0]}", flush=True)

    else:
        # ── Single-protein mode (backward compatible) ──
        print(f"Starting ProteinEBM scoring for {args.protein_id} (batch_size={args.batch_size})", flush=True)
        chain = args.chain
        if chain is None and "_" in args.protein_id:
            chain = args.protein_id.split("_", 1)[1]
        scores_csv, summary_json = run_proteinebm_scoring_for_protein(
            protein_id=args.protein_id,
            inference_output_dir=args.inference_output_dir,
            output_dir=args.output_dir,
            reference_cif=args.reference_cif,
            reference_chain=chain,
            proteinebm_config=args.proteinebm_config,
            proteinebm_checkpoint=args.proteinebm_checkpoint,
            t=args.t,
            template_self_condition=args.proteinebm_template_self_condition,
            batch_size=args.batch_size,
            verbose=args.verbose,
            num_workers=args.num_workers,
        )
        print(f"Completed ProteinEBM scoring for {args.protein_id}", flush=True)
        print(f"Scores: {scores_csv}", flush=True)
        print(f"Summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()



