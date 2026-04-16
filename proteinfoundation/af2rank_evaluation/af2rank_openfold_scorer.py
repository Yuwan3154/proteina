#!/usr/bin/env python3
"""
AF2Rank Scorer Module - OpenFold (PyTorch) Backend

Provides the same AF2Rank scoring functionality as af2rank_scorer.py but using
OpenFold (PyTorch) instead of ColabDesign (JAX). Both backends load the same
AlphaFold weights and should produce roughly equivalent results (pTM diff < 0.01).

Supports model_1_ptm and model_2_ptm for the AF2Rank-on-ProteinEBM top-k protocol.
"""

import gc
import glob
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm

import openfold.np.residue_constants as rc
from openfold.np import protein as openfold_protein
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.af2rank_evaluation.usalign_tabular import parse_usalign_pair_outfmt2
from scipy.stats import spearmanr


KALIGN_BINARY_PATH = "/usr/bin/kalign"
DEFAULT_PARAMS_DIR = os.path.expanduser("~/openfold/openfold/resources/params")

_USALIGN_PARALLEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


# ---------------------------------------------------------------------------
# Utility functions (self-contained, no ColabDesign/JAX dependency)
# ---------------------------------------------------------------------------

def tmscore(x, y, tmscore_exe="USalign", env=None):
    """Calculate TMscore between two coordinate arrays (CA atoms)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        for f, z in zip([f1, f2], [x, y]):
            coords = z if isinstance(z, np.ndarray) else z.detach().cpu().numpy()
            for k, c in enumerate(coords):
                f.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                        % (k + 1, "CA", "ALA", "A", k + 1, c[0], c[1], c[2], 1, 0))
            f.flush()

    output_text = None
    used_exe = None
    for exe in [tmscore_exe, "./TMscore", "TMscore", "/usr/local/bin/TMscore"]:
        if os.path.exists(exe) or os.system(f"which {exe} > /dev/null 2>&1") == 0:
            cmd = [exe, f1.name, f2.name]
            if os.path.basename(exe) == "USalign":
                cmd += ["-TMscore", "5", "-outfmt", "2"]
            subprocess_env = os.environ.copy()
            if env is not None:
                subprocess_env.update(env)
            output_text = subprocess.check_output(cmd, text=True, env=subprocess_env)
            used_exe = exe
            break
    else:
        x_np = x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
        y_np = y if isinstance(y, np.ndarray) else y.detach().cpu().numpy()
        rmsd = np.sqrt(np.mean(np.sum((x_np - y_np) ** 2, axis=-1)))
        os.unlink(f1.name)
        os.unlink(f2.name)
        return {"rms": float(rmsd), "tms": 0.0, "gdt": 0.0}

    os.unlink(f1.name)
    os.unlink(f2.name)

    if os.path.basename(used_exe) == "USalign":
        return parse_usalign_pair_outfmt2(output_text)

    parse_float = lambda s: float(s.split("=")[1].split()[0])
    o = {"rms": 0.0, "tms": 0.0, "gdt": 0.0}
    for line in output_text.splitlines():
        line = line.rstrip()
        if line.startswith("RMSD"):
            o["rms"] = parse_float(line)
        elif line.startswith("TM-score"):
            o["tms"] = parse_float(line)
        elif line.startswith("GDT-TS-score"):
            o["gdt"] = parse_float(line)
    return o


def _save_openfold_prediction_pdb(reference_sequence: str, out: dict, output_pdb: str) -> None:
    atom37 = out["final_atom_positions"].detach().cpu().numpy()
    atom37_mask = out["final_atom_mask"].detach().cpu().numpy() if "final_atom_mask" in out else None
    aatype = np.array([rc.restype_order.get(aa, rc.restype_num) for aa in reference_sequence], dtype=np.int32)
    residue_index = np.arange(atom37.shape[0], dtype=np.int32) + 1
    chain_index = np.zeros(atom37.shape[0], dtype=np.int32)
    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    write_prot_to_pdb(
        atom37,
        output_pdb,
        aatype=aatype,
        atom37_mask=atom37_mask,
        residue_index=residue_index,
        chain_index=chain_index,
        overwrite=True,
        no_indexing=True,
    )


def _is_ca_only_pdb(pdb_file):
    """Check if a PDB file contains only CA atoms."""
    atom_names = set()
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_names.add(line[12:16].strip())
                if len(atom_names) > 1:
                    return False
    return atom_names == {"CA"} or len(atom_names) == 0


def _fix_pdb_model_number(pdb_path: str) -> None:
    """Rewrite MODEL 0 → MODEL 1 in cg2all PDB output.

    MDAnalysis (used by cg2all) writes 0-indexed MODEL records, but the PDB
    format standard and OpenFold's template reader expect 1-indexed models.
    """
    import re
    with open(pdb_path) as f:
        text = f.read()
    patched = re.sub(r"^MODEL(\s+)0(\s*)$", r"MODEL\g<1>1\2", text, flags=re.MULTILINE)
    if patched != text:
        with open(pdb_path, "w") as f:
            f.write(patched)


class CG2AllReconstructor:
    """Reconstruct full all-atom PDBs from CA-only PDBs using cg2all (GPU-batched)."""

    def __init__(self, device="cuda"):
        import dgl
        import cg2all.lib.libcg
        import cg2all.lib.libmodel
        from cg2all.lib.libconfig import MODEL_HOME
        from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch
        from cg2all.lib.libter import patch_termini

        self.dgl = dgl
        self.PredictionData = PredictionData
        self.create_trajectory_from_batch = create_trajectory_from_batch
        self.patch_termini = patch_termini

        self.device = torch.device(device)
        ckpt_path = MODEL_HOME / "CalphaBasedModel.ckpt"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt["hyper_parameters"]
        self.cg_model = cg2all.lib.libcg.CalphaBasedModel
        config = cg2all.lib.libmodel.set_model_config(config, self.cg_model, flattened=False)
        self.config = config
        model = cg2all.lib.libmodel.Model(config, self.cg_model, compute_loss=False)
        state_dict = ckpt["state_dict"]
        for key in list(state_dict):
            state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.set_constant_tensors(self.device)
        model.eval()
        self.model = model
        logger.info(f"cg2all CalphaBasedModel loaded on {device}")

    def reconstruct_single(self, pdb_file):
        """Reconstruct a single CA-only PDB to all-atom. Returns temp PDB path."""
        ds = self.PredictionData(pdb_file, self.cg_model, radius=self.config.globals.radius)
        loader = self.dgl.dataloading.GraphDataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
        batch = next(iter(loader)).to(self.device)
        with torch.no_grad():
            R = self.model.forward(batch)[0]["R"]
        traj_s, _ = self.create_trajectory_from_batch(batch, R)
        output = self.patch_termini(traj_s[0])
        fd, out_path = tempfile.mkstemp(suffix=".pdb", prefix="cg2all_")
        os.close(fd)
        output.save(out_path)
        _fix_pdb_model_number(out_path)
        return out_path

    def reconstruct_batch(self, pdb_files, batch_size=32):
        """Reconstruct multiple CA-only PDBs in chunked GPU forward passes.

        Returns dict mapping input path -> temp all-atom PDB path.
        """
        graphs = []
        valid_indices = []
        for i, pdb_file in enumerate(pdb_files):
            try:
                ds = self.PredictionData(pdb_file, self.cg_model, radius=self.config.globals.radius)
                g = ds[0]
                graphs.append(g)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"cg2all failed to load {pdb_file}: {e}")

        if not graphs:
            return {}

        result = {}
        for start in range(0, len(graphs), batch_size):
            chunk_graphs = graphs[start:start + batch_size]
            chunk_indices = valid_indices[start:start + batch_size]
            batched = self.dgl.batch(chunk_graphs).to(self.device)
            with torch.no_grad():
                R = self.model.forward(batched)[0]["R"]
            traj_s, _ = self.create_trajectory_from_batch(batched, R)
            for idx, traj in zip(chunk_indices, traj_s):
                output = self.patch_termini(traj)
                fd, out_path = tempfile.mkstemp(suffix=".pdb", prefix="cg2all_")
                os.close(fd)
                output.save(out_path)
                _fix_pdb_model_number(out_path)
                result[pdb_files[idx]] = out_path
        return result


# Module-level singleton (lazy-loaded)
_cg2all_reconstructor = None


def _get_cg2all_reconstructor():
    global _cg2all_reconstructor
    if _cg2all_reconstructor is None:
        _cg2all_reconstructor = CG2AllReconstructor(device="cuda")
    return _cg2all_reconstructor


def reconstruct_all_atom(pdb_file):
    """Reconstruct full all-atom PDB from CA-only PDB using cg2all (GPU).

    Returns path to a temporary all-atom PDB file (caller must delete).
    If the input already has all atoms, returns None (no reconstruction needed).
    """
    if not _is_ca_only_pdb(pdb_file):
        return None
    try:
        reconstructor = _get_cg2all_reconstructor()
        return reconstructor.reconstruct_single(pdb_file)
    except Exception as e:
        logger.warning(f"cg2all reconstruction failed for {pdb_file}: {e}")
        return None


def convert_pdb_to_cif(pdb_file, chain_id=None):
    """Convert a PDB file to a temporary mmCIF file using OpenFold's protein module.

    Uses openfold.np.protein.to_modelcif() which produces CIF files that are
    compatible with OpenFold's mmcif_parsing (unlike BioPython's MMCIFIO which
    misses required fields like _entry.id).
    """
    with open(pdb_file) as f:
        pdb_str = f.read()
    prot = openfold_protein.from_pdb_string(pdb_str, chain_id=chain_id)
    # to_modelcif expects 1-indexed residue_index (ihm convention),
    # but from_pdb_string produces 0-indexed. Fix by adding 1.
    prot = openfold_protein.Protein(
        atom_positions=prot.atom_positions,
        atom_mask=prot.atom_mask,
        aatype=prot.aatype,
        residue_index=prot.residue_index + 1,
        chain_index=prot.chain_index,
        b_factors=prot.b_factors,
    )
    cif_str = openfold_protein.to_modelcif(prot)

    temp_cif = tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False)
    temp_cif.write(cif_str)
    temp_cif.close()
    return temp_cif.name


def get_sequence_from_structure(pdb_file, chain=None):
    """Extract amino acid sequence from PDB/CIF file using BioPython."""
    is_cif = pdb_file.lower().endswith('.cif')
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure('protein', pdb_file)

    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    # Find the target chain
    target_chain = None
    for model in structure:
        for c in model:
            if chain is None or c.id == chain:
                target_chain = c
                break
        if target_chain is not None:
            break

    if target_chain is None:
        # Fallback: use first chain
        for model in structure:
            for c in model:
                target_chain = c
                break
            break

    sequence = []
    for residue in target_chain:
        resname = residue.get_resname().strip()
        if resname in three_to_one:
            sequence.append(three_to_one[resname])
        elif residue.id[0] == ' ':  # Standard residue but unknown
            sequence.append('X')
    return "".join(sequence)


def get_ca_coords_from_structure(pdb_file, chain=None):
    """Extract CA coordinates from PDB/CIF file using BioPython."""
    is_cif = pdb_file.lower().endswith('.cif')
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure('protein', pdb_file)

    target_chain = None
    for model in structure:
        for c in model:
            if chain is None or c.id == chain:
                target_chain = c
                break
        if target_chain is not None:
            break

    if target_chain is None:
        for model in structure:
            for c in model:
                target_chain = c
                break
            break

    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    coords = []
    for residue in target_chain:
        resname = residue.get_resname().strip()
        if resname not in three_to_one and residue.id[0] != ' ':
            continue
        if 'CA' in residue:
            coords.append(residue['CA'].get_vector().get_array())
    return np.array(coords, dtype=np.float32)


def rescale(a, amin=None, amax=None):
    """Rescale array values to [0,1] range."""
    a = np.copy(a)
    if amin is None:
        amin = a.min()
    if amax is None:
        amax = a.max()
    if amax == amin:
        return np.ones_like(a) * 0.5
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin) / (amax - amin)


def plot_metric(scores, x="ptm", y="composite",
                title=None, diag=False, scale_axis=True,
                save_path=None, **kwargs):
    """Create AF2Rank analysis plot."""
    plt.figure(figsize=(8, 6), dpi=100)
    if title is not None:
        plt.title(title)

    x_vals = np.array([k.get(x, 0.0) for k in scores if "error" not in k])
    y_vals = np.array([k.get(y, 0.0) for k in scores if "error" not in k])
    c = rescale(np.array([k.get("plddt", 0.7) for k in scores if "error" not in k]), 0.5, 0.9)

    if len(x_vals) == 0:
        plt.text(0.5, 0.5, 'No valid data to plot', transform=plt.gca().transAxes,
                 ha='center', va='center', fontsize=12)
        correlation = 0.0
    else:
        plt.scatter(x_vals, y_vals, c=c * 0.75, s=20, vmin=0, vmax=1,
                    cmap="gist_rainbow", **kwargs)

        if diag and len(x_vals) > 0:
            lims = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
            plt.plot(lims, lims, color="black", linestyle="--", alpha=0.5)

        valid_mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        if valid_mask.sum() > 1:
            correlation = spearmanr(x_vals[valid_mask], y_vals[valid_mask]).correlation
            if not np.isnan(correlation):
                plt.text(0.05, 0.95, f'Spearman R: {correlation:.3f}',
                         transform=plt.gca().transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                correlation = 0.0
        else:
            correlation = 0.0

    labels = {
        "ptm": "Predicted TM-score (pTM)",
        "plddt": "Predicted LDDT (pLDDT)",
        "composite": "AF2Rank Score (pTM x pLDDT)",
        "pae_mean": "Predicted Aligned Error",
        "tm_ref_template": "TM-score (Reference vs Template)",
        "tm_ref_pred": "TM-score (Reference vs Prediction)",
        "tm_template_pred": "TM-score (Template vs Prediction)",
        "rmsd_ref_template": "RMSD (Reference vs Template)",
        "rmsd_ref_pred": "RMSD (Reference vs Prediction)",
        "rmsd_template_pred": "RMSD (Template vs Prediction)",
        "gdt_ref_template": "GDT-TS (Reference vs Template)",
        "gdt_ref_pred": "GDT-TS (Reference vs Prediction)",
        "gdt_template_pred": "GDT-TS (Template vs Prediction)",
    }

    plt.xlabel(labels.get(x, x.replace('_', ' ').title()))
    plt.ylabel(labels.get(y, y.replace('_', ' ').title()))

    if scale_axis and x in ["ptm", "plddt", "composite"] and y in ["ptm", "plddt", "composite"]:
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)

    if len(x_vals) > 0:
        cbar = plt.colorbar()
        cbar.set_label('pLDDT (scaled)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {save_path}")

    plt.close()
    return correlation


def plot_af2rank_results(scores: List[Dict], output_dir: str, protein_id: str):
    """Generate comprehensive AF2Rank analysis plots."""
    plot_configs = [
        {"x": "tm_ref_template", "y": "composite",
         "title": "AF2Rank Analysis: Template Quality vs AF2 Confidence",
         "filename": f"af2rank_{protein_id}_template_quality_vs_composite.png"},
        {"x": "tm_ref_pred", "y": "ptm",
         "title": "AF2Rank Analysis: Prediction Quality vs pTM",
         "filename": f"af2rank_{protein_id}_prediction_quality_vs_ptm.png"},
        {"x": "tm_ref_template", "y": "tm_ref_pred", "diag": True,
         "title": "AF2Rank Analysis: Template vs Prediction Quality",
         "filename": f"af2rank_{protein_id}_template_vs_prediction_quality.png"},
        {"x": "composite", "y": "tm_ref_pred",
         "title": "AF2Rank Analysis: AF2 Composite Score vs True Quality",
         "filename": f"af2rank_{protein_id}_composite_vs_true_quality.png"},
        {"x": "ptm", "y": "composite",
         "title": "AF2Rank Analysis: pTM vs Composite Score",
         "filename": f"af2rank_{protein_id}_ptm_vs_composite.png"},
        {"x": "plddt", "y": "composite",
         "title": "AF2Rank Analysis: pLDDT vs Composite Score",
         "filename": f"af2rank_{protein_id}_plddt_vs_composite.png"},
        {"x": "ptm", "y": "plddt",
         "title": "AF2 Internal: pTM vs pLDDT correlation",
         "filename": f"af2rank_{protein_id}_ptm_vs_plddt.png"},
        {"x": "pae_mean", "y": "composite",
         "title": "AF2Rank Analysis: PAE vs Composite Score",
         "filename": f"af2rank_{protein_id}_pae_mean_vs_composite.png"},
    ]

    correlations = {}
    for plot_config in plot_configs:
        filename = plot_config.pop("filename")
        save_path = os.path.join(output_dir, filename)
        correlation = plot_metric(scores, save_path=save_path, **plot_config)
        correlations[filename] = correlation

    return correlations


def save_af2rank_scores(scores: List[Dict], output_dir: str, protein_id: str) -> str:
    """Save AF2Rank scores to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(scores)
    csv_path = os.path.join(output_dir, f"af2rank_scores_{protein_id}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(scores)} AF2Rank scores to: {csv_path}")
    return csv_path


def load_af2rank_scores_from_csv(csv_path: str) -> List[Dict]:
    """Load AF2Rank scores from CSV file."""
    df = pd.read_csv(csv_path)
    scores = df.to_dict('records')
    logger.info(f"Loaded {len(scores)} scores from: {csv_path}")
    return scores


def _detect_chain_in_cif(cif_file, chain_hint=None):
    """Detect the correct chain ID in a CIF file, with fallback mapping."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', cif_file)
    available_chains = []
    for model in structure:
        for chain in model:
            available_chains.append(chain.id)
        break

    if chain_hint is not None and chain_hint in available_chains:
        return chain_hint

    if chain_hint is not None:
        chain_mappings = {
            'A': ['E', 'N', 'X', '1', 'a'],
            'B': ['F', 'O', 'Y', '2', 'b'],
            'C': ['G', 'P', 'Z', '3', 'c'],
        }
        if chain_hint in chain_mappings:
            for alt in chain_mappings[chain_hint]:
                if alt in available_chains:
                    return alt

    return available_chains[0] if available_chains else 'A'


# ---------------------------------------------------------------------------
# OpenFold AF2Rank Scorer
# ---------------------------------------------------------------------------

class OpenFoldAF2Rank:
    """AF2Rank implementation using OpenFold (PyTorch) backend.

    Drop-in replacement for ModernAF2Rank from af2rank_scorer.py.
    Produces identical output keys so downstream analysis/plotting works unchanged.
    """

    def __init__(
        self,
        reference_pdb: str,
        chain: Optional[str] = None,
        model_name: str = "model_1_ptm",
        recycles: int = 3,
        debug: bool = False,
        chunk_size: Optional[int] = None,
        use_deepspeed_evoformer_attention: bool = False,
        use_cuequivariance_attention: bool = False,
        use_cuequivariance_multiplicative_update: bool = False,
        skip_ref_metrics: bool = False,
    ):
        if chain is None:
            chain = "A"
        self.chain = chain
        self.model_name = model_name
        self.debug = debug
        self.recycles = recycles
        self.skip_ref_metrics = skip_ref_metrics

        # Detect correct chain in reference CIF
        is_cif = reference_pdb.lower().endswith('.cif')
        if is_cif:
            detected_chain = _detect_chain_in_cif(reference_pdb, chain)
        else:
            detected_chain = chain

        # Extract reference sequence and coordinates
        self.reference_sequence = get_sequence_from_structure(reference_pdb, detected_chain)
        self.reference_coords = get_ca_coords_from_structure(reference_pdb, detected_chain)

        logger.info(f"Reference sequence length: {len(self.reference_sequence)}")
        logger.info(f"Reference coords shape: {self.reference_coords.shape}")

        # Build residue_type tensor from reference sequence
        self._residue_type = torch.tensor(
            [[rc.restype_order.get(aa, rc.restype_num) for aa in self.reference_sequence]],
            dtype=torch.long,
        )
        self._mask = torch.ones((1, len(self.reference_sequence)), dtype=torch.float32)

        # Load OpenFold model
        jax_params_path = os.path.join(DEFAULT_PARAMS_DIR, f"params_{model_name}.npz")
        if not os.path.exists(jax_params_path):
            raise FileNotFoundError(f"Weight file not found: {jax_params_path}")

        logger.info(f"Loading OpenFold model {model_name} from {jax_params_path}...")
        self.model = OpenFoldTemplateInference(
            model_name=model_name,
            jax_params_path=jax_params_path,
            skip_template_alignment=True,  # No alignment needed (sequences match)
            max_recycling_iters=recycles,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )

        # Set chunk_size if specified (0 or 1 disables chunking)
        if chunk_size is not None:
            self.model.cfg.globals.chunk_size = chunk_size if chunk_size > 0 else None
            logger.info(f"Set chunk_size to {chunk_size}")

        logger.info("OpenFold model loaded successfully")

    def reset_reference(self, reference_pdb: str, chain: Optional[str] = None) -> None:
        """Update the reference structure without reloading the model weights.

        Call this between proteins when reusing the same scorer instance across
        multiple proteins in a single model-load pass.
        """
        if chain is None:
            chain = "A"
        self.chain = chain

        is_cif = reference_pdb.lower().endswith('.cif')
        detected_chain = _detect_chain_in_cif(reference_pdb, chain) if is_cif else chain

        self.reference_sequence = get_sequence_from_structure(reference_pdb, detected_chain)
        self.reference_coords = get_ca_coords_from_structure(reference_pdb, detected_chain)
        self._residue_type = torch.tensor(
            [[rc.restype_order.get(aa, rc.restype_num) for aa in self.reference_sequence]],
            dtype=torch.long,
        )
        self._mask = torch.ones((1, len(self.reference_sequence)), dtype=torch.float32)
        logger.info(f"reset_reference: new sequence length {len(self.reference_sequence)}")

    def _featurize(
        self,
        decoy_pdb: str,
        decoy_chain: Optional[str] = None,
        seed: int = 0,
        _original_pdb: Optional[str] = None,
    ) -> tuple:
        """CPU-only featurization for a single decoy. Returns (batch, template_coords).

        Separated from score_structure so that featurization for decoy i+1 can be
        prefetched in a background thread while the GPU runs decoy i's forward pass.
        Temp CIF files are created and deleted internally — nothing leaks to the caller.
        """
        if decoy_chain is None:
            decoy_chain = "A"

        ca_source = _original_pdb if _original_pdb else decoy_pdb
        template_coords = get_ca_coords_from_structure(ca_source, decoy_chain)

        # If decoy is already all-atom, use directly; otherwise reconstruct.
        # Note: score_proteina_structures_openfold pre-reconstructs all CA-only
        # structures via cg2all before the loop, so this branch is a fallback
        # for direct callers of score_structure with CA-only input.
        if _is_ca_only_pdb(decoy_pdb):
            allatom_pdb = reconstruct_all_atom(decoy_pdb)
            pdb_for_template = allatom_pdb if allatom_pdb else decoy_pdb
        else:
            allatom_pdb = None
            pdb_for_template = decoy_pdb

        temp_cif = convert_pdb_to_cif(pdb_for_template, chain_id=decoy_chain)
        try:
            cif_chain = _detect_chain_in_cif(temp_cif)
            # Set numpy/torch CPU seeds before build_batch.
            # torch.cuda.manual_seed is intentionally omitted: build_batch is
            # CPU-only, and setting the CUDA seed from a background thread is
            # unnecessary and could interfere with the GPU forward on the main thread.
            np.random.seed(seed)
            torch.manual_seed(seed)
            batch = self.model.build_batch(
                distogram_probs=None,
                residue_type=self._residue_type,
                mask=self._mask,
                template_mode="full_template",
                template_mmcif_path=temp_cif,
                template_chain_id=cif_chain,
                kalign_binary_path=KALIGN_BINARY_PATH,
                mask_template_aatype=True,  # AF2Rank: mask template sequence to all-X post-pipeline
            )
        finally:
            if os.path.exists(temp_cif):
                os.unlink(temp_cif)
            if allatom_pdb and os.path.exists(allatom_pdb):
                os.unlink(allatom_pdb)

        return batch, template_coords

    def score_structure(
        self,
        decoy_pdb: str,
        decoy_chain: Optional[str] = None,
        recycles: int = 3,
        seed: int = 0,
        output_pdb: Optional[str] = None,
        verbose: bool = False,
        _original_pdb: Optional[str] = None,
    ) -> Dict:
        """Score a single decoy structure using AF2Rank protocol via OpenFold.

        Args:
            _original_pdb: If provided, use this for TMscore CA extraction instead
                of decoy_pdb (useful when decoy_pdb is already a cg2all-reconstructed
                all-atom PDB and we want TMscore against the original CA trace).
        """
        if decoy_chain is None:
            decoy_chain = "A"

        if verbose:
            logger.debug(f"Scoring {decoy_pdb} with OpenFold AF2Rank")

        batch, template_coords = self._featurize(
            decoy_pdb,
            decoy_chain=decoy_chain,
            seed=seed,
            _original_pdb=_original_pdb,
        )

        with torch.no_grad():
            out = self.model.model(batch)

        scores = self._extract_scores(out)
        if output_pdb is not None:
            _save_openfold_prediction_pdb(self.reference_sequence, out, output_pdb)
            scores["predicted_structure_path"] = output_pdb
            scores["predicted_structure_file"] = os.path.basename(output_pdb)

        # TM scores: reference vs template/prediction, template vs prediction
        if not self.skip_ref_metrics:
            scores["tm_ref_template"] = tmscore(
                self.reference_coords, template_coords, env=_USALIGN_PARALLEL_ENV
            ).get("tms", 0.0)
        if "final_atom_positions" in out:
            pred_ca = out["final_atom_positions"][:, 1, :].detach().cpu().numpy()
            if not self.skip_ref_metrics:
                scores["tm_ref_pred"] = tmscore(
                    self.reference_coords, pred_ca, env=_USALIGN_PARALLEL_ENV
                ).get("tms", 0.0)
            scores["tm_template_pred"] = tmscore(
                template_coords, pred_ca, env=_USALIGN_PARALLEL_ENV
            ).get("tms", 0.0)

        gc.collect()
        torch.cuda.empty_cache()

        if verbose:
            logger.debug(
                f"OpenFold AF2Rank scores - pLDDT: {scores['plddt']:.3f}, "
                f"pTM: {scores['ptm']:.3f}, Composite: {scores['composite']:.3f}"
            )

        return scores

    def _extract_scores(self, out: dict) -> Dict:
        """Extract AF2 confidence metrics from OpenFold output."""
        scores = {}

        # pTM score (0-1)
        scores["ptm"] = float(out["ptm_score"].item()) if "ptm_score" in out else 0.0

        # pLDDT (OpenFold returns 0-100, normalize to 0-1 to match ColabDesign)
        if "plddt" in out:
            plddt_per_res = out["plddt"]  # [N_res] in 0-100
            scores["plddt"] = float(plddt_per_res.mean().item()) / 100.0
        else:
            scores["plddt"] = 0.0

        # Predicted Aligned Error (mean, in Angstroms)
        if "predicted_aligned_error" in out:
            scores["pae_mean"] = float(out["predicted_aligned_error"].mean().item())
        else:
            scores["pae_mean"] = 0.0

        # Composite score
        scores["composite"] = scores["ptm"] * scores["plddt"]

        return scores


def score_proteina_structures_openfold(
    protein_id: str,
    reference_cif: str,
    inference_output_dir: str,
    chain: str = "A",
    recycles: int = 3,
    model_name: str = "model_1_ptm",
    verbose: bool = False,
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = False,
    use_cuequivariance_multiplicative_update: bool = True,
    scorer: Optional["OpenFoldAF2Rank"] = None,
    predicted_structure_dir: Optional[str] = None,
) -> List[Dict]:
    """Score all Proteina-generated structures using AF2Rank with OpenFold backend.

    Args:
        scorer: Optional pre-loaded OpenFoldAF2Rank instance.  When provided, the
            caller is responsible for calling scorer.reset_reference() before this
            function so that it uses the correct reference structure for protein_id.
            Model-loading kwargs (model_name, recycles, use_*) are ignored when a
            scorer is supplied.  Pass scorer= to avoid the per-protein model reload
            when scoring many proteins in a single process (sharded / batched use).
    """
    pdb_files = sorted(glob.glob(os.path.join(inference_output_dir, "*.pdb")))

    if not pdb_files:
        logger.warning(f"No PDB files found in {inference_output_dir}")
        return []

    logger.info(f"Starting OpenFold AF2Rank scoring for {len(pdb_files)} structures of {protein_id}")

    # Batch-reconstruct all CA-only structures via cg2all
    ca_only_pdbs = [p for p in pdb_files if _is_ca_only_pdb(p)]
    allatom_map = {}  # original_pdb -> reconstructed_pdb
    if ca_only_pdbs:
        logger.info(f"Reconstructing {len(ca_only_pdbs)} CA-only structures with cg2all...")
        reconstructor = _get_cg2all_reconstructor()
        allatom_map = reconstructor.reconstruct_batch(ca_only_pdbs)
        logger.info(f"Reconstructed {len(allatom_map)} structures")

    _owns_scorer = scorer is None
    if _owns_scorer:
        scorer = OpenFoldAF2Rank(
            reference_cif, chain=chain, model_name=model_name, recycles=recycles,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )

    # Pipeline featurization and GPU forward: while the GPU runs the model for
    # decoy i, a background thread pre-featurizes decoy i+1 (build_batch is
    # CPU-only, ~0.7–1.4 s; GPU forward is ~2–5 s, so overlap is nearly free).
    items = [(p, allatom_map.get(p, p)) for p in pdb_files]

    def _featurize_item(item):
        orig_pdb, scored_pdb = item
        batch, template_coords = scorer._featurize(
            scored_pdb, decoy_chain="A", _original_pdb=orig_pdb,
        )
        return orig_pdb, batch, template_coords

    scores = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_featurize_item, items[0])

        for i, pdb_path in enumerate(tqdm(pdb_files, desc=f"AF2Rank-OpenFold scoring {protein_id}")):
            pdb_filename = os.path.basename(pdb_path)
            output_pdb = None
            if predicted_structure_dir is not None:
                os.makedirs(predicted_structure_dir, exist_ok=True)
                output_pdb = os.path.join(predicted_structure_dir, pdb_filename)

            # Submit featurization for the next decoy before waiting for the
            # current one — this is the overlap that hides featurization latency.
            if i + 1 < len(items):
                next_future = executor.submit(_featurize_item, items[i + 1])

            try:
                _, batch, template_coords = future.result()

                with torch.no_grad():
                    out = scorer.model.model(batch)

                structure_scores = scorer._extract_scores(out)
                if output_pdb is not None:
                    _save_openfold_prediction_pdb(scorer.reference_sequence, out, output_pdb)
                    structure_scores["predicted_structure_path"] = output_pdb
                    structure_scores["predicted_structure_file"] = os.path.basename(output_pdb)

                # TM scores: reference vs template/prediction, template vs prediction
                structure_scores["tm_ref_template"] = tmscore(
                    scorer.reference_coords, template_coords, env=_USALIGN_PARALLEL_ENV
                ).get("tms", 0.0)
                if "final_atom_positions" in out:
                    pred_ca = out["final_atom_positions"][:, 1, :].detach().cpu().numpy()
                    structure_scores["tm_ref_pred"] = tmscore(
                        scorer.reference_coords, pred_ca, env=_USALIGN_PARALLEL_ENV
                    ).get("tms", 0.0)
                    structure_scores["tm_template_pred"] = tmscore(
                        template_coords, pred_ca, env=_USALIGN_PARALLEL_ENV
                    ).get("tms", 0.0)

                gc.collect()
                torch.cuda.empty_cache()

                structure_scores.update({
                    "protein_id": protein_id,
                    "structure_file": pdb_filename,
                    "structure_path": pdb_path,
                })
                scores.append(structure_scores)
            except Exception as e:
                logger.error(f"Failed to score {pdb_filename}: {e}")
                scores.append({
                    "protein_id": protein_id,
                    "structure_file": pdb_filename,
                    "structure_path": pdb_path,
                    "error": str(e),
                })

            if i + 1 < len(items):
                future = next_future

    # Clean up batch-reconstructed temp files
    for temp_path in allatom_map.values():
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    logger.info(f"Completed OpenFold AF2Rank scoring for {protein_id}")
    return scores


def run_af2rank_analysis_openfold(
    protein_id: str,
    reference_cif: str,
    inference_output_dir: str,
    output_dir: str,
    chain: str = "A",
    recycles: int = 3,
    model_name: str = "model_1_ptm",
    verbose: bool = False,
    regenerate_summary: bool = False,
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = False,
    use_cuequivariance_multiplicative_update: bool = True,
    scorer: Optional["OpenFoldAF2Rank"] = None,
) -> str:
    """Run complete AF2Rank analysis using OpenFold backend.

    Args:
        scorer: Optional pre-loaded OpenFoldAF2Rank instance.  Caller must call
            scorer.reset_reference(reference_cif, chain) before this function.
            When provided, model_name / recycles / use_* kwargs are ignored.
    """
    os.makedirs(output_dir, exist_ok=True)

    scores_csv_path = os.path.join(output_dir, f"af2rank_scores_{protein_id}.csv")

    if os.path.exists(scores_csv_path) and not regenerate_summary:
        logger.info(f"Loading existing scores for {protein_id}")
        scores = load_af2rank_scores_from_csv(scores_csv_path)
    else:
        logger.info(f"Generating new scores for {protein_id}")
        scores = score_proteina_structures_openfold(
            protein_id=protein_id,
            reference_cif=reference_cif,
            inference_output_dir=inference_output_dir,
            chain=chain,
            recycles=recycles,
            model_name=model_name,
            verbose=verbose,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
            scorer=scorer,
            predicted_structure_dir=os.path.join(output_dir, "predicted_structures"),
        )
        if not scores:
            logger.warning(f"No scores generated for {protein_id}")
            return None
        scores_csv_path = save_af2rank_scores(scores, output_dir, protein_id)

    summary = {
        "protein_id": protein_id,
        "total_structures": len(scores),
        "successful_scores": len([s for s in scores if "error" not in s]),
        "failed_scores": len([s for s in scores if "error" in s]),
        "reference_structure": reference_cif,
        "inference_directory": inference_output_dir,
        "output_directory": output_dir,
        "af2rank_directory": output_dir,
        "chain": chain,
        "recycles": recycles,
        "scores_csv": scores_csv_path,
        "status": "scored_metrics_only",
    }

    summary_path = os.path.join(output_dir, f"af2rank_summary_{protein_id}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return scores_csv_path


def run_af2rank_plot_only_openfold(
    protein_id: str,
    reference_cif: str,
    inference_output_dir: str,
    output_dir: str,
    chain: str = "A",
    recycles: int = 3,
    regenerate_summary: bool = True,
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
) -> str:
    """Generate only AF2Rank plots from existing CSV data (no scoring)."""
    scores_csv_path = os.path.join(output_dir, f"af2rank_scores_{protein_id}.csv")
    if not os.path.exists(scores_csv_path):
        logger.error(f"No existing scores found for {protein_id} at {scores_csv_path}")
        return None

    scores = load_af2rank_scores_from_csv(scores_csv_path)
    logger.info(f"Loaded {len(scores)} scores for {protein_id}")
    plot_af2rank_results(scores, output_dir, protein_id)

    if regenerate_summary:
        return run_af2rank_analysis_openfold(
            protein_id=protein_id,
            reference_cif=reference_cif,
            inference_output_dir=inference_output_dir,
            output_dir=output_dir,
            chain=chain,
            recycles=recycles,
            regenerate_summary=False,
            use_deepspeed_evoformer_attention=use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=use_cuequivariance_multiplicative_update,
        )

    return scores_csv_path


if __name__ == "__main__":
    print("AF2Rank OpenFold scorer module loaded successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
