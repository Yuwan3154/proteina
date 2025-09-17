#!/usr/bin/env python3
"""
AF2Rank Scorer Module

Contains the ModernAF2Rank class and related functions for protein structure scoring
using AlphaFold2 via ColabDesign. This module provides a modern implementation
following the predict.ipynb approach.

Key Features:
- Modern AF2Rank implementation using ColabDesign
- Batch processing of multiple structures
- Comprehensive scoring with AF2 confidence metrics
- Visualization and analysis functions
- Proper memory management and cleanup
"""

import os
import sys
import glob
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

# Try to import JAX/ColabDesign components
try:
    import jax
    import jax.numpy as jnp
    from colabdesign import mk_af_model, clear_mem
    from colabdesign.af.contrib import predict
    from colabdesign.shared.protein import _np_rmsd
    from colabdesign.shared.utils import copy_dict
    from colabdesign.af.prep import prep_pdb
    from colabdesign.af.alphafold.common import residue_constants
    JAX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"JAX/ColabDesign not available: {e}")
    JAX_AVAILABLE = False


def get_pdb_fn(pdb_code):
    """Simple PDB getter function following predict.ipynb approach."""
    # If it's a file path that exists, return it
    if os.path.isfile(pdb_code):
        return pdb_code
    # Otherwise assume it's a PDB code and try to download
    elif len(pdb_code) == 4:
        os.makedirs("tmp", exist_ok=True)
        os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_code}.cif -P tmp/")
        return f"tmp/{pdb_code}.cif"
    else:
        # For AF database
        os.makedirs("tmp", exist_ok=True)
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb -P tmp/")
        return f"tmp/AF-{pdb_code}-F1-model_v4.pdb"


def get_available_models(pdb_file):
    """Get list of available model numbers in a PDB file."""
    models = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                model_str = line[5:].strip()
                try:
                    model_num = int(model_str)
                    if model_num not in models:
                        models.append(model_num)
                except ValueError:
                    continue
    # If no MODEL records found, assume it's a single model (model 1)
    if not models:
        models = [1]
    return sorted(models)


def robust_get_template_feats(pdbs, chains, query_seq, **kwargs):
    """Wrapper around predict.get_template_feats that handles model number issues."""
    
    # Import and store original functions
    from colabdesign.shared import protein
    from colabdesign.af import prep
    
    # Store original functions
    original_protein_pdb_to_string = protein.pdb_to_string
    original_prep_pdb_to_string = prep.pdb_to_string
    
    def robust_pdb_to_string(pdb_file, chains=None, models=None, auth_chains=True):
        """Robust version of pdb_to_string that handles different model numbers."""
        
        # If models is not specified, use the original function
        if models is None:
            return original_protein_pdb_to_string(pdb_file, chains=chains, models=models, auth_chains=auth_chains)
        
        # Try with the specified models first
        result = original_protein_pdb_to_string(pdb_file, chains=chains, models=models, auth_chains=auth_chains)
        if len(result.strip()) > 0:
            return result
        
        # If that failed or returned empty, detect available models
        available_models = get_available_models(pdb_file)
        
        # Try each available model
        for model_num in available_models:
            result = original_protein_pdb_to_string(pdb_file, chains=chains, models=[model_num], auth_chains=auth_chains)
            if len(result.strip()) > 0:
                return result
        
        # If all models failed, fall back to no model specification
        return original_protein_pdb_to_string(pdb_file, chains=chains, models=None, auth_chains=auth_chains)
    
    # Temporarily replace with our robust version
    protein.pdb_to_string = robust_pdb_to_string
    prep.pdb_to_string = robust_pdb_to_string
    
    # Call the function
    result = predict.get_template_feats(
        pdbs=pdbs, chains=chains, query_seq=query_seq,
        get_pdb_fn=get_pdb_fn, **kwargs
    )
    
    # Restore original functions
    protein.pdb_to_string = original_protein_pdb_to_string
    prep.pdb_to_string = original_prep_pdb_to_string
    
    return result


def run_do_not_align(query_sequence, target_sequence, **kwargs):
    """No alignment function - just return sequences as-is."""
    return [query_sequence, target_sequence], [0, 0]


def get_sequence_from_pdb(pdb_file, chain=None):
    """Get sequence from PDB using predict.get_template_feats approach."""
    # If no chain specified, detect the first chain or default to "A"
    if chain is None:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    chain = line[21]
                    break
        if chain is None:
            chain = "A"  # Default fallback
    
    # Create a dummy query sequence of similar length for alignment
    with open(pdb_file, 'r') as f:
        atom_count = sum(1 for line in f 
                       if line.startswith('ATOM') and line[12:16].strip() == 'CA'
                       and line[21] == chain)
    
    dummy_query = "A" * max(50, atom_count)  # Dummy sequence for extraction
    
    # Use robust get_template_feats to properly parse the PDB
    batch = robust_get_template_feats(
        pdbs=[pdb_file],
        chains=[chain],  # Always pass a valid chain identifier
        query_seq=dummy_query,
        copies=1,
        use_seq=True,  # We want the sequence
        align_fn=run_do_not_align
    )
    
    # Extract sequence from aatype
    sequence = "".join([predict.residue_constants.restypes[aa] if aa < 20 else "X" 
                       for aa in batch["aatype"]])
    return sequence


def tmscore(x, y, tmscore_exe="USalign"):
    """Calculate TMscore between two structures using temporary files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f2:
        
        # Save structures to temporary PDB files
        for f, z in zip([f1, f2], [x, y]):
            for k, c in enumerate(z):
                f.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" 
                          % (k+1, "CA", "ALA", "A", k+1, c[0], c[1], c[2], 1, 0))
            f.flush()
    
    # Run TMscore
    for exe in [tmscore_exe, "./TMscore", "TMscore", "/usr/local/bin/TMscore"]:
        if os.path.exists(exe) or os.system(f"which {exe} > /dev/null 2>&1") == 0:
            cmd = f'{exe} {f1.name} {f2.name}'
            if exe == "USalign":
                cmd += " -TMscore 1"
            output = os.popen(cmd).readlines()
            break
    else:
        print("Warning: TMscore executable not found. Using RMSD calculation.")
        rmsd = _np_rmsd(x, y, use_jax=False)
        return {"rms": rmsd, "tms": 0.0, "gdt": 0.0}
    
    # Parse outputs
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    o = {"rms": 0.0, "tms": 0.0, "gdt": 0.0}
        
    for line in output:
        line = line.rstrip()
        if line.startswith("RMSD"): 
            o["rms"] = parse_float(line)
        elif line.startswith("TM-score"): 
            o["tms"] = parse_float(line)
        elif line.startswith("GDT-TS-score"): 
            o["gdt"] = parse_float(line)
    
    # Clean up temporary files
    for temp_file in [f1.name, f2.name]:
        os.unlink(temp_file)
    
    return o


class ModernAF2Rank:
    """Modern AF2Rank implementation following predict.ipynb approach exactly."""
    
    def __init__(self, reference_pdb, chain=None, model_type="auto", debug=False):
        """Initialize AF2Rank with reference structure."""
        if not JAX_AVAILABLE:
            raise ImportError("JAX/ColabDesign not available. Please ensure colabdesign environment is activated.")
        
        self.reference_pdb = reference_pdb
        if chain is None:
            self.chain = "A"
        else:
            self.chain = chain
        self.debug = debug
        self.model_type = model_type
        
        # Get reference sequence using proper ColabDesign approach
        self.reference_sequence = get_sequence_from_pdb(reference_pdb, chain)
        self.reference_lengths = [len(self.reference_sequence)]
        
        # Determine model type
        if model_type == "auto":
            use_multimer = False  # AF2Rank typically uses monomer model
        elif model_type == "multimer":
            use_multimer = True
        else:
            use_multimer = False
            
        # Model options following predict.ipynb exactly
        self.model_opts = {
            "num_msa": 1,  # Single sequence for AF2Rank
            "num_extra_msa": 1,
            "num_templates": 1,
            "use_cluster_profile": False,
            "use_multimer": use_multimer,
            "pseudo_multimer": False,
            "use_templates": True,
            "use_batch_as_template": False,
            "use_dgram": True,
            "protocol": "hallucination",
            "best_metric": "plddt",
            "optimize_seq": False,
            "debug": debug,
            "clear_prev": True  # CRITICAL: Clear previous state to prevent accumulation
        }
        
        # Store reference coordinates for TMscore calculations (persistent across reinits)
        self.reference_coords = self._get_coords_from_pdb(reference_pdb, chain)
        
        # Initialize the model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the AlphaFold model."""
        logger.info("Loading AlphaFold parameters for AF2Rank scoring...")
        
        # Set environment variables for AlphaFold parameters
        import os
        params_path = os.path.expanduser("~/params")
        os.environ["ALPHAFOLD_DATA_DIR"] = params_path
        
        # Create model with explicit parameter directory
        model_opts = self.model_opts.copy()
        model_opts['data_dir'] = params_path
        
        # Initialize fresh model
        self.af = mk_af_model(use_mlm=False, **model_opts)
        
        # Prepare inputs exactly like predict.ipynb
        self.af.prep_inputs(self.reference_lengths, copies=1, seed=0)
        
        # Set single sequence MSA (AF2Rank protocol)
        self._setup_single_sequence_msa()
        
        logger.info("✓ AF2Rank model initialized")
        
    def _setup_single_sequence_msa(self):
        """Setup single sequence MSA following predict.ipynb approach."""
        sequence = self.reference_sequence
        # Create MSA with just the reference sequence (no evolutionary information)
        msa = np.array([[predict.residue_constants.restype_order.get(aa, 20) for aa in sequence]])
        deletion_matrix = np.zeros_like(msa)
        # Set MSA exactly like predict.ipynb
        self.af.set_msa(msa, deletion_matrix)
        
    def _get_coords_from_pdb(self, pdb_file, chain=None):
        """Extract CA coordinates from PDB file using prep_pdb."""
        # Auto-detect chain if not specified
        if chain is None:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        chain = line[21]
                        break
        if chain is None: 
                chain = 'A'  # Default fallback
        
        # Apply our robust PDB handling
        from colabdesign.shared import protein
        from colabdesign.af import prep
        
        # Store original functions
        original_protein_pdb_to_string = protein.pdb_to_string
        original_prep_pdb_to_string = prep.pdb_to_string
        
        def robust_pdb_to_string(pdb_file, chains=None, models=None, auth_chains=True):
            """Robust version of pdb_to_string that handles different model numbers."""
            
            # If models is not specified, use the original function
            if models is None:
                return original_protein_pdb_to_string(pdb_file, chains=chains, models=models, auth_chains=auth_chains)
            
            # Try with the specified models first
            result = original_protein_pdb_to_string(pdb_file, chains=chains, models=models, auth_chains=auth_chains)
            if len(result.strip()) > 0:
                return result
            
            # If that failed or returned empty, detect available models
            available_models = get_available_models(pdb_file)
            
            # Try each available model
            for model_num in available_models:
                result = original_protein_pdb_to_string(pdb_file, chains=chains, models=[model_num], auth_chains=auth_chains)
                if len(result.strip()) > 0:
                    return result
            
            # If all models failed, fall back to no model specification
            return original_protein_pdb_to_string(pdb_file, chains=chains, models=None, auth_chains=auth_chains)
        
        # Temporarily replace with our robust version
        protein.pdb_to_string = robust_pdb_to_string
        prep.pdb_to_string = robust_pdb_to_string
        
        # Use prep_pdb to extract PDB information  
        info = prep_pdb(pdb_file, chain=chain, ignore_missing=True)
        
        # Restore original functions
        protein.pdb_to_string = original_protein_pdb_to_string
        prep.pdb_to_string = original_prep_pdb_to_string
        
        # Extract CA coordinates from all_atom_positions
        ca_idx = residue_constants.atom_order["CA"]
        coords = info["batch"]["all_atom_positions"][:, ca_idx, :]
        
        return coords
        
    def score_structure(self, decoy_pdb, decoy_chain=None, 
                rm_seq=True, rm_sc=True, rm_ic=False,
                      recycles=3, seed=0, 
                      output_pdb=None, verbose=False):
        """Score a single structure using AF2Rank protocol following predict.ipynb exactly."""
        
        if verbose:
            logger.debug(f"Starting AF2Rank scoring for {decoy_pdb}")
        
        # Set random seed for reproducibility
        self.af.set_seed(seed)
        
        # Handle chain specification properly 
        if decoy_chain is None:
            with open(decoy_pdb, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        decoy_chain = line[21]
                        break
            if decoy_chain is None:
                decoy_chain = "A"  # Default fallback
        
        # Create template batch using robust get_template_feats to handle model number issues
        if verbose:
            logger.debug(f"Creating template batch for {decoy_pdb}")
            
        batch = robust_get_template_feats(
            pdbs=[decoy_pdb],
            chains=[decoy_chain],  # Always pass a valid chain identifier
            query_seq=self.reference_sequence,
            query_a3m=None,  # No MSA for AF2Rank
            copies=1,
            propagate_to_copies=False,
            use_seq=not rm_seq,  # AF2Rank protocol: remove sequence
            align_fn=run_do_not_align  # No alignment needed
        )
        
        if verbose:
            logger.debug("Template batch created successfully")
        
        # Set template in model EXACTLY like predict.ipynb
        self.af.set_template(batch=batch, n=0)
        
        if verbose:
            logger.debug("Template set in model")
        
        # Set template options EXACTLY like predict.ipynb
        self.af.set_opt("template",
                       rm_sc=rm_sc,     # Remove sidechains
                       rm_seq=rm_seq)   # Remove sequence (key for AF2Rank)
        
        if verbose:
            logger.debug("Template options set")
        
        # Set recycles
        self.af.set_opt(num_recycles=recycles)
        
        if verbose:
            logger.debug(f"Recycles set to {recycles}")
        
        # Get template coordinates for TMscore analysis
        if verbose:
            logger.debug("Getting template coordinates")
        template_coords = self._get_coords_from_pdb(decoy_pdb, decoy_chain)
        
        # Run prediction exactly like predict.ipynb
        if verbose:
            logger.debug("Starting AF2 prediction...")
            
        self.af.predict(verbose=verbose)
        
        if verbose:
            logger.debug("AF2 prediction completed successfully")
        
        # Calculate scores using AF2 outputs including TMscore analysis
        if verbose:
            logger.debug("Calculating scores")
        scores = self._calculate_scores(template_coords=template_coords)
        
        # Save output PDB if requested
        if output_pdb:
            self.af.save_pdb(output_pdb)
        
        # Safe cleanup to prevent memory accumulation
        import gc
        gc.collect()
        
        if verbose:
            logger.debug(f"AF2Rank scores - pLDDT: {scores['plddt']:.3f}, pTM: {scores['ptm']:.3f}, Composite: {scores['composite']:.3f}")
                
        return scores
    
    def _calculate_scores(self, template_coords=None):
        """Calculate scoring metrics from AF2 outputs including TMscore analysis."""
        aux = self.af.aux
        scores = {}
        
        # Basic AF2 metrics from aux["log"]
        scores["plddt"] = float(aux["log"]["plddt"])
        scores["ptm"] = float(aux["log"]["ptm"])
        scores["pae_mean"] = float(aux["log"]["pae"]) * 31.0  # Convert to Angstroms
        
        # Get predicted coordinates (CA atoms)
        pred_coords = aux["atom_positions"][:, 1]
        
        # TMscore calculations for AF2Rank analysis
        if self.reference_coords is not None:
            # TMscore between reference and prediction (key metric)
            tm_ref_pred = tmscore(self.reference_coords, pred_coords)
            scores["tm_ref_pred"] = tm_ref_pred["tms"]
            scores["rmsd_ref_pred"] = tm_ref_pred["rms"]
            scores["gdt_ref_pred"] = tm_ref_pred.get("gdt", 0.0)
        
        if template_coords is not None:
            # TMscore between reference and template (input quality)
            if self.reference_coords is not None:
                tm_ref_template = tmscore(self.reference_coords, template_coords)
                scores["tm_ref_template"] = tm_ref_template["tms"]
                scores["rmsd_ref_template"] = tm_ref_template["rms"]
                scores["gdt_ref_template"] = tm_ref_template.get("gdt", 0.0)
            
            # TMscore between template and prediction (improvement)
            tm_template_pred = tmscore(template_coords, pred_coords)
            scores["tm_template_pred"] = tm_template_pred["tms"] 
            scores["rmsd_template_pred"] = tm_template_pred["rms"]
            scores["gdt_template_pred"] = tm_template_pred.get("gdt", 0.0)
        
        # AF2Rank composite score (key metric)
        scores["composite"] = scores["ptm"] * scores["plddt"]
        
        # Store coordinates for external analysis
        scores["pred_coords"] = pred_coords
        
        return scores


def score_proteina_structures(protein_id: str, reference_cif: str, inference_output_dir: str, 
                             chain: str = "A", recycles: int = 3, verbose: bool = False) -> List[Dict]:
    """
    Score all Proteina-generated structures using AF2Rank.
    
    Args:
        protein_id: Protein identifier (e.g., "1a2y_C")
        reference_cif: Path to reference CIF file
        inference_output_dir: Directory containing generated PDB files
        chain: Chain identifier for analysis
        recycles: Number of AF2 recycles
        verbose: Enable verbose output
        
    Returns:
        List of score dictionaries
    """
    if not JAX_AVAILABLE:
        logger.error("JAX/ColabDesign not available. Cannot perform AF2Rank scoring.")
        return []
    
    # Find all PDB files in the inference directory
    pdb_files = glob.glob(os.path.join(inference_output_dir, "*.pdb"))
    
    if not pdb_files:
        logger.warning(f"No PDB files found in {inference_output_dir}")
        return []
    
    logger.info(f"Starting AF2Rank scoring for {len(pdb_files)} structures of {protein_id}")
    
    # Initialize AF2Rank scorer
    try:
        scorer = ModernAF2Rank(reference_cif, chain=chain)
    except Exception as e:
        logger.error(f"Failed to initialize AF2Rank scorer: {e}")
        return []
    
    scores = []
    
    # Score each structure
    for pdb_path in tqdm(pdb_files, desc=f"AF2Rank scoring {protein_id}"):
        pdb_filename = os.path.basename(pdb_path)
        
        try:
            # Score the structure
            structure_scores = scorer.score_structure(
                pdb_path,
                decoy_chain="A",  # Force chain A since Proteina generates chain A
                recycles=recycles,
                verbose=verbose
            )
            
            # Add metadata
            structure_scores.update({
                "protein_id": protein_id,
                "structure_file": pdb_filename,
                "structure_path": pdb_path
            })
            
            # Remove coordinates from output (too large for CSV)
            if "pred_coords" in structure_scores:
                del structure_scores["pred_coords"]
            
            scores.append(structure_scores)
            
        except Exception as e:
            logger.error(f"Failed to score {pdb_filename}: {e}")
            scores.append({
                "protein_id": protein_id,
                "structure_file": pdb_filename,
                "structure_path": pdb_path,
                "error": str(e)
            })
    
    logger.info(f"Completed AF2Rank scoring for {protein_id}")
    return scores


def run_af2rank_analysis(protein_id: str, reference_cif: str, inference_output_dir: str,
                        output_dir: str, chain: str = "A", recycles: int = 3, 
                        verbose: bool = False) -> str:
    """
    Run complete AF2Rank analysis including scoring and visualization.
    
    Returns:
        Path to the generated scores CSV file
    """
    if not JAX_AVAILABLE:
        logger.error("JAX/ColabDesign not available. Cannot perform AF2Rank analysis.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Score structures
    scores = score_proteina_structures(
        protein_id=protein_id,
        reference_cif=reference_cif,
        inference_output_dir=inference_output_dir,
        chain=chain,
        recycles=recycles,
        verbose=verbose
    )
    
    if not scores:
        logger.warning(f"No scores generated for {protein_id}")
        return None
    
    # Save scores to CSV
    scores_csv_path = save_af2rank_scores(scores, output_dir, protein_id)
    
    # Generate plots
    plot_af2rank_results(scores, output_dir, protein_id)
    
    # Save summary
    summary = {
        "protein_id": protein_id,
        "total_structures": len(scores),
        "successful_scores": len([s for s in scores if "error" not in s]),
        "failed_scores": len([s for s in scores if "error" in s]),
        "reference_structure": reference_cif,
        "inference_directory": inference_output_dir,
        "output_directory": output_dir,
        "chain": chain,
        "recycles": recycles
    }
    
    summary_path = os.path.join(output_dir, f"af2rank_summary_{protein_id}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return scores_csv_path


def rescale(a, amin=None, amax=None):  
    """Rescale array values to [0,1] range."""
    a = np.copy(a)
    if amin is None: 
        amin = a.min()
    if amax is None: 
        amax = a.max()
    if amax == amin:  # Handle case where all values are the same
        return np.ones_like(a) * 0.5
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)


def plot_metric(scores, x="ptm", y="composite", 
                title=None, diag=False, scale_axis=True, 
                save_path=None, **kwargs):
    """Create a comprehensive AF2Rank analysis plot following af2rank_cli.py approach."""
    
    plt.figure(figsize=(8, 6), dpi=100)
    if title is not None: 
        plt.title(title)
        
    # Extract data, handling missing keys gracefully
    x_vals = np.array([k.get(x, 0.0) for k in scores if "error" not in k])
    y_vals = np.array([k.get(y, 0.0) for k in scores if "error" not in k])
    c = rescale(np.array([k.get("plddt", 0.7) for k in scores if "error" not in k]), 0.5, 0.9)
    
    if len(x_vals) == 0:
        plt.text(0.5, 0.5, 'No valid data to plot', transform=plt.gca().transAxes, 
                ha='center', va='center', fontsize=12)
        correlation = 0.0
    else:
        plt.scatter(x_vals, y_vals, c=c*0.75, s=20, vmin=0, vmax=1, 
                   cmap="gist_rainbow", **kwargs)
    
        if diag and len(x_vals) > 0:
            lims = [min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())]
            plt.plot(lims, lims, color="black", linestyle="--", alpha=0.5)
        
        # Calculate and display correlation if we have valid data
        valid_mask = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
        if valid_mask.sum() > 1:
            from scipy.stats import spearmanr
            correlation = spearmanr(x_vals[valid_mask], y_vals[valid_mask]).correlation
            if not np.isnan(correlation):
                plt.text(0.05, 0.95, f'Spearman R: {correlation:.3f}', 
                        transform=plt.gca().transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                correlation = 0.0
        else:
            correlation = 0.0
    
    # AF2Rank analysis labels
    labels = {
        "ptm": "Predicted TM-score (pTM)",
        "plddt": "Predicted LDDT (pLDDT)",
        "composite": "AF2Rank Score (pTM × pLDDT)",
        "pae_mean": "Predicted Aligned Error",
        "tm_ref_template": "TM-score (Reference vs Template)",
        "tm_ref_pred": "TM-score (Reference vs Prediction)",
        "tm_template_pred": "TM-score (Template vs Prediction)",
        "rmsd_ref_template": "RMSD (Reference vs Template)",
        "rmsd_ref_pred": "RMSD (Reference vs Prediction)",
        "rmsd_template_pred": "RMSD (Template vs Prediction)",
        "gdt_ref_template": "GDT-TS (Reference vs Template)",
        "gdt_ref_pred": "GDT-TS (Reference vs Prediction)",
        "gdt_template_pred": "GDT-TS (Template vs Prediction)"
    }

    plt.xlabel(labels.get(x, x.replace('_', ' ').title()))
    plt.ylabel(labels.get(y, y.replace('_', ' ').title()))
    
    if scale_axis and x in ["ptm", "plddt", "composite"] and y in ["ptm", "plddt", "composite"]:
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
    
    # Add colorbar if we have data
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
    """Generate comprehensive AF2Rank analysis plots following af2rank_cli.py approach."""
    
    # Generate plots focused on AF2Rank analysis (AF2 confidence vs structural quality)
    plot_configs = [
        {
            "x": "tm_ref_template", "y": "composite",
            "title": "AF2Rank Analysis: Template Quality vs AF2 Confidence",
            "filename": f"af2rank_{protein_id}_template_quality_vs_composite.png"
        },
        {
            "x": "tm_ref_pred", "y": "ptm",
            "title": "AF2Rank Analysis: Prediction Quality vs pTM",
            "filename": f"af2rank_{protein_id}_prediction_quality_vs_ptm.png"
        },
        {
            "x": "tm_ref_template", "y": "tm_ref_pred", "diag": True,
            "title": "AF2Rank Analysis: Template vs Prediction Quality",
            "filename": f"af2rank_{protein_id}_template_vs_prediction_quality.png"
        },
        {
            "x": "composite", "y": "tm_ref_pred",
            "title": "AF2Rank Analysis: AF2 Composite Score vs True Quality",
            "filename": f"af2rank_{protein_id}_composite_vs_true_quality.png"
        },
        {
            "x": "ptm", "y": "composite",
            "title": "AF2Rank Analysis: pTM vs Composite Score",
            "filename": f"af2rank_{protein_id}_ptm_vs_composite.png"
        },
        {
            "x": "plddt", "y": "composite",
            "title": "AF2Rank Analysis: pLDDT vs Composite Score", 
            "filename": f"af2rank_{protein_id}_plddt_vs_composite.png"
        },
        {
            "x": "ptm", "y": "plddt",
            "title": "AF2 Internal: pTM vs pLDDT correlation",
            "filename": f"af2rank_{protein_id}_ptm_vs_plddt.png"
        },
        {
            "x": "pae_mean", "y": "composite",
            "title": "AF2Rank Analysis: PAE vs Composite Score",
            "filename": f"af2rank_{protein_id}_pae_mean_vs_composite.png"
        }
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
    
    # Create DataFrame
    df = pd.DataFrame(scores)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"af2rank_scores_{protein_id}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(scores)} AF2Rank scores to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    # Test functionality
    if not JAX_AVAILABLE:
        print("JAX/ColabDesign not available. Cannot run AF2Rank scorer.")
        sys.exit(1)
    
    print("AF2Rank scorer module loaded successfully!")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
