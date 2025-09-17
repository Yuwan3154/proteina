#!/usr/bin/env python3
"""
AF2Rank Command Line Interface

A command line tool for ranking protein structural decoys using AlphaFold2.
Based on the AF2Rank protocol from Roney & Ovchinnikov (2022).
Updated to use the modern predict.ipynb approach from ColabDesign.

Features:
- Automatic resume from existing results (af2rank_scores.csv)
- Progress tracking and intermediate saves (every 5 structures by default)
- Memory monitoring and conservative cleanup to prevent memory issues
- Timeout detection to identify stuck processes
- Comprehensive AF2Rank analysis with TMscore calculations
- Support for file lists and batch processing
- Enhanced debugging for troubleshooting freezing issues
- Safe memory management without destroying JAX computational state

Usage:
    # Activate the conda environment first
    conda activate colabdesign
    
    # Basic usage (all PDB files in directory)
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results
    
    # Using a file list to specify which decoys to analyze
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results --file_list sampled_files.txt
    
    # With specific chain and AF2Rank settings  
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results --chain A --recycles 3 --verbose --save_pdbs
    
    # For multimers
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results --model_type multimer --file_list my_decoys.txt
    
    # Resume from existing results (default behavior)
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results
    
    # Force fresh start (ignore existing results)
    python af2rank_cli.py --decoy_dir ./decoys --reference_file native.pdb --output_dir ./results --no_resume

Requirements:
    - ColabDesign conda environment activated
    - TMscore executable in PATH or current directory
    - AlphaFold2 parameters
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import argparse
import sys
import warnings
import time
import tempfile
import re
import gc
import signal
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import jax
import jax.numpy as jnp
from tqdm import tqdm
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign import mk_af_model, clear_mem
from colabdesign.af.contrib import predict
from colabdesign.shared.protein import _np_rmsd
from colabdesign.shared.utils import copy_dict


def get_memory_usage_gb():
    """Get current memory usage in GB using system info."""
    # Get memory info from /proc/meminfo (Linux)
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
    
    # Extract memory values
    mem_total = None
    mem_available = None
    
    for line in meminfo.split('\n'):
        if line.startswith('MemTotal:'):
            mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
        elif line.startswith('MemAvailable:'):
            mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
    
    if mem_total and mem_available:
        used_gb = (mem_total - mem_available) / (1024**3)
        total_gb = mem_total / (1024**3)
        return used_gb, total_gb
    else:
        return None, None

def get_gpu_memory_usage_gb():
    """Get current GPU memory usage in GB using nvidia-smi."""
    # Query both used and total memory in one command
    output = os.popen("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits").readlines()
    if output:
        # Parse the first GPU's memory info
        used_mb, total_mb = output[0].strip().split(', ')
        used_gb = float(used_mb) / 1024
        total_gb = float(total_mb) / 1024
        return used_gb, total_gb
    else:
        return None, None

def log_memory_and_status(step_name):
    """Log memory usage and processing status."""
    # Log CPU memory usage
    used_gb, total_gb = get_memory_usage_gb()
    
    if used_gb is not None:
        print(f"[MEMORY] {step_name}: {used_gb:.2f}GB / {total_gb:.2f}GB used", flush=True)
        
        # Warning if memory usage is high
        if used_gb / total_gb > 0.85:
            print(f"[WARNING] High memory usage detected: {used_gb/total_gb*100:.1f}%", flush=True)
    else:
        print(f"[STATUS] {step_name}", flush=True)
    
    # Log GPU memory usage
    gpu_used_gb, gpu_total_gb = get_gpu_memory_usage_gb()
    if gpu_used_gb is not None:
        print(f"[GPU MEMORY] {step_name}: {gpu_used_gb:.2f}GB / {gpu_total_gb:.2f}GB used", flush=True)
    else:
        print(f"[GPU MEMORY] {step_name}: nvidia-smi not available", flush=True)


def safe_cleanup():
    """Safe garbage collection without destroying JAX state."""
    gc.collect()


def aggressive_cleanup():
    """More aggressive cleanup - use sparingly."""
    gc.collect()
    clear_mem()


class TimeoutHandler:
    """Timeout handler to detect if processing gets stuck."""
    
    def __init__(self, timeout_seconds=300):  # 5 minute timeout
        self.timeout_seconds = timeout_seconds
        self.last_update_time = time.time()
        self.current_file = "unknown"
    
    def update_progress(self, current_file):
        """Update progress to reset timeout."""
        self.last_update_time = time.time()
        self.current_file = current_file
    
    def check_timeout(self):
        """Check if we've exceeded the timeout."""
        elapsed = time.time() - self.last_update_time
        if elapsed > self.timeout_seconds:
            print(f"[WARNING] Timeout detected: {elapsed:.1f}s since last update", flush=True)
            print(f"[WARNING] Last file being processed: {self.current_file}", flush=True)
            log_memory_and_status("Timeout detected")
            return True
        return False


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
    # Try different common locations for TMscore
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


def get_available_models(pdb_file):
    """Get list of available model numbers in a PDB file."""
    models = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                # Extract model number, handling different formatting
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


class ModernAF2Rank:
    """Modern AF2Rank implementation following predict.ipynb approach exactly."""
    
    def __init__(self, reference_pdb, chain=None, model_type="auto", debug=False):
        """Initialize AF2Rank with reference structure."""
        self.reference_pdb = reference_pdb
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
        print("Loading AlphaFold parameters...")
        
        # Initialize fresh model
        self.af = mk_af_model(use_mlm=False, **self.model_opts)
        
        # Prepare inputs exactly like predict.ipynb
        self.af.prep_inputs(self.reference_lengths, copies=1, seed=0)
        
        # Set single sequence MSA (AF2Rank protocol)
        self._setup_single_sequence_msa()
        
        print(f"✓ Model initialized")
        
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
        from colabdesign.af.prep import prep_pdb
        from colabdesign.af.alphafold.common import residue_constants
        
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
        
    def rank_structure(self, decoy_pdb, decoy_chain=None, 
                rm_seq=True, rm_sc=True, rm_ic=False,
                      recycles=3, seed=0, 
                      output_pdb=None, verbose=False):
        """Rank a single structure using AF2Rank protocol following predict.ipynb exactly."""
        
        if verbose:
            print(f"[DEBUG] Starting rank_structure for {decoy_pdb}")
        
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
            print(f"[DEBUG] Creating template batch for {decoy_pdb}")
            
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
            print(f"[DEBUG] Template batch created successfully")
        
        # Set template in model EXACTLY like predict.ipynb
        self.af.set_template(batch=batch, n=0)
        
        if verbose:
            print(f"[DEBUG] Template set in model")
        
        # Set template options EXACTLY like predict.ipynb
        self.af.set_opt("template",
                       rm_sc=rm_sc,     # Remove sidechains
                       rm_seq=rm_seq)   # Remove sequence (key for AF2Rank)
        
        if verbose:
            print(f"[DEBUG] Template options set")
        
        # Set recycles
        self.af.set_opt(num_recycles=recycles)
        
        if verbose:
            print(f"[DEBUG] Recycles set to {recycles}")
        
        # Get template coordinates for TMscore analysis
        if verbose:
            print(f"[DEBUG] Getting template coordinates")
        template_coords = self._get_coords_from_pdb(decoy_pdb, decoy_chain)
        
        # Run prediction exactly like predict.ipynb
        if verbose:
            print(f"[DEBUG] Starting AF2 prediction...")
            
        self.af.predict(verbose=verbose)
        
        if verbose:
            print(f"[DEBUG] AF2 prediction completed successfully")
        
        # Calculate scores using AF2 outputs including TMscore analysis
        if verbose:
            print(f"[DEBUG] Calculating scores")
        scores = self._calculate_scores(template_coords=template_coords)
        
        # Save output PDB if requested
        if output_pdb:
            self.af.save_pdb(output_pdb)
        
        # Safe cleanup to prevent memory accumulation
        safe_cleanup()
        
        if verbose:
            print(f"pLDDT: {scores['plddt']:.3f}, pTM: {scores['ptm']:.3f}, "
                  f"Composite: {scores['composite']:.3f}")
            if 'tm_ref_pred' in scores:
                print(f"TM-score (ref-pred): {scores['tm_ref_pred']:.3f}")
            if 'tm_ref_template' in scores:
                print(f"TM-score (ref-template): {scores['tm_ref_template']:.3f}")
                
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


def plot_and_save(scores, output_dir, name_prefix="af2rank"):
    """Generate and save visualization plots."""
    
    def rescale(a, amin=None, amax=None):  
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
    
    def plot_me(scores, x="ptm", y="composite", 
                title=None, diag=False, scale_axis=True, 
                save_path=None, **kwargs):
        
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
            print(f"Saved plot: {save_path}")
        
        plt.close()
        return correlation
    
    # Generate plots focused on AF2Rank analysis (AF2 confidence vs structural quality)
    plots = [
        {
            "x": "tm_ref_template", "y": "composite",
            "title": "AF2Rank Analysis: Template Quality vs AF2 Confidence",
            "filename": f"{name_prefix}_template_quality_vs_composite.png"
        },
        {
            "x": "tm_ref_pred", "y": "ptm",
            "title": "AF2Rank Analysis: Prediction Quality vs pTM",
            "filename": f"{name_prefix}_prediction_quality_vs_ptm.png"
        },
        {
            "x": "tm_ref_template", "y": "tm_ref_pred", "diag": True,
            "title": "AF2Rank Analysis: Template vs Prediction Quality",
            "filename": f"{name_prefix}_template_vs_prediction_quality.png"
        },
        {
            "x": "composite", "y": "tm_ref_pred",
            "title": "AF2Rank Analysis: AF2 Composite Score vs True Quality",
            "filename": f"{name_prefix}_composite_vs_true_quality.png"
        },
        {
            "x": "ptm", "y": "plddt",
            "title": "AF2 Internal: pTM vs pLDDT correlation",
            "filename": f"{name_prefix}_ptm_vs_plddt.png"
        }
    ]
    
    correlations = {}
    for plot_config in plots:
        filename = plot_config.pop("filename")
        save_path = os.path.join(output_dir, filename)
        correlation = plot_me(scores, save_path=save_path, **plot_config)
        correlations[filename] = correlation
    
    return correlations


def save_scores_to_csv(scores, output_path):
    """Save scores to CSV file."""
    df = pd.DataFrame(scores)
    df.to_csv(output_path, index=False)
    print(f"[SAVE] Saved {len(scores)} scores to: {output_path}", flush=True)


def load_existing_scores(csv_path):
    """Load existing scores from CSV file if it exists."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Found existing results file: {csv_path}")
        print(f"Loaded {len(df)} existing scores")
        
        # Convert DataFrame back to list of dictionaries
        scores = df.to_dict('records')
        
        # Extract already processed file IDs
        processed_ids = set()
        for score in scores:
            if 'id' in score and score.get('structure_type') == 'decoy':
                processed_ids.add(score['id'])
        
        print(f"Found {len(processed_ids)} already processed decoy structures")
        return scores, processed_ids
    else:
        return [], set()


def filter_unprocessed_files(decoy_files, processed_ids):
    """Filter out already processed files."""
    unprocessed = [f for f in decoy_files if f not in processed_ids]
    skipped = len(decoy_files) - len(unprocessed)
    
    if skipped > 0:
        print(f"Skipping {skipped} already processed files")
        print(f"Remaining files to process: {len(unprocessed)}")
    
    return unprocessed


def main():
    parser = argparse.ArgumentParser(
        description="AF2Rank: Rank protein structural decoys using AlphaFold2 (Modern ColabDesign)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--decoy_dir", required=True, 
                       help="Directory containing decoy PDB files")
    parser.add_argument("--reference_file", required=True,
                       help="Path to reference/native PDB file")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--chain", default=None,
                       help="Chain to analyze (e.g., 'A')")
    parser.add_argument("--file_list", default=None,
                       help="Text file containing list of PDB files to use from decoy_dir (one per line)")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from existing results if available (default: True)")
    parser.add_argument("--no_resume", action="store_true",
                       help="Disable resume from existing results (start fresh)")
    
    # Model settings (simplified for AF2Rank)
    parser.add_argument("--model_type", choices=["auto", "monomer", "multimer"], 
                       default="auto", help="AlphaFold model type")
    parser.add_argument("--recycles", type=int, default=3,
                       help="Number of recycles (AF2Rank paper uses 3)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility")
    
    # AF2Rank protocol settings
    parser.add_argument("--mask_sequence", action="store_true", default=True,
                       help="Mask template sequence (key for AF2Rank)")
    parser.add_argument("--mask_sidechains", action="store_true", default=True,
                       help="Mask template sidechains")
    parser.add_argument("--mask_interchain", action="store_false", default=False,
                       help="Mask interchain contacts")
    
    # Output settings
    parser.add_argument("--save_pdbs", action="store_true",
                       help="Save output PDB structures")
    parser.add_argument("--save_interval", type=int, default=5,
                       help="Save CSV results every N structures (reduced to 5 for better progress tracking)")
    parser.add_argument("--cleanup_interval", type=int, default=100,
                       help="Safe memory cleanup every N structures (default: 100)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # AF2Rank settings
    af2rank_settings = {
        "rm_seq": args.mask_sequence,
        "rm_sc": args.mask_sidechains,
        "rm_ic": args.mask_interchain,
        "recycles": args.recycles,
        "seed": args.seed
    }
    
    print("AF2Rank: Protein Structure Ranking using AlphaFold2")
    print("Updated to use modern ColabDesign predict.ipynb approach")
    print("=" * 60)
    print(f"Reference structure: {args.reference_file}")
    print(f"Decoy directory: {args.decoy_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Chain: {args.chain}")
    print(f"Recycles: {args.recycles}")
    print(f"AF2Rank settings: {af2rank_settings}")
    print("=" * 60)
    
    # Verify inputs
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file not found: {args.reference_file}")
        sys.exit(1)
    
    if not os.path.exists(args.decoy_dir):
        print(f"Error: Decoy directory not found: {args.decoy_dir}")
        sys.exit(1)
    
    # Get list of decoy files
    if args.file_list:
        # Use specified file list
        if not os.path.exists(args.file_list):
            print(f"Error: File list not found: {args.file_list}")
            sys.exit(1)
        
        with open(args.file_list, 'r') as f:
            decoy_files = [line.strip() for line in f if line.strip()]
        
        # Verify files exist in decoy directory
        valid_decoy_files = []
        for f in decoy_files:
            full_path = os.path.join(args.decoy_dir, f)
            if os.path.exists(full_path):
                valid_decoy_files.append(f)
            else:
                print(f"Warning: File {f} not found in {args.decoy_dir}")
        decoy_files = valid_decoy_files
        
        print(f"Using file list: {args.file_list}")
        print(f"Found {len(decoy_files)} valid files from list")
        
    else:
        # Use all PDB files in directory
        decoy_files = [f for f in os.listdir(args.decoy_dir) 
                       if f.endswith(('.pdb', '.PDB'))]
        print(f"Using all PDB files in directory")
    
    if not decoy_files:
        print(f"Error: No PDB files found")
        if args.file_list:
            print(f"Check that files in {args.file_list} exist in {args.decoy_dir}")
        sys.exit(1)
    
    print(f"Found {len(decoy_files)} decoy structures")
    
    # Initialize Modern AF2Rank
    print("Initializing Modern AF2Rank model...")
    
    af2rank = ModernAF2Rank(
        args.reference_file, 
        chain=args.chain,
        model_type=args.model_type,
        debug=args.debug
    )
    
    # Create output PDB directory if needed
    if args.save_pdbs:
        pdb_output_dir = os.path.join(args.output_dir, "output_pdbs")
        os.makedirs(pdb_output_dir, exist_ok=True)
    
    # Check for existing results and resume if possible
    final_csv_path = os.path.join(args.output_dir, "af2rank_scores.csv")
    
    # Check if resume is disabled
    if args.no_resume:
        print("Resume disabled - starting fresh analysis")
        scores = []
        processed_ids = set()
        unprocessed_files = decoy_files
    else:
        # Try to resume from existing results
        scores, processed_ids = load_existing_scores(final_csv_path)
        
        if len(processed_ids) > 0:
            print(f"Resume mode: Found {len(processed_ids)} already processed files")
        
        # Filter out already processed files
        unprocessed_files = filter_unprocessed_files(decoy_files, processed_ids)
    
    if len(unprocessed_files) == 0:
        print("All files have already been processed!")
        print("Results are available in:", final_csv_path)
        
        # Generate plots and summary from existing results
        print("Generating visualizations from existing results...")
        correlations = plot_and_save(scores, args.output_dir, "af2rank")
        
        # Save summary
        summary = {
            "total_structures": len(scores),
            "successful_predictions": len([s for s in scores if "error" not in s]),
            "failed_predictions": len([s for s in scores if "error" in s]),
            "reference_file": args.reference_file,
            "decoy_directory": args.decoy_dir,
            "model_type": args.model_type,
            "af2rank_settings": af2rank_settings,
            "correlations": correlations,
            "runtime_minutes": 0,  # No new processing
            "status": "resumed_from_existing_results"
        }
        
        summary_path = os.path.join(args.output_dir, "af2rank_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nAF2Rank analysis complete (resumed from existing results)!")
        print(f"Results: {final_csv_path}")
        print(f"Summary: {summary_path}")
        return
    
    # If we have existing scores but need to process more files, 
    # we need to ensure the reference structure is included
    has_reference = any(s.get('id') == 'reference' for s in scores)
    
    start_time = time.time()
    
    # Score reference structure first (self-reference) if not already present
    if not has_reference:
        print("Scoring reference structure...")
        reference_score = af2rank.rank_structure(
            args.reference_file,
            decoy_chain=args.chain,
            **af2rank_settings,
        verbose=args.verbose
    )
        reference_score.update({"id": "reference", "structure_type": "reference"})
        scores.append(reference_score)
        print("✓ Reference structure scored successfully")
    else:
        print("Reference structure already scored, skipping...")
    
    # Score remaining decoy structures with progress bar
    print(f"Scoring {len(unprocessed_files)} remaining decoy structures...")
    
    # Initialize timeout handler
    timeout_handler = TimeoutHandler(timeout_seconds=600)  # 10 minute timeout per structure
    
    with tqdm(total=len(unprocessed_files), desc="Processing decoys") as pbar:
        for i, decoy_file in enumerate(unprocessed_files):
            # Update timeout handler
            timeout_handler.update_progress(decoy_file)
            input_pdb = os.path.join(args.decoy_dir, decoy_file)
            
            if args.save_pdbs:
                output_pdb = os.path.join(pdb_output_dir, f"af2rank_{decoy_file}")
            else:
                output_pdb = None
            
            # Process structure 
            score = af2rank.rank_structure(
                input_pdb,
                decoy_chain=args.chain,
                output_pdb=output_pdb,
                **af2rank_settings,
                verbose=args.verbose
            )
            score.update({"id": decoy_file, "structure_type": "decoy"})
            scores.append(score)
            
            # Save intermediate results
            if (i + 1) % args.save_interval == 0:
                save_scores_to_csv(scores, final_csv_path)
            
            # Safe cleanup to prevent memory buildup
            if (i + 1) % args.cleanup_interval == 0:
                safe_cleanup()
                
            # Log memory status every 100 structures and check for timeout
            if (i + 1) % args.save_interval == 0:
                if timeout_handler.check_timeout():
                    print("[ERROR] Processing appears to be stuck. Saving current progress and exiting.")
                    break
            
            pbar.update(1)
            pbar.set_postfix({
                'Current': decoy_file[:20],
                'Composite': f"{score.get('composite', 0):.3f}",
            })
    
    # Save final results
    print("\nSaving final results...")
    
    # Save scores to CSV (overwrite existing file with complete results)
    save_scores_to_csv(scores, final_csv_path)
    
    # Generate and save visualizations
    print("Generating visualizations...")
    correlations = plot_and_save(scores, args.output_dir, "af2rank")
    
    # Save summary statistics
    successful_scores = [s for s in scores if "error" not in s]
    summary = {
        "total_structures": len(scores),
        "successful_predictions": len(successful_scores),
        "failed_predictions": len(scores) - len(successful_scores),
        "reference_file": args.reference_file,
        "decoy_directory": args.decoy_dir,
        "model_type": args.model_type,
        "af2rank_settings": af2rank_settings,
        "correlations": correlations,
        "runtime_minutes": (time.time() - start_time) / 60,
        "status": "completed",
        "resumed_from_existing": len(processed_ids) > 0,
        "newly_processed": len(unprocessed_files),
        "top_ranked_decoys": sorted(
            [s for s in successful_scores if s.get('structure_type') == 'decoy'], 
            key=lambda x: x['composite'], reverse=True
        )[:10]
    }
    
    summary_path = os.path.join(args.output_dir, "af2rank_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nAF2Rank analysis complete!")
    if summary.get('resumed_from_existing'):
        print(f"Resumed from existing results: {len(processed_ids)} files already processed")
        print(f"Newly processed: {summary['newly_processed']} files")
    print(f"Runtime: {summary['runtime_minutes']:.1f} minutes")
    print(f"Successful predictions: {summary['successful_predictions']}/{summary['total_structures']}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final scores: {final_csv_path}")
    print(f"Summary: {summary_path}")
    
    # Print top-ranked structures
    if summary['top_ranked_decoys']:
        print("\nTop 5 ranked decoys by composite score:")
        top_decoys = summary['top_ranked_decoys'][:5]
        for i, decoy in enumerate(top_decoys, 1):
            print(f"{i}. {decoy['id']}: {decoy['composite']:.4f}")
    else:
        print("\nNo successful predictions to rank.")
    
    # Memory cleanup
    clear_mem()


if __name__ == "__main__":
    main()
