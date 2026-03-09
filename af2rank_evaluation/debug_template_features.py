#!/usr/bin/env python3
"""
Diagnostic script to inspect ColabDesign template features for CA-only vs all-atom PDBs.

Runs in the colabdesign environment. Uses cg2all_reconstruct.py (via subprocess to
cue_openfold) to reconstruct the CA-only decoy, then compares template features.

Usage:
    conda run -n colabdesign python debug_template_features.py
"""

import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'

import sys
import json
import tempfile
import subprocess
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DECOY_PDB = ("/home/ubuntu/proteina/inference/"
             "inference_seq_cond_sampling_ca_dssp_beta-2.5-2.0_finetune-all_v1.6_"
             "default-fold_21-seq-S25_128-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise/"
             "5SSV_A/5SSV_A_111.pdb")
REFERENCE_CIF = "/home/ubuntu/data/bad_afdb/pdb/SS/5SSV.cif"

# ── ColabDesign imports ─────────────────────────────────────────────────────
from colabdesign.af.contrib import predict
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared import protein
from colabdesign.af import prep

sys.path.insert(0, SCRIPT_DIR)
from af2rank_scorer import (
    get_sequence_from_pdb, robust_get_template_feats, run_do_not_align,
    convert_cif_to_pdb_for_colabdesign, _is_ca_only_pdb,
)


def reconstruct_via_cg2all(pdb_file):
    """Reconstruct CA-only PDB to all-atom via subprocess in cue_openfold env."""
    tmp_dir = tempfile.mkdtemp(prefix="cg2all_debug_")
    input_json = os.path.join(tmp_dir, "inputs.json")
    output_map = os.path.join(tmp_dir, "output_map.json")
    with open(input_json, 'w') as f:
        json.dump([pdb_file], f)
    wrapper_script = os.path.join(SCRIPT_DIR, "run_with_proteina_env.sh")
    cmd = [
        wrapper_script, "python", os.path.join(SCRIPT_DIR, "cg2all_reconstruct.py"),
        "--inputs", input_json,
        "--output_dir", tmp_dir,
        "--output_map", output_map,
    ]
    subprocess.run(cmd, check=True, timeout=300)
    with open(output_map) as f:
        mapping = json.load(f)
    return mapping.get(pdb_file)


def get_template_features(pdb_file, reference_sequence, chain="A"):
    """Extract template features via ColabDesign's pipeline."""
    batch = robust_get_template_feats(
        pdbs=[pdb_file],
        chains=[chain],
        query_seq=reference_sequence,
        query_a3m=None,
        copies=1,
        propagate_to_copies=False,
        use_seq=False,          # AF2Rank: no sequence info
        align_fn=run_do_not_align,
    )
    return batch


def print_template_stats(label, batch, reference_sequence):
    n_res = len(reference_sequence)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Batch keys: {sorted(batch.keys())}")
    print(f"  Reference residues: {n_res}")

    # ── print shapes of key fields ──────────────────────────────────────────
    for key in sorted(batch.keys()):
        val = batch[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}")
        else:
            print(f"  {key}: type={type(val)}")

    # ── try common key variants ─────────────────────────────────────────────
    pb_mask_key = None
    for k in ["template_pseudo_beta_mask", "pseudo_beta_mask"]:
        if k in batch:
            pb_mask_key = k; break
    atom_mask_key = None
    for k in ["template_all_atom_masks", "template_all_atom_mask", "all_atom_mask"]:
        if k in batch:
            atom_mask_key = k; break
    atom_pos_key = None
    for k in ["template_all_atom_positions", "all_atom_positions"]:
        if k in batch:
            atom_pos_key = k; break
    pb_pos_key = None
    for k in ["template_pseudo_beta", "pseudo_beta"]:
        if k in batch:
            pb_pos_key = k; break

    print(f"\n  Using keys: pb_mask={pb_mask_key}, atom_mask={atom_mask_key}, atom_pos={atom_pos_key}, pb_pos={pb_pos_key}")

    # ── pseudo-beta (CB) coverage ───────────────────────────────────────────
    pb_mask = np.array(batch[pb_mask_key]) if pb_mask_key else np.zeros(n_res)
    # flatten if extra dims
    if pb_mask.ndim > 1: pb_mask = pb_mask[0]
    cb_present = int(pb_mask.sum())
    print(f"\n  pseudo_beta_mask (CB present): {cb_present}/{len(pb_mask)} residues  ({100*cb_present/max(1,len(pb_mask)):.1f}%)")

    # ── backbone atom coverage ──────────────────────────────────────────────
    if atom_mask_key:
        all_atom_mask = np.array(batch[atom_mask_key])
        if all_atom_mask.ndim > 2: all_atom_mask = all_atom_mask[0]
        backbone_atoms = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
        for atom_name, idx in backbone_atoms.items():
            if idx < all_atom_mask.shape[-1]:
                present = int(all_atom_mask[:, idx].sum())
                print(f"  {atom_name:3s} present: {present}/{len(all_atom_mask)}  ({100*present/max(1,len(all_atom_mask)):.1f}%)")
    else:
        print("  (no atom_mask key found)")

    # ── pseudo-beta positions (check if non-zero) ───────────────────────────
    if pb_pos_key:
        pb_pos = np.array(batch[pb_pos_key])
        if pb_pos.ndim > 2: pb_pos = pb_pos[0]
        nonzero_cb = int((np.abs(pb_pos).sum(axis=-1) > 0).sum())
        print(f"  non-zero CB coordinate vectors: {nonzero_cb}/{len(pb_pos)}")

        # ── CB-CB pairwise distances ────────────────────────────────────────
        if cb_present > 0 and pb_mask_key:
            cb_coords = pb_pos[pb_mask > 0]
            diff = cb_coords[:, None, :] - cb_coords[None, :, :]
            dists = np.sqrt((diff**2).sum(-1))
            pos_dists = dists[dists > 0]
            if len(pos_dists) > 0:
                print(f"  CB-CB distances: min={pos_dists.min():.2f}  max={dists.max():.2f}  mean={pos_dists.mean():.2f} Å")
        else:
            print("  CB-CB distances: N/A")

    # ── dgram statistics ────────────────────────────────────────────────────
    if "dgram" in batch:
        dgram = np.array(batch["dgram"])
        print(f"\n  dgram shape: {dgram.shape}")
        print(f"  dgram min={dgram.min():.4f}  max={dgram.max():.4f}  mean={dgram.mean():.4f}  std={dgram.std():.4f}")
        # Entropy of the distogram distribution per pair (high entropy = uncertain/flat)
        eps = 1e-8
        probs = dgram / (dgram.sum(-1, keepdims=True) + eps)
        entropy = -(probs * np.log(probs + eps)).sum(-1)
        print(f"  dgram entropy per pair: min={entropy.min():.3f}  max={entropy.max():.3f}  mean={entropy.mean():.3f}")
        print(f"  (lower entropy = more confident distance signal)")

    # ── backbone torsion angles ─────────────────────────────────────────────
    for key in ["template_torsion_angles_sin_cos", "torsion_angles_sin_cos"]:
        if key in batch:
            torsion = np.array(batch[key])
            if torsion.ndim > 3: torsion = torsion[0]
            nonzero = int((np.abs(torsion).sum(axis=(-1, -2)) > 0).sum())
            print(f"  non-zero torsion rows ({key}): {nonzero}/{torsion.shape[0]}")
            break


def main():
    # ── reference sequence ──────────────────────────────────────────────────
    print(f"Loading reference from {REFERENCE_CIF}")
    ref_pdb = convert_cif_to_pdb_for_colabdesign(REFERENCE_CIF, chain_id="A")
    reference_sequence = get_sequence_from_pdb(ref_pdb, chain="A")
    os.unlink(ref_pdb)
    print(f"Reference sequence length: {len(reference_sequence)}")

    # ── check decoy type ────────────────────────────────────────────────────
    is_ca_only = _is_ca_only_pdb(DECOY_PDB)
    print(f"\nDecoy: {DECOY_PDB}")
    print(f"CA-only: {is_ca_only}")

    # ── Case 1: CA-only PDB directly into ColabDesign ──────────────────────
    print("\nExtracting template features from CA-only PDB (baseline / pre-fix)...")
    batch_ca = get_template_features(DECOY_PDB, reference_sequence)
    print_template_stats("CA-only PDB (no cg2all)", batch_ca, reference_sequence)

    # ── Case 2: cg2all-reconstructed all-atom PDB ──────────────────────────
    print("\nReconstructing with cg2all (subprocess → cue_openfold)...")
    allatom_pdb = reconstruct_via_cg2all(DECOY_PDB)
    if allatom_pdb and os.path.exists(allatom_pdb):
        print(f"Reconstructed: {allatom_pdb}")
        batch_aa = get_template_features(allatom_pdb, reference_sequence)
        print_template_stats("cg2all all-atom PDB (post-fix)", batch_aa, reference_sequence)

        # ── Direct dgram comparison ─────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  DGRAM COMPARISON (are they numerically identical?)")
        print(f"{'='*60}")
        dgram_ca = np.array(batch_ca["dgram"])
        dgram_aa = np.array(batch_aa["dgram"])
        are_identical = np.allclose(dgram_ca, dgram_aa)
        print(f"  dgrams numerically identical: {are_identical}")
        if not are_identical:
            diff = np.abs(dgram_ca - dgram_aa)
            print(f"  max abs diff: {diff.max():.6f}")
            print(f"  mean abs diff: {diff.mean():.6f}")
            changed_pairs = int((diff.sum(-1) > 0.01).sum())
            print(f"  pairs with significant difference: {changed_pairs}/{dgram_ca.shape[0]*dgram_ca.shape[1]}")
        else:
            print("  WARNING: dgrams are identical - CB positions not affecting distogram!")
            print("  This means cg2all preprocessing has NO effect on template signal.")

        # ── Inspect how set_template uses the batch ─────────────────────────
        print(f"\n{'='*60}")
        print(f"  HOW set_template uses dgram vs all_atom_positions")
        print(f"{'='*60}")
        import inspect
        from colabdesign.af import inputs as af_inputs
        src = inspect.getsource(af_inputs._update_template)
        print(src[:3000])

        os.unlink(allatom_pdb)
    else:
        print("ERROR: cg2all reconstruction failed!")

    print("\nDone.")


if __name__ == "__main__":
    main()
