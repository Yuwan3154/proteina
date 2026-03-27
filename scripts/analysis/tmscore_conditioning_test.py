"""
TMscore Conditioning Effectiveness Test

Tests whether initial Gaussian noise dominates the sampling trajectory by:
1. Finding protein pairs with the same length but different CATH codes
2. Generating N samples for each protein using the SAME random seed
   (identical initial Gaussian noise)
3. Computing TMscore between sample_i of protein_1 and sample_i of protein_2

If TMscore is consistently high despite different conditioning (sequence + CATH),
the initial noise dominates and conditioning is ineffective.

Usage:
    python scripts/analysis/tmscore_conditioning_test.py \
        --ckpt data/weights/<checkpoint>.ckpt \
        --n_samples 10 \
        --output_dir results/tmscore_conditioning
"""

import argparse
import csv
import gzip
import os
import sys
from collections import defaultdict

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

import glob

import numpy as np
import torch
torch.set_float32_matmul_precision("medium")

from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.coors_utils import trans_nm_to_atom37
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.datasets.cath_utils import cath_code_strings_to_indices_for_model
from proteinfoundation.af2rank_evaluation.proteinebm_scorer import tmscore_pdb_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cath_code_lookup(cathdata_dir):
    """Parse cath-b-newest-all.gz to build {pdb_chain: [cath_code, ...]} mapping."""
    lookup = defaultdict(list)
    gz_path = os.path.join(cathdata_dir, "cath-b-newest-all.gz")
    if not os.path.exists(gz_path):
        print(f"WARNING: {gz_path} not found — CATH codes unavailable")
        return lookup
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            domain_id = parts[0]
            cath_code = parts[2]
            if len(domain_id) < 5:
                continue
            pdb = domain_id[:4].lower()
            chain = domain_id[4].upper()
            key = f"{pdb}_{chain}"
            if cath_code not in lookup[key]:
                lookup[key].append(cath_code)
    return lookup


def find_protein_pairs(data_dir, cath_lookup, n_pairs=3, min_length=60, max_length=300, seed=42):
    """Find pairs of proteins with the same length but different CATH superfamilies."""
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    # Group by length
    length_groups = defaultdict(list)
    for f in pt_files:
        protein_id = os.path.basename(f).replace(".pt", "")
        pt = torch.load(f, weights_only=False)
        n = pt.residue_type.shape[0]
        if min_length <= n <= max_length:
            cath_codes = cath_lookup.get(protein_id, [])
            # Only keep proteins with known CATH codes
            if cath_codes:
                # Use the topology level (x.x.x) for comparison
                topology = ".".join(cath_codes[0].split(".")[:3])
                length_groups[n].append((protein_id, f, cath_codes, topology))

    # Find pairs with different CATH topologies at the same length
    pairs = []
    rng = np.random.default_rng(seed)
    lengths = sorted(length_groups.keys())
    rng.shuffle(lengths)

    for length in lengths:
        proteins = length_groups[length]
        if len(proteins) < 2:
            continue

        # Group by topology, pick from different topologies
        topo_groups = defaultdict(list)
        for p in proteins:
            topo_groups[p[3]].append(p)

        topos = list(topo_groups.keys())
        if len(topos) < 2:
            continue

        # Pick one from each of two different topologies
        topo_idx = rng.choice(len(topos), size=2, replace=False)
        t1, t2 = topos[topo_idx[0]], topos[topo_idx[1]]
        p1 = topo_groups[t1][rng.integers(len(topo_groups[t1]))]
        p2 = topo_groups[t2][rng.integers(len(topo_groups[t2]))]
        pairs.append((p1, p2, length))

        if len(pairs) >= n_pairs:
            break

    return pairs


def generate_with_seed(model, seed, n, dt, residue_type, cath_indices, cath_mask,
                       self_cond, sampling_mode, sc_scale_noise, device):
    """Generate a single sample with a specific random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    result = model.generate(
        nsamples=1,
        n=n,
        dt=dt,
        self_cond=self_cond,
        cath_code_indices=cath_indices,
        cath_code_indices_mask=cath_mask,
        residue_type=residue_type,
        guidance_weight=1.0,
        autoguidance_ratio=0.0,
        dtype=torch.float32,
        schedule_mode="log",
        schedule_p=2.0,
        sampling_mode=sampling_mode,
        sc_scale_noise=sc_scale_noise,
        sc_scale_score=1.0,
        gt_mode="1/t",
        gt_p=1.0,
        gt_clamp_val=None,
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TMscore conditioning effectiveness test")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of samples per protein per pair")
    parser.add_argument("--n_pairs", type=int, default=3,
                        help="Number of protein pairs to test")
    parser.add_argument("--output_dir", default="results/tmscore_conditioning",
                        help="Output directory")
    parser.add_argument("--data_dir", default="data/pdb_train/processed",
                        help="Directory with processed .pt files")
    parser.add_argument("--cathdata_dir", default="data/cathdata",
                        help="Directory with CATH data")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="Integration step size")
    parser.add_argument("--sampling_mode", default="sc",
                        help="Sampling mode: vf (ODE) or sc (SDE)")
    parser.add_argument("--sc_scale_noise", type=float, default=0.45,
                        help="Noise scale for SDE sampling")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pdb_dir = os.path.join(args.output_dir, "pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    model = Proteina.load_from_checkpoint(args.ckpt, strict=False)
    model.eval()
    model.to(device)
    model._inf_zero_sin_pos_emb = True  # checkpoint trained with zero_sin_pos_emb=True
    print(f"Model loaded on {device}")

    cath_code_dir = model.cfg_exp.model.nn.get("cath_code_dir", "")
    if cath_code_dir and "${" in cath_code_dir:
        cath_code_dir = os.path.expandvars(cath_code_dir.replace("${oc.env:", "$").replace("}", ""))
    if not os.path.isabs(cath_code_dir):
        cath_code_dir = os.path.join(root, cath_code_dir)
    multilabel_mode = model.cfg_exp.model.nn.get("multilabel_mode", "sample")

    # Load CATH lookup
    cath_lookup = load_cath_code_lookup(os.path.join(root, args.cathdata_dir))
    print(f"Loaded CATH lookup with {len(cath_lookup)} entries")

    # Find protein pairs
    print(f"\nSearching for {args.n_pairs} protein pairs with same length, different CATH...")
    pairs = find_protein_pairs(
        os.path.join(root, args.data_dir), cath_lookup,
        n_pairs=args.n_pairs, seed=args.seed,
    )

    if not pairs:
        print("ERROR: No suitable protein pairs found!")
        return

    print(f"Found {len(pairs)} pairs:")
    for i, (p1, p2, length) in enumerate(pairs):
        print(f"  Pair {i}: {p1[0]} (CATH: {p1[2][0]}) vs {p2[0]} (CATH: {p2[2][0]}), length={length}")

    # Generate and compare
    results = []

    for pair_idx, (p1_info, p2_info, length) in enumerate(pairs):
        p1_id, p1_file, p1_cath_codes, _ = p1_info
        p2_id, p2_file, p2_cath_codes, _ = p2_info

        print(f"\n{'='*60}")
        print(f"Pair {pair_idx}: {p1_id} vs {p2_id} (length={length})")
        print(f"  CATH codes: {p1_cath_codes[0]} vs {p2_cath_codes[0]}")

        # Load protein data for sequence and CATH
        pt1 = torch.load(p1_file, weights_only=False)
        pt2 = torch.load(p2_file, weights_only=False)

        # Sequence (residue type)
        seq1 = torch.as_tensor(pt1.residue_type, dtype=torch.long)
        seq1 = torch.where(seq1 == -1, torch.tensor(20), seq1)
        seq2 = torch.as_tensor(pt2.residue_type, dtype=torch.long)
        seq2 = torch.where(seq2 == -1, torch.tensor(20), seq2)

        # CATH indices
        cath_indices_1, cath_mask_1 = cath_code_strings_to_indices_for_model(
            [p1_cath_codes], cath_code_dir, multilabel_mode, device=device
        )
        cath_indices_2, cath_mask_2 = cath_code_strings_to_indices_for_model(
            [p2_cath_codes], cath_code_dir, multilabel_mode, device=device
        )

        # Amino acid types for PDB writing
        aatype1 = np.array(pt1.residue_type).astype(int)
        aatype2 = np.array(pt2.residue_type).astype(int)

        for sample_idx in range(args.n_samples):
            seed = args.seed + sample_idx * 1000  # Different seed per sample, same for both proteins

            # Generate protein 1
            result1 = generate_with_seed(
                model, seed, length, args.dt,
                residue_type=seq1.unsqueeze(0).to(device),
                cath_indices=cath_indices_1, cath_mask=cath_mask_1,
                self_cond=True, sampling_mode=args.sampling_mode,
                sc_scale_noise=args.sc_scale_noise, device=device,
            )

            # Generate protein 2 (same seed = same initial noise)
            result2 = generate_with_seed(
                model, seed, length, args.dt,
                residue_type=seq2.unsqueeze(0).to(device),
                cath_indices=cath_indices_2, cath_mask=cath_mask_2,
                self_cond=True, sampling_mode=args.sampling_mode,
                sc_scale_noise=args.sc_scale_noise, device=device,
            )

            # Convert to atom37 and write PDBs
            coords1 = result1["coords"]  # [1, n, 3] in nm
            coords2 = result2["coords"]  # [1, n, 3] in nm

            atom37_1 = trans_nm_to_atom37(coords1).squeeze(0).cpu().numpy()  # [n, 37, 3] in Å
            atom37_2 = trans_nm_to_atom37(coords2).squeeze(0).cpu().numpy()  # [n, 37, 3] in Å

            pdb1_path = os.path.join(pdb_dir, f"pair{pair_idx}_{p1_id}_{sample_idx}.pdb")
            pdb2_path = os.path.join(pdb_dir, f"pair{pair_idx}_{p2_id}_{sample_idx}.pdb")

            write_prot_to_pdb(atom37_1, pdb1_path, aatype=aatype1, overwrite=True, no_indexing=True)
            write_prot_to_pdb(atom37_2, pdb2_path, aatype=aatype2, overwrite=True, no_indexing=True)

            # Compute TMscore
            tm_result = tmscore_pdb_paths(pdb1_path, pdb2_path)

            results.append({
                "pair_id": pair_idx,
                "protein1": p1_id,
                "protein2": p2_id,
                "cath1": p1_cath_codes[0],
                "cath2": p2_cath_codes[0],
                "length": length,
                "sample_idx": sample_idx,
                "seed": seed,
                "tmscore_1": tm_result.get("tms", float("nan")),
                "tmscore_2": tm_result.get("tms2", float("nan")),
                "rmsd": tm_result.get("rms", float("nan")),
            })

            print(f"  Sample {sample_idx}: TM1={tm_result.get('tms', 'N/A'):.3f} "
                  f"TM2={tm_result.get('tms2', 'N/A'):.3f} "
                  f"RMSD={tm_result.get('rms', 'N/A'):.2f}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "tmscore_conditioning_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "protein1", "protein2", "cath1", "cath2",
            "length", "sample_idx", "seed", "tmscore_1", "tmscore_2", "rmsd"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for pair_idx in range(len(pairs)):
        pair_results = [r for r in results if r["pair_id"] == pair_idx]
        if not pair_results:
            continue
        tm1_vals = [r["tmscore_1"] for r in pair_results if not np.isnan(r["tmscore_1"])]
        tm2_vals = [r["tmscore_2"] for r in pair_results if not np.isnan(r["tmscore_2"])]
        rmsd_vals = [r["rmsd"] for r in pair_results if not np.isnan(r["rmsd"])]

        p1 = pair_results[0]["protein1"]
        p2 = pair_results[0]["protein2"]
        print(f"\nPair {pair_idx}: {p1} vs {p2}")
        print(f"  CATH: {pair_results[0]['cath1']} vs {pair_results[0]['cath2']}")
        if tm1_vals:
            print(f"  TM-score (norm by {p1}): {np.mean(tm1_vals):.3f} +/- {np.std(tm1_vals):.3f}")
        if tm2_vals:
            print(f"  TM-score (norm by {p2}): {np.mean(tm2_vals):.3f} +/- {np.std(tm2_vals):.3f}")
        if rmsd_vals:
            print(f"  RMSD: {np.mean(rmsd_vals):.2f} +/- {np.std(rmsd_vals):.2f}")

    print("\nInterpretation:")
    all_tm1 = [r["tmscore_1"] for r in results if not np.isnan(r["tmscore_1"])]
    if all_tm1:
        mean_tm = np.mean(all_tm1)
        if mean_tm > 0.5:
            print(f"  Mean TM-score = {mean_tm:.3f} (>0.5) -> Initial noise DOMINATES trajectory.")
            print("  Conditioning (sequence + CATH) has limited effect on structure.")
        elif mean_tm > 0.3:
            print(f"  Mean TM-score = {mean_tm:.3f} (0.3-0.5) -> Mixed: noise has some influence")
            print("  but conditioning also steers the trajectory.")
        else:
            print(f"  Mean TM-score = {mean_tm:.3f} (<0.3) -> Conditioning DOMINATES.")
            print("  The model effectively uses sequence/CATH to generate different structures.")
    print(f"\nPDB files saved to {pdb_dir}")


if __name__ == "__main__":
    main()
