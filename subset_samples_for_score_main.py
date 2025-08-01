from pathlib import Path
import os
os.environ["DATA_PATH"] = str(Path.home() / "proteina/data")
import subprocess
import argparse
import json
import hydra
import lightning as L
import numpy as np
import pandas as pd
import scipy
import torch
from proteinfoundation.datasets.transforms import CATHLabelTransform
from Bio.PDB import PDBParser, PDBIO, is_aa
from tqdm import tqdm
from matplotlib import pyplot as plt

def save_PDB(coords, residues, chain_id, resi_ids, pdb_out, atoms=["N", "CA", "C", "O", "CB"]):
    '''
    ===================================================================
    input: (ensemble_size, length, atoms=(N, CA, C, O, CB), coords=(x, y, z))
    ===================================================================
    '''
    num_models = coords.shape[0]
    out = open(pdb_out, "w")
    k = 1
    for m, model in enumerate(coords):
        if num_models > 1:
            out.write("MODEL    %5d\n" % (m + 1))
        for r, residue in enumerate(model[:len(residues)]):
            res_name = residues[r]
            res_id = resi_ids[r]
            for a, atom in enumerate(residue):
                x, y, z = atom
                if not np.isnan(x):
                    out.write(
                        "ATOM  %5d  %-2s  %3s %s%4s    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                        % (k, atoms[a], res_name, chain_id, res_id, x, y, z, 1.00, 0.00)
                    )
                k += 1
        if num_models > 1:
            out.write("ENDMDL\n")
    out.close()

def subset_samples(inference_path, aln_df, bin_num=10, num_samples_per_bin=10):
    tms = aln_df["TM1"]
    min_tms = min(tms)
    max_tms = max(tms)
    bin_width = (max_tms - min_tms) / bin_num
    bins = [min_tms + i * bin_width for i in range(bin_num)]
    counts = [0] * bin_num
    binned_indices = [[] for _ in range(bin_num)]
    for i, tm in enumerate(tms):
        bin_index = int((tm - min_tms) // bin_width) - 1
        binned_indices[bin_index].append(i)
        counts[bin_index] += 1
    
    if not os.path.exists(inference_path / "sampled_files.txt"):
        sampled_indices = []
        sampled_files = []
        sampled_tms = []
        # Draw 10 examples in each bin
        for i in range(bin_num):
            if counts[i] > 0:
                sampled_indices.append(np.random.choice(binned_indices[i], size=min(num_samples_per_bin, counts[i]), replace=False))
                sampled_tms.extend(tms[sampled_indices[i]])
                in_files = aln_df.iloc[sampled_indices[i]]["PDBchain2"].str.split("/").str[-1].str.split(":").str[0].tolist()
                out_files = [file.replace(".pdb", "_cg2all.pdb") for file in in_files]
                sampled_files.extend(out_files)

        # Save sampled_files to a text file
        with open(inference_path / "sampled_files.txt", "w") as f:
            for file in sampled_files:
                f.write(file + "\n")

    # If the sampled_files.txt file already exists, load the sampled_files and sampled_indices from the file
    else:
        with open(inference_path / "sampled_files.txt", "r") as f:
            sampled_files = [line.strip() for line in f.readlines()]
        print(f"Found {len(sampled_files)} existing sampled files")
        sampled_indices = []
        sampled_tms = []
        out_names = aln_df["PDBchain2"].str.split("/").str[-1].str.split(":").str[0]
        for file in sampled_files:
            matching_indices = out_names[out_names == file.replace("_cg2all.pdb", ".pdb")].index
            if len(matching_indices) > 0:
                sampled_indices.append(matching_indices[0])
                sampled_tms.append(tms[sampled_indices[-1]])
            else:
                print(f"Warning: File {file} not found in dataframe, skipping")

    # Save sampled_files to a csv called subset_data.csv with one column "description" to follow formatting with existing analysis code
    if not os.path.exists(inference_path / "subset_data.csv"):
        # Save sampled_files to a csv called subset_data.csv with one column "description" to follow formatting with existing analysis code
        sampled_files_df = pd.DataFrame({"description": sampled_files})
        sampled_files_df["description"] = sampled_files_df["description"].str.replace(".pdb", "")
        sampled_files_df.to_csv(inference_path / "subset_data.csv", index=False)
    
    # Plot the histogram
    plt.hist(tms, bins=np.arange(min_tms, max_tms+bin_width, bin_width), color="blue", alpha=0.5, label="Original")
    plt.hist(sampled_tms, bins=np.arange(min_tms, max_tms+bin_width, bin_width), color="red", alpha=0.5, label="Sampled")
    plt.yscale("log")
    plt.legend()
    plt.savefig(inference_path / "sampled_tms_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_path", "-i", type=str, required=True)
    parser.add_argument("--aln_file", "-f", type=str, required=True)
    parser.add_argument("--pt_name", "-p", type=str, required=True)
    parser.add_argument("--bin_num", "-b", type=int, default=10)
    parser.add_argument("--num_samples_per_bin", "-n", type=int, default=10)
    args = parser.parse_args()

    inference_path = Path(args.inference_path) / args.pt_name
    aln_df = pd.read_csv(inference_path / args.aln_file, sep="\t")
    subset_samples(inference_path, aln_df, bin_num=args.bin_num, num_samples_per_bin=args.num_samples_per_bin) 