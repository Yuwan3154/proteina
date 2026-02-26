from pathlib import Path
import subprocess
import argparse
import pandas as pd

def convert_cg2all_files(inference_path, aln_file):
    """
    Convert the sampled files using convert_cg2all command.
    This function reads the sampled_files.txt file and converts the corresponding input files.
    """
    inference_path = Path(inference_path)
    aln_df = pd.read_csv(inference_path / aln_file, sep="\t")
    
    # Read the sampled files
    with open(inference_path / "sampled_files.txt", "r") as f:
        sampled_files = [line.strip() for line in f.readlines()]
    
    print(f"Converting {len(sampled_files)} files using convert_cg2all...")
    
    # Get the input file names (remove _cg2all.pdb suffix)
    in_files = [file.replace("_cg2all.pdb", ".pdb") for file in sampled_files]
    
    # Convert each file
    for in_file, out_file in zip(in_files, sampled_files):
        print(f"Converting {in_file} to {out_file}")
        try:
            result = subprocess.run([
                "convert_cg2all", 
                "-p", str(inference_path / in_file), 
                "-o", str(inference_path / out_file), 
                "--cg", "ca", 
                "--device", "cpu"
            ], capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                print(f"Error converting {in_file}: {result.stderr}")
            else:
                print(f"Successfully converted {in_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error converting {in_file}: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error converting {in_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_path", "-i", type=str, required=True)
    parser.add_argument("--aln_file", "-f", type=str, required=True)
    args = parser.parse_args()

    convert_cg2all_files(args.inference_path, args.aln_file) 