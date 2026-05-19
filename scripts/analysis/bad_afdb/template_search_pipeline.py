#!/usr/bin/env python3
"""
Template Search and Structure Comparison Pipeline

This pipeline processes proteins from a CSV dataset through:
1. MSA download from AlphaFold Database
2. Template search using hhsearch against pdb70
3. Structure download and chain extraction
4. Structural alignment using USalign
5. TM-score histogram generation

All paths are CLI-driven. Defaults match the ~/data/bad_afdb/ layout used by the
rest of the bad_afdb pipeline (see ./README.md), but any of them may be overridden.

The hhsearch and USalign steps run inside a conda environment. By default the
environment is named ``openfold`` and the conda hook is resolved from
``$CONDA_EXE`` (falling back to ``/opt/tljh/user/bin/conda`` for the original
hosting environment). Override with --conda_env / --conda_hook as needed.

Usage:
  python template_search_pipeline.py --top_k 4
  python template_search_pipeline.py --input_csv .../small.csv --top_k 1 \\
    --conda_env openfold --conda_hook /opt/conda/bin/conda
"""

import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, Select


# Defaults — overridable via CLI. Data directories live under ~/data/bad_afdb/
# regardless of where this script lives in the proteina repo.
DEFAULT_BASE_DIR = Path.home() / "data" / "bad_afdb"
DEFAULT_INPUT_CSV = DEFAULT_BASE_DIR / "pdb_70_cluster_reps_aligned_confidence_aggregate_plddt_50_length_50-768_tms_in-train.csv"
DEFAULT_PDB70_DB = Path.home() / "data" / "pdb70_220313" / "pdb70"


@dataclass
class Config:
    """Resolved pipeline configuration (paths + conda settings)."""
    input_csv: Path
    pdb70_db: Path
    msa_dir: Path
    template_msa_dir: Path
    pdb_dir: Path
    template_pdb_dir: Path
    template_aln_dir: Path
    conda_env: str
    conda_hook: str
    top_k: int
    tm_mode: int


def ensure_dirs(cfg: Config) -> None:
    for d in (cfg.msa_dir, cfg.template_msa_dir, cfg.template_pdb_dir, cfg.template_aln_dir):
        d.mkdir(parents=True, exist_ok=True)


def run_conda_command(cfg: Config, cmd: str, cwd=None):
    """Run a command with a conda environment activated."""
    full_cmd = (
        f'eval "$({cfg.conda_hook} shell.bash hook)" && '
        f'conda activate {cfg.conda_env} && {cmd}'
    )
    return subprocess.run(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def download_msa(cfg: Config, uniprot_id: str):
    """Download MSA from AlphaFold Database."""
    msa_file = cfg.msa_dir / f"{uniprot_id}.a3m"
    if msa_file.exists():
        print(f"  MSA already exists: {msa_file}")
        return msa_file

    url = f"https://alphafold.ebi.ac.uk/files/msa/AF-{uniprot_id}-F1-msa_v6.a3m"
    print(f"  Downloading MSA from: {url}")

    result = subprocess.run(
        ["wget", "-q", "-O", str(msa_file), url],
        capture_output=True,
    )

    if result.returncode != 0 or not msa_file.exists():
        print(f"  WARNING: Failed to download MSA for {uniprot_id}")
        return None

    print(f"  MSA downloaded: {msa_file}")
    return msa_file


def run_hhsearch(cfg: Config, msa_file: Path, uniprot_id: str):
    """Run hhsearch against pdb70 database."""
    hhr_file = cfg.template_msa_dir / f"{uniprot_id}.hhr"
    if hhr_file.exists():
        print(f"  hhsearch output already exists: {hhr_file}")
        return hhr_file

    print("  Running hhsearch...")
    cmd = f"hhsearch -i {msa_file} -d {cfg.pdb70_db} -o {hhr_file} -maxseq 1000000 -cpu 8 -v 1"
    result = run_conda_command(cfg, cmd)

    if result.returncode != 0 or not hhr_file.exists():
        print(f"  WARNING: hhsearch failed for {uniprot_id}")
        print(f"  STDERR: {result.stderr}")
        return None

    print(f"  hhsearch completed: {hhr_file}")
    return hhr_file


def parse_hhr_file(hhr_file: Path, top_k: int = 4):
    """Parse hhsearch .hhr file to extract top k hits by sum_probs."""
    hits = []

    with open(hhr_file, "r") as f:
        content = f.read()

    hit_pattern = r">(\S+)\s+([^\n]*)\n.*?Sum_probs=([\d.]+)"
    matches = re.findall(hit_pattern, content, re.DOTALL)

    for match in matches:
        pdb_chain = match[0]
        sum_probs = float(match[2])

        parts = pdb_chain.split("_")
        if len(parts) >= 2:
            hits.append(
                {
                    "pdb_id": parts[0],
                    "chain_id": parts[1],
                    "sum_probs": sum_probs,
                    "full_id": pdb_chain,
                }
            )

    hits.sort(key=lambda x: x["sum_probs"], reverse=True)
    return hits[:top_k]


def download_pdb_file(cfg: Config, pdb_id: str):
    """Download PDB file from RCSB into the mid-2-char subdir."""
    middle_chars = pdb_id[1:3].upper()
    pdb_subdir = cfg.pdb_dir / middle_chars
    pdb_subdir.mkdir(parents=True, exist_ok=True)

    pdb_file = pdb_subdir / f"{pdb_id}.cif"
    if pdb_file.exists():
        print(f"    PDB file already exists: {pdb_file}")
        return pdb_file

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    print(f"    Downloading PDB from: {url}")

    result = subprocess.run(
        ["wget", "-q", "-O", str(pdb_file), url],
        capture_output=True,
    )

    if result.returncode != 0 or not pdb_file.exists():
        print(f"    WARNING: Failed to download PDB {pdb_id}")
        return None

    print(f"    PDB downloaded: {pdb_file}")
    return pdb_file


def extract_chain_from_cif(cif_file: Path, chain_id: str, output_pdb: Path):
    """Extract a specific chain from CIF file and save as PDB.

    Tries a fast text-based parse first; falls back to BioPython on any sign of
    missing/malformed data (e.g. '?' coordinates or empty chain extraction).
    """
    if output_pdb.exists():
        print(f"    Extracted chain already exists: {output_pdb}")
        return output_pdb

    atoms = []
    with open(cif_file, "r") as f:
        in_atom_site = False
        header_parsed = False

        for line in f:
            if line.startswith("_atom_site."):
                if not header_parsed:
                    in_atom_site = True
                continue

            if line.startswith("ATOM") or line.startswith("HETATM"):
                in_atom_site = True
                header_parsed = True

            if in_atom_site and not line.startswith("_") and not line.startswith("#"):
                parts = line.split()
                if len(parts) < 20:
                    continue
                line_chain = parts[18] if len(parts) > 18 else parts[6] if len(parts) > 6 else None
                if line_chain and line_chain == chain_id:
                    atoms.append(line)

    if not atoms:
        print(f"    WARNING: No atoms found for chain {chain_id} in {cif_file}")
        return extract_chain_using_biopython(cif_file, chain_id, output_pdb)

    parsed_successfully = False
    with open(output_pdb, "w") as f:
        atom_num = 1
        for line in atoms[:1000]:
            parts = line.split()
            if len(parts) >= 20:
                record = parts[0]
                atom_name = parts[3]
                res_name = parts[5]
                res_seq = parts[8]
                x, y, z = parts[10], parts[11], parts[12]
                element = parts[2]

                if x == "?" or y == "?" or z == "?":
                    print("    WARNING: Missing coordinates in CIF file, using BioPython")
                    parsed_successfully = False
                    break

                pdb_line = (
                    f"{record:<6s}{atom_num:>5d}  {atom_name:<4s}{res_name:>3s} "
                    f"{chain_id}{res_seq:>4s}    {float(x):>8.3f}{float(y):>8.3f}{float(z):>8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
                f.write(pdb_line)
                atom_num += 1
                parsed_successfully = True

    if not parsed_successfully:
        output_pdb.unlink(missing_ok=True)
        return extract_chain_using_biopython(cif_file, chain_id, output_pdb)

    print(f"    Chain extracted: {output_pdb}")
    return output_pdb


def extract_chain_using_biopython(cif_file: Path, chain_id: str, output_pdb: Path):
    """Fallback: extract chain using BioPython (handles multi-char chain IDs)."""

    class ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return chain.id == self.chain_id

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", str(cif_file))

    # BioPython PDBIO only supports single-character chain IDs.
    if len(chain_id) > 1:
        output_cif = output_pdb.with_suffix(".cif")
        io_cif = MMCIFIO()
        io_cif.set_structure(structure)
        io_cif.save(str(output_cif), ChainSelect(chain_id))

        parser2 = MMCIFParser(QUIET=True)
        structure2 = parser2.get_structure("struct2", str(output_cif))

        for model in structure2:
            for chain in model:
                if chain.id == chain_id:
                    chain.id = chain_id[0]

        io_pdb = PDBIO()
        io_pdb.set_structure(structure2)
        io_pdb.save(str(output_pdb))

        output_cif.unlink(missing_ok=True)
    else:
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_pdb), ChainSelect(chain_id))

    if output_pdb.exists():
        print(f"    Chain extracted with BioPython: {output_pdb}")
        return output_pdb

    print(f"    WARNING: Failed to extract chain {chain_id} from {cif_file}")
    return None


def run_usalign(cfg: Config, target_pdb: Path, template_pdbs, pdb_id: str, chain_id: str, tm_mode: int):
    """Run USalign to compare target structure with template structures."""
    pdb_chain_id = f"{pdb_id}_{chain_id}"
    aln_file = cfg.template_aln_dir / f"{pdb_chain_id}.tsv"
    if aln_file.exists():
        print(f"  USalign output already exists: {aln_file}")
        return aln_file

    if not template_pdbs:
        print("  WARNING: No template PDBs to align")
        return None

    temp_list_file = cfg.template_aln_dir / f"{pdb_chain_id}_temp_list.txt"
    with open(temp_list_file, "w") as f:
        for tpdb in template_pdbs:
            f.write(f"{tpdb.name}\n")

    print(f"  Running USalign with {len(template_pdbs)} templates...")
    cmd = f"USalign -outfmt 2 {target_pdb} -dir2 {cfg.template_pdb_dir}/ {temp_list_file} -TMscore {tm_mode}"
    result = run_conda_command(cfg, cmd)

    temp_list_file.unlink()

    if result.returncode != 0:
        print(f"  WARNING: USalign failed for {pdb_chain_id}")
        print(f"  STDERR: {result.stderr}")
        return None

    if not result.stdout or len(result.stdout.strip()) == 0:
        print(f"  WARNING: USalign produced no output for {pdb_chain_id}")
        if result.stderr:
            print(f"  STDERR: {result.stderr}")
        return None

    with open(aln_file, "w") as f:
        f.write(result.stdout)

    print(f"  USalign completed: {aln_file}")
    return aln_file


def parse_usalign_output(aln_file: Path):
    """Parse USalign output to extract TM1 scores in template order."""
    tm_scores = []

    with open(aln_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line or line.startswith("Warning"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                tm_scores.append(float(parts[2]))

    return tm_scores


def process_protein(cfg: Config, pdb_id: str, chain_id: str, uniprot_id: str):
    """Run the full pipeline on a single protein.

    Returns dict with 'top1_tm', 'max_tm', 'all_tm' (or None on failure).
    """
    print(f"\nProcessing {pdb_id}_{chain_id} (UniProt: {uniprot_id})")

    msa_file = download_msa(cfg, uniprot_id)
    if not msa_file:
        return None

    hhr_file = run_hhsearch(cfg, msa_file, uniprot_id)
    if not hhr_file:
        return None

    top_hits = parse_hhr_file(hhr_file, top_k=cfg.top_k)
    print(f"  Found {len(top_hits)} top hits")

    template_pdbs = []
    for hit in top_hits:
        print(f"    Hit: {hit['full_id']} (sum_probs={hit['sum_probs']:.1f})")

        pdb_file = download_pdb_file(cfg, hit["pdb_id"])
        if not pdb_file:
            continue

        output_pdb = cfg.template_pdb_dir / f"{hit['pdb_id']}_{hit['chain_id']}.pdb"
        chain_pdb = extract_chain_from_cif(pdb_file, hit["chain_id"], output_pdb)
        if chain_pdb:
            template_pdbs.append(chain_pdb)

    middle_chars = pdb_id[1:3].upper()
    target_pdb = cfg.pdb_dir / middle_chars / f"{pdb_id}.cif"

    if not target_pdb.exists():
        print(f"  WARNING: Target PDB not found: {target_pdb}")
        return None

    target_chain_pdb = cfg.template_pdb_dir / f"{pdb_id}_{chain_id}_target.pdb"
    if not target_chain_pdb.exists():
        extract_chain_from_cif(target_pdb, chain_id, target_chain_pdb)

    if not target_chain_pdb.exists():
        print("  WARNING: Failed to extract target chain")
        return None

    aln_file = run_usalign(cfg, target_chain_pdb, template_pdbs, pdb_id, chain_id, cfg.tm_mode)
    if not aln_file:
        return None

    tm_scores = parse_usalign_output(aln_file)
    if not tm_scores:
        print("  WARNING: No TM-scores found in alignment output")
        return None

    top1_tm = tm_scores[0]
    max_tm = max(tm_scores)

    print(f"  Top-1 TM-score: {top1_tm:.4f}, Max TM-score: {max_tm:.4f}")
    return {"top1_tm": top1_tm, "max_tm": max_tm, "all_tm": tm_scores}


def generate_histogram(cfg: Config, top1_scores, max_scores):
    """Generate histograms for both top-1 and top-k TM-scores."""
    print("\nGenerating histograms...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(top1_scores, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("TM-score (TM1)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of Top-1 TM-scores\n(Best Template by sum_probs)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    mean_tm1 = np.mean(top1_scores)
    median_tm1 = np.median(top1_scores)
    ax1.axvline(mean_tm1, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_tm1:.3f}")
    ax1.axvline(median_tm1, color="g", linestyle="--", linewidth=2, label=f"Median: {median_tm1:.3f}")
    ax1.legend()

    ax2.hist(max_scores, bins=50, edgecolor="black", alpha=0.7, color="coral")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Maximum TM-score (TM1)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(
        f"Distribution of Maximum TM-scores\n(Best among Top-{cfg.top_k} Templates)",
        fontsize=14,
    )
    ax2.grid(True, alpha=0.3)

    mean_max = np.mean(max_scores)
    median_max = np.median(max_scores)
    ax2.axvline(mean_max, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_max:.3f}")
    ax2.axvline(median_max, color="g", linestyle="--", linewidth=2, label=f"Median: {median_max:.3f}")
    ax2.legend()

    plt.tight_layout()
    output_file = cfg.template_aln_dir / "tm_score_distribution.png"
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    print(f"Histograms saved: {output_file}")
    plt.close()

    print("\nStatistics:")
    print(f"  Total proteins: {len(top1_scores)}")
    print("\n  Top-1 (by sum_probs):")
    print(f"    Mean TM-score: {mean_tm1:.4f}")
    print(f"    Median TM-score: {median_tm1:.4f}")
    print(f"    Min TM-score: {min(top1_scores):.4f}")
    print(f"    Max TM-score: {max(top1_scores):.4f}")
    print(f"\n  Top-{cfg.top_k} (maximum):")
    print(f"    Mean TM-score: {mean_max:.4f}")
    print(f"    Median TM-score: {median_max:.4f}")
    print(f"    Min TM-score: {min(max_scores):.4f}")
    print(f"    Max TM-score: {max(max_scores):.4f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Template Search and Structure Comparison Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV,
                   help="Input CSV with 'pdb' (PDB_chain) and 'uniprot' columns.")
    p.add_argument("--pdb70_db", type=Path, default=DEFAULT_PDB70_DB,
                   help="Path prefix to pdb70 hhsearch database.")
    p.add_argument("--msa_dir", type=Path, default=DEFAULT_BASE_DIR / "msa",
                   help="Directory for downloaded AFDB MSAs (.a3m).")
    p.add_argument("--template_msa_dir", type=Path, default=DEFAULT_BASE_DIR / "template_msa",
                   help="Directory for hhsearch outputs (.hhr).")
    p.add_argument("--pdb_dir", type=Path, default=DEFAULT_BASE_DIR / "pdb",
                   help="Directory for native PDB mmCIF files (mid-2-char subdirs).")
    p.add_argument("--template_pdb_dir", type=Path, default=DEFAULT_BASE_DIR / "template_pdb",
                   help="Directory for extracted template chains (.pdb).")
    p.add_argument("--template_aln_dir", type=Path, default=DEFAULT_BASE_DIR / "template_aln",
                   help="Directory for USalign TSV outputs and histograms.")
    p.add_argument("--conda_env", default="openfold",
                   help="Conda environment with hhsearch + USalign installed.")
    p.add_argument(
        "--conda_hook",
        default=os.environ.get("CONDA_EXE", "/opt/tljh/user/bin/conda"),
        help="Path to the conda executable used to source the bash hook "
             "(defaults to $CONDA_EXE if set, else /opt/tljh/user/bin/conda).",
    )
    p.add_argument("-k", "--top_k", type=int, default=4,
                   help="Number of top templates to use.")
    p.add_argument("--tm_mode", type=int, default=5, help="TM-score mode to use.")
    return p


def main():
    args = build_arg_parser().parse_args()

    cfg = Config(
        input_csv=Path(args.input_csv).expanduser(),
        pdb70_db=Path(args.pdb70_db).expanduser(),
        msa_dir=Path(args.msa_dir).expanduser(),
        template_msa_dir=Path(args.template_msa_dir).expanduser(),
        pdb_dir=Path(args.pdb_dir).expanduser(),
        template_pdb_dir=Path(args.template_pdb_dir).expanduser(),
        template_aln_dir=Path(args.template_aln_dir).expanduser(),
        conda_env=args.conda_env,
        conda_hook=args.conda_hook,
        top_k=args.top_k,
        tm_mode=args.tm_mode,
    )
    ensure_dirs(cfg)

    print("=" * 80)
    print("Template Search and Structure Comparison Pipeline")
    print(f"Using top-{cfg.top_k} templates")
    print(f"Input CSV         : {cfg.input_csv}")
    print(f"pdb70 db          : {cfg.pdb70_db}")
    print(f"MSA dir           : {cfg.msa_dir}")
    print(f"Template MSA dir  : {cfg.template_msa_dir}")
    print(f"PDB dir           : {cfg.pdb_dir}")
    print(f"Template PDB dir  : {cfg.template_pdb_dir}")
    print(f"Template aln dir  : {cfg.template_aln_dir}")
    print(f"Conda env / hook  : {cfg.conda_env} / {cfg.conda_hook}")
    print("=" * 80)

    proteins = []
    with open(cfg.input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_chain = row["pdb"]
            uniprot_id = row["uniprot"]
            parts = pdb_chain.split("_")
            if len(parts) >= 2:
                proteins.append(
                    {"pdb_id": parts[0], "chain_id": parts[1], "uniprot_id": uniprot_id}
                )

    print(f"\nFound {len(proteins)} proteins to process")

    top1_scores = []
    max_scores = []
    results_file = cfg.template_aln_dir / "tm_scores_results.tsv"

    processed_proteins = set()
    if results_file.exists():
        with open(results_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pdb_chain_id = f"{row['pdb_id']}_{row['chain_id']}"
                processed_proteins.add(pdb_chain_id)
                if "top1_tm_score" in row:
                    top1_scores.append(float(row["top1_tm_score"]))
                    max_scores.append(float(row["max_tm_score"]))
                else:
                    score = float(row["max_tm_score"])
                    top1_scores.append(score)
                    max_scores.append(score)
        print(f"Resuming from {len(processed_proteins)} already processed proteins")

    write_header = not results_file.exists()
    with open(results_file, "a") as f:
        if write_header:
            f.write("pdb_id\tchain_id\tuniprot_id\ttop1_tm_score\tmax_tm_score\n")

        for i, protein in enumerate(proteins, 1):
            pdb_chain_id = f"{protein['pdb_id']}_{protein['chain_id']}"

            if pdb_chain_id in processed_proteins:
                continue

            print(f"\n[{i}/{len(proteins)}]", end=" ")
            result = process_protein(
                cfg, protein["pdb_id"], protein["chain_id"], protein["uniprot_id"]
            )

            if result is not None:
                top1_scores.append(result["top1_tm"])
                max_scores.append(result["max_tm"])
                f.write(
                    f"{protein['pdb_id']}\t{protein['chain_id']}\t{protein['uniprot_id']}"
                    f"\t{result['top1_tm']:.6f}\t{result['max_tm']:.6f}\n"
                )
                f.flush()

    if top1_scores and max_scores:
        generate_histogram(cfg, top1_scores, max_scores)
    else:
        print("\nWARNING: No TM-scores collected")

    print("\n" + "=" * 80)
    print("Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
