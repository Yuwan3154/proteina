#!/usr/bin/env python3
"""
Benchmark AF2Rank speed: OpenFold vs ColabDesign backends.

Scores 5 decoys of same length with both backends to compare:
- ColabDesign (JAX)
- OpenFold (PyTorch) with chunking disabled (chunk_size=0)
"""

import os
import sys
import time
import json
import tempfile
import subprocess

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

INFERENCE_DIR = (
    "/home/ubuntu/proteina/inference/"
    "inference_seq_cond_sampling_ca_beta-2.5-2.0_finetune-all_v1.4_default-fold_4-seq-S25_64-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise"
    "/1a2y_C"
)
REFERENCE_CIF = "/home/ubuntu/data/af2rank_single/pdb/a2/1a2y.cif"
CHAIN = "C"
NUM_DECOYS = 5
RECYCLES = 3


def run_openfold_benchmark(model_name="model_1_ptm", chunk_size=None):
    """Benchmark OpenFold backend scoring."""
    from af2rank_openfold_scorer import OpenFoldAF2Rank

    decoy_pdbs = sorted(
        [os.path.join(INFERENCE_DIR, f) for f in os.listdir(INFERENCE_DIR) if f.endswith(".pdb")]
    )[:NUM_DECOYS]

    chunk_desc = f"chunk_size={chunk_size}" if chunk_size is not None else "chunking=default"
    print(f"\n{'='*70}")
    print(f"OpenFold scoring ({model_name}, {chunk_desc}) - {len(decoy_pdbs)} decoys")
    print(f"{'='*70}")

    scorer = OpenFoldAF2Rank(
        REFERENCE_CIF, chain=CHAIN, model_name=model_name, recycles=RECYCLES, chunk_size=chunk_size
    )

    times = []
    for i, pdb_path in enumerate(decoy_pdbs):
        fname = os.path.basename(pdb_path)
        t0 = time.time()
        scores = scorer.score_structure(pdb_path, decoy_chain="A", recycles=RECYCLES)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  {fname}: {elapsed:.2f}s  pTM={scores['ptm']:.4f}  pLDDT={scores['plddt']:.4f}")

    avg_time = sum(times) / len(times)
    total_time = sum(times)
    print(f"\nTotal: {total_time:.2f}s, Average: {avg_time:.2f}s/structure")
    return {"times": times, "total": total_time, "average": avg_time}


def run_colabdesign_benchmark(model_name="model_1_ptm"):
    """Benchmark ColabDesign backend scoring via subprocess."""
    decoy_pdbs = sorted(
        [os.path.join(INFERENCE_DIR, f) for f in os.listdir(INFERENCE_DIR) if f.endswith(".pdb")]
    )[:NUM_DECOYS]

    decoy_paths_json = json.dumps([str(p) for p in decoy_pdbs])

    py_code = f"""
import os, sys, json, time
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
sys.path.insert(0, '/home/ubuntu/proteina/af2rank_evaluation')

from af2rank_scorer import ModernAF2Rank, suppress_stdout

os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/params')

decoy_pdbs = json.loads('{decoy_paths_json}')

with suppress_stdout():
    scorer = ModernAF2Rank({REFERENCE_CIF!r}, chain={CHAIN!r}, model_name={model_name!r})

results = []
times = []
for pdb_path in decoy_pdbs:
    fname = os.path.basename(pdb_path)
    t0 = time.time()
    with suppress_stdout():
        scores = scorer.score_structure(pdb_path, decoy_chain="A", recycles={RECYCLES})
    elapsed = time.time() - t0
    times.append(elapsed)
    row = {{
        "file": fname,
        "ptm": float(scores["ptm"]),
        "plddt": float(scores["plddt"]),
        "elapsed": elapsed,
    }}
    results.append(row)

total = sum(times)
avg = total / len(times)
print(f"\\nTotal: {{total:.2f}}s, Average: {{avg:.2f}}s/structure")
print(json.dumps(results))
"""

    print(f"\n{'='*70}")
    print(f"ColabDesign scoring ({model_name}) - {len(decoy_pdbs)} decoys")
    print(f"{'='*70}")

    wrapper = "/home/ubuntu/proteina/af2rank_evaluation/run_with_colabdesign_env.sh"
    colabdesign_python = "/home/ubuntu/miniforge3/envs/colabdesign/bin/python"
    result = subprocess.run(
        [wrapper, colabdesign_python, "-c", py_code],
        cwd="/home/ubuntu/proteina/af2rank_evaluation",
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"ColabDesign STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ColabDesign scoring failed (rc={result.returncode})")

    # Parse output
    times = []
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("["):
            results = json.loads(line)
            for r in results:
                print(f"  {r['file']}: {r['elapsed']:.2f}s  pTM={r['ptm']:.4f}  pLDDT={r['plddt']:.4f}")
                times.append(r['elapsed'])
            break

    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    return {"times": times, "total": total_time, "average": avg_time}


def main():
    print(f"\nBenchmarking AF2Rank: OpenFold vs ColabDesign")
    print(f"Model: model_1_ptm, {NUM_DECOYS} decoys, {RECYCLES} recycles")

    results = {}

    # ColabDesign baseline
    try:
        results["colabdesign"] = run_colabdesign_benchmark("model_1_ptm")
    except Exception as e:
        print(f"ColabDesign benchmark failed: {e}")
        results["colabdesign"] = None

    # OpenFold with default chunking
    try:
        results["openfold_default"] = run_openfold_benchmark("model_1_ptm", chunk_size=None)
    except Exception as e:
        print(f"OpenFold default chunking failed: {e}")
        results["openfold_default"] = None

    # OpenFold with chunking disabled
    try:
        results["openfold_no_chunk"] = run_openfold_benchmark("model_1_ptm", chunk_size=0)
    except Exception as e:
        print(f"OpenFold no-chunk failed: {e}")
        results["openfold_no_chunk"] = None

    # Summary
    print(f"\n{'='*70}")
    print("SPEED COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'Total (s)':>12} {'Per-struct (s)':>15}")
    print("-" * 70)

    for method, res in results.items():
        if res:
            print(f"{method:<30} {res['total']:>12.2f} {res['average']:>15.2f}")
        else:
            print(f"{method:<30} {'FAILED':>12} {'-':>15}")

    print(f"{'='*70}")

    # Compute speedups
    if results.get("colabdesign") and results.get("openfold_no_chunk"):
        cd_avg = results["colabdesign"]["average"]
        of_avg = results["openfold_no_chunk"]["average"]
        speedup = cd_avg / of_avg
        print(f"\nOpenFold (no-chunk) is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than ColabDesign")

    if results.get("openfold_default") and results.get("openfold_no_chunk"):
        of_default = results["openfold_default"]["average"]
        of_no_chunk = results["openfold_no_chunk"]["average"]
        speedup = of_default / of_no_chunk
        print(f"OpenFold (no-chunk) is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than OpenFold (default)")


if __name__ == "__main__":
    main()
