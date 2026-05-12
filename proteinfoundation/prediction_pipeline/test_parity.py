#!/usr/bin/env python3
"""
Parity test: Compare OpenFold vs ColabDesign AF2Rank scoring on the same decoys.
Run with: ./run_with_proteina_env.sh python test_parity.py

This script:
1. Scores 3 decoys with OpenFold (model_1_ptm and model_2_ptm)
2. Scores the same 3 decoys with ColabDesign (via subprocess)
3. Compares pTM, pLDDT, composite scores between backends
"""

import json
import os
import subprocess
import sys
import tempfile
import warnings

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from af2rank_openfold_scorer import OpenFoldAF2Rank

INFERENCE_DIR = (
    "/home/ubuntu/proteina/inference/"
    "inference_seq_cond_sampling_ca_beta-2.5-2.0_finetune-all_v1.4_default-fold_4-seq-S25_64-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise"
    "/1a2y_C"
)
REFERENCE_CIF = "/home/ubuntu/data/af2rank_single/pdb/a2/1a2y.cif"
CHAIN = "C"
NUM_DECOYS = 3
RECYCLES = 3


def run_openfold_scoring(model_name="model_1_ptm"):
    """Score decoys with OpenFold backend."""
    decoy_pdbs = sorted(
        [os.path.join(INFERENCE_DIR, f) for f in os.listdir(INFERENCE_DIR) if f.endswith(".pdb")]
    )[:NUM_DECOYS]

    print(f"\n{'='*60}")
    print(f"OpenFold scoring ({model_name}) - {len(decoy_pdbs)} decoys")
    print(f"{'='*60}")

    scorer = OpenFoldAF2Rank(
        REFERENCE_CIF, chain=CHAIN, model_name=model_name, recycles=RECYCLES,
    )

    results = []
    for pdb_path in decoy_pdbs:
        fname = os.path.basename(pdb_path)
        scores = scorer.score_structure(pdb_path, decoy_chain="A", recycles=RECYCLES)
        print(f"  {fname}: pTM={scores['ptm']:.4f}  pLDDT={scores['plddt']:.4f}  composite={scores['composite']:.4f}")
        results.append({"file": fname, **{k: v for k, v in scores.items() if k != "pred_coords"}})
    return results


def run_colabdesign_scoring(model_name="model_1_ptm"):
    """Score decoys with ColabDesign backend via subprocess."""
    decoy_pdbs = sorted(
        [os.path.join(INFERENCE_DIR, f) for f in os.listdir(INFERENCE_DIR) if f.endswith(".pdb")]
    )[:NUM_DECOYS]

    decoy_paths_json = json.dumps([str(p) for p in decoy_pdbs])

    py_code = f"""
import os, sys, json
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'
sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})

from af2rank_scorer import ModernAF2Rank, suppress_stdout

os.environ['ALPHAFOLD_DATA_DIR'] = os.path.expanduser('~/openfold/openfold/resources/params')

decoy_pdbs = json.loads('{decoy_paths_json}')

with suppress_stdout():
    scorer = ModernAF2Rank({REFERENCE_CIF!r}, chain={CHAIN!r}, model_name={model_name!r})

results = []
for pdb_path in decoy_pdbs:
    fname = os.path.basename(pdb_path)
    with suppress_stdout():
        scores = scorer.score_structure(pdb_path, decoy_chain="A", recycles={RECYCLES})
    row = {{
        "file": fname,
        "ptm": float(scores["ptm"]),
        "plddt": float(scores["plddt"]),
        "composite": float(scores["composite"]),
        "pae_mean": float(scores["pae_mean"]),
        "tm_ref_pred": float(scores.get("tm_ref_pred", 0)),
        "tm_ref_template": float(scores.get("tm_ref_template", 0)),
    }}
    results.append(row)

print(json.dumps(results))
"""

    print(f"\n{'='*60}")
    print(f"ColabDesign scoring ({model_name}) - {len(decoy_pdbs)} decoys")
    print(f"{'='*60}")

    wrapper = os.path.join(os.path.dirname(__file__), "run_with_colabdesign_env.sh")
    result = subprocess.run(
        [wrapper, "python", "-c", py_code],
        cwd=os.path.dirname(__file__),
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        print(f"ColabDesign STDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"ColabDesign scoring failed (rc={result.returncode})")

    # Extract JSON from last line of stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("["):
            results = json.loads(line)
            break
    else:
        print(f"ColabDesign STDOUT:\n{result.stdout[-2000:]}")
        raise RuntimeError("Could not parse ColabDesign output")

    for r in results:
        print(f"  {r['file']}: pTM={r['ptm']:.4f}  pLDDT={r['plddt']:.4f}  composite={r['composite']:.4f}")

    return results


def compare_results(of_results, cd_results, label=""):
    """Compare OpenFold vs ColabDesign results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON {label}")
    print(f"{'='*60}")
    print(f"{'File':<20} {'Metric':<12} {'OpenFold':>10} {'ColabDesign':>12} {'Diff':>8} {'OK?':>5}")
    print("-" * 70)

    all_ok = True
    metrics = ["ptm", "plddt", "composite", "pae_mean"]

    for of_r, cd_r in zip(of_results, cd_results):
        assert of_r["file"] == cd_r["file"], f"File mismatch: {of_r['file']} vs {cd_r['file']}"
        for metric in metrics:
            of_val = of_r[metric]
            cd_val = cd_r[metric]
            diff = abs(of_val - cd_val)
            threshold = 0.01 if metric in ("ptm", "plddt", "composite") else 1.0
            ok = diff < threshold
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"{of_r['file']:<20} {metric:<12} {of_val:>10.4f} {cd_val:>12.4f} {diff:>8.4f} {status:>5}")

    print("-" * 70)
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - differences exceed threshold")
    return all_ok


if __name__ == "__main__":
    # Test model_1_ptm
    of_m1 = run_openfold_scoring("model_1_ptm")
    cd_m1 = run_colabdesign_scoring("model_1_ptm")
    ok1 = compare_results(of_m1, cd_m1, label="model_1_ptm")

    # Test model_2_ptm
    of_m2 = run_openfold_scoring("model_2_ptm")
    cd_m2 = run_colabdesign_scoring("model_2_ptm")
    ok2 = compare_results(of_m2, cd_m2, label="model_2_ptm")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {'ALL PASS' if (ok1 and ok2) else 'SOME FAILURES'}")
    print(f"{'='*60}")

    sys.exit(0 if (ok1 and ok2) else 1)
