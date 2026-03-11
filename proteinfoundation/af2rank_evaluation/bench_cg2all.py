#!/usr/bin/env python3
"""Benchmark cg2all: CPU multiprocessing (via CLI) vs GPU batching (via Python API).

Tests on 20 CA-only decoy PDBs to measure throughput.
"""

import os
import sys
import time
import glob
import tempfile
import subprocess
from multiprocessing import Pool

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"

import torch
import dgl
import mdtraj

from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch
from cg2all.lib.libter import patch_termini
import cg2all.lib.libcg
import cg2all.lib.libmodel

INFERENCE_DIR = (
    "/home/ubuntu/proteina/inference/"
    "inference_seq_cond_sampling_ca_beta-2.5-2.0_finetune-all_v1.4_default-fold_4-seq-S25_64-eff-bs_purge-test_warmup_cutoff-190828_last_045-noise/1a2y_C"
)
CG2ALL_BIN = os.path.join(os.path.dirname(sys.executable), "convert_cg2all")
N_STRUCTURES = 20


def get_test_pdbs():
    pdbs = sorted(glob.glob(os.path.join(INFERENCE_DIR, "*.pdb")))[:N_STRUCTURES]
    assert len(pdbs) == N_STRUCTURES, f"Need {N_STRUCTURES} PDBs, found {len(pdbs)}"
    return pdbs


# ── Method 1: CPU multiprocessing via CLI ────────────────────────────────────

def _run_cli_single(args):
    pdb_path, out_path = args
    result = subprocess.run(
        [CG2ALL_BIN, "-p", pdb_path, "-o", out_path, "--cg", "CA", "--device", "cpu"],
        capture_output=True, text=True, timeout=120,
    )
    return result.returncode == 0


def bench_cpu_sequential(pdbs):
    """Sequential CPU via CLI (baseline)."""
    tmpdir = tempfile.mkdtemp(prefix="cg2all_seq_")
    work = [(p, os.path.join(tmpdir, f"{i}.pdb")) for i, p in enumerate(pdbs)]
    t0 = time.time()
    for args in work:
        _run_cli_single(args)
    elapsed = time.time() - t0
    # cleanup
    for _, out in work:
        if os.path.exists(out):
            os.unlink(out)
    os.rmdir(tmpdir)
    return elapsed


def bench_cpu_multiproc(pdbs, n_workers=4):
    """Parallel CPU via CLI with multiprocessing."""
    tmpdir = tempfile.mkdtemp(prefix="cg2all_mp_")
    work = [(p, os.path.join(tmpdir, f"{i}.pdb")) for i, p in enumerate(pdbs)]
    t0 = time.time()
    with Pool(n_workers) as pool:
        pool.map(_run_cli_single, work)
    elapsed = time.time() - t0
    for _, out in work:
        if os.path.exists(out):
            os.unlink(out)
    os.rmdir(tmpdir)
    return elapsed


# ── Method 2: GPU batching via Python API ────────────────────────────────────

def _load_cg2all_model(device="cuda"):
    """Load cg2all CalphaBasedModel once."""
    ckpt_path = MODEL_HOME / "CalphaBasedModel.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["hyper_parameters"]
    cg_model = cg2all.lib.libcg.CalphaBasedModel
    config = cg2all.lib.libmodel.set_model_config(config, cg_model, flattened=False)
    model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    return model, config, cg_model


def bench_gpu_sequential(pdbs, model, config, cg_model, device="cuda"):
    """Sequential GPU: load each PDB, forward one at a time."""
    tmpdir = tempfile.mkdtemp(prefix="cg2all_gpuseq_")
    t0 = time.time()
    for i, pdb_path in enumerate(pdbs):
        ds = PredictionData(pdb_path, cg_model, radius=config.globals.radius)
        loader = dgl.dataloading.GraphDataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
        batch = next(iter(loader)).to(device)
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        out_path = os.path.join(tmpdir, f"{i}.pdb")
        output = patch_termini(traj_s[0])
        output.save(out_path)
        os.unlink(out_path)
    elapsed = time.time() - t0
    os.rmdir(tmpdir)
    return elapsed


def bench_gpu_batched(pdbs, model, config, cg_model, device="cuda"):
    """GPU batched: load all PDBs as DGL graphs, batch them, single forward."""
    tmpdir = tempfile.mkdtemp(prefix="cg2all_gpubatch_")

    t0 = time.time()
    # Build all graphs
    graphs = []
    for pdb_path in pdbs:
        ds = PredictionData(pdb_path, cg_model, radius=config.globals.radius)
        g = ds[0]
        graphs.append(g)

    # Batch all graphs together
    batched = dgl.batch(graphs).to(device)

    # Single forward pass
    with torch.no_grad():
        R = model.forward(batched)[0]["R"]

    # Unbatch and save
    traj_s, ssbond_s = create_trajectory_from_batch(batched, R)
    for i, traj in enumerate(traj_s):
        out_path = os.path.join(tmpdir, f"{i}.pdb")
        output = patch_termini(traj)
        output.save(out_path)
        os.unlink(out_path)

    elapsed = time.time() - t0
    os.rmdir(tmpdir)
    return elapsed


def main():
    pdbs = get_test_pdbs()
    print(f"Benchmarking cg2all on {len(pdbs)} CA-only PDBs\n")

    # Load GPU model once (outside timing for fair comparison of per-batch cost)
    print("Loading cg2all model on GPU...")
    device = "cuda"
    model, config, cg_model = _load_cg2all_model(device)
    print("Model loaded.\n")

    # Warm up GPU
    ds = PredictionData(pdbs[0], cg_model, radius=config.globals.radius)
    loader = dgl.dataloading.GraphDataLoader(ds, batch_size=1, num_workers=0, shuffle=False)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        _ = model.forward(batch)
    torch.cuda.synchronize()
    print("GPU warmed up.\n")

    # ── Run benchmarks ──
    results = {}

    print("1) GPU batched (single forward pass)...")
    t = bench_gpu_batched(pdbs, model, config, cg_model, device)
    results["gpu_batched"] = t
    print(f"   {t:.3f}s ({t/len(pdbs)*1000:.1f} ms/structure)\n")

    print("2) GPU sequential (one forward per structure)...")
    t = bench_gpu_sequential(pdbs, model, config, cg_model, device)
    results["gpu_sequential"] = t
    print(f"   {t:.3f}s ({t/len(pdbs)*1000:.1f} ms/structure)\n")

    print("3) CPU sequential (CLI)...")
    t = bench_cpu_sequential(pdbs)
    results["cpu_sequential"] = t
    print(f"   {t:.3f}s ({t/len(pdbs)*1000:.1f} ms/structure)\n")

    for n_workers in [2, 4, 8]:
        label = f"cpu_{n_workers}workers"
        print(f"4) CPU multiprocessing (CLI, {n_workers} workers)...")
        t = bench_cpu_multiproc(pdbs, n_workers=n_workers)
        results[label] = t
        print(f"   {t:.3f}s ({t/len(pdbs)*1000:.1f} ms/structure)\n")

    # ── Summary ──
    print("=" * 60)
    print(f"{'Method':<35} {'Total (s)':>10} {'Per-struct (ms)':>15}")
    print("-" * 60)
    for method, t in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:<35} {t:>10.3f} {t/len(pdbs)*1000:>15.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
