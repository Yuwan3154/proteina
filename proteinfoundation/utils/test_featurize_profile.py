#!/usr/bin/env python
"""Profile the OpenFold featurization pipeline to find bottlenecks.

Times each stage of build_batch() separately:
  1. Raw feature construction (template stub or mmcif parsing + kalign alignment)
  2. Feature pipeline processing (data transforms, MSA sampling, etc.)
  3. GPU transfer
  4. Template injection + mask overrides
  5. Model forward pass (for comparison)

Usage:
    python -u test_featurize_profile.py [--seq_len 100] [--with-template] [--n_repeats 5]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/ubuntu/proteina")
sys.path.insert(0, "/home/ubuntu/openfold")

import openfold.np.residue_constants as rc
from openfold.data.feature_pipeline import FeaturePipeline
from openfold.data.data_pipeline import (
    make_dummy_msa_feats,
    make_sequence_features_with_custom_template,
    make_sequence_features,
)
from openfold.utils.tensor_utils import tensor_tree_map
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


def p(*args, **kwargs):
    print(*args, flush=True, **kwargs)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_dummy_inputs(seq_len, device=torch.device("cpu"), sequence=None):
    """Create dummy inputs. If sequence is given, use that instead of random."""
    if sequence is not None:
        # Convert sequence string to residue indices
        mapping = {aa: i for i, aa in enumerate(rc.restypes)}
        mapping["X"] = rc.restype_num
        idxs = [mapping.get(aa, rc.restype_num) for aa in sequence[:seq_len]]
        # Pad if needed
        while len(idxs) < seq_len:
            idxs.append(rc.restype_num)
        residue_type = torch.tensor([idxs], dtype=torch.long, device=device)
    else:
        residue_type = torch.randint(0, 20, (1, seq_len), device=device)
    mask = torch.ones(1, seq_len, dtype=torch.float32, device=device)
    distogram_probs = torch.randn(1, seq_len, seq_len, 39, device=device).softmax(dim=-1)
    return distogram_probs, residue_type, mask

# 3zee chain A sequence (84 residues)
TEMPLATE_3ZEE_SEQ = "SEFKVTVCFGRTRVVVPCGDGRMKVFSLIQQAVTRYRKAVAKDPNYWIQVHRLEHGDGGILDLDDILCDVADDKDRLVAVFDEQ"


def profile_build_batch_stages(infer, seq_len, n_repeats=5,
                                template_mmcif_path=None,
                                template_chain_id=None,
                                kalign_binary_path=None,
                                sequence=None):
    """Profile each stage of build_batch individually."""
    device = infer.device
    dgram, rtype, mask = make_dummy_inputs(seq_len, device=device, sequence=sequence)

    # Pre-compute shared values
    l = seq_len  # mask is all ones
    seq = infer._restype_idx_to_str(rtype[0])
    mask_np = np.ones(l, dtype=np.float32)

    use_custom_template = template_mmcif_path is not None

    # ---- Stage 1: Raw feature construction ----
    times_raw = []
    for _ in range(n_repeats):
        t0 = time.time()
        if use_custom_template:
            pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
            raw = make_sequence_features_with_custom_template(
                sequence=seq,
                mmcif_path=template_mmcif_path,
                pdb_id=pdb_id,
                chain_id=template_chain_id,
                kalign_binary_path=kalign_binary_path,
                rm_template_sequence=False,
                skip_alignment=infer.skip_template_alignment,
            )
        else:
            raw = infer._make_template_stub_features(
                sequence=seq,
                mask=mask_np,
                rm_template_sequence=False,
            )
        times_raw.append(time.time() - t0)

    # ---- Stage 2: Feature pipeline ----
    times_pipeline = []
    for _ in range(n_repeats):
        # Re-create raw each time since process_features may mutate
        if use_custom_template:
            pdb_id = os.path.splitext(os.path.basename(template_mmcif_path))[0]
            raw = make_sequence_features_with_custom_template(
                sequence=seq,
                mmcif_path=template_mmcif_path,
                pdb_id=pdb_id,
                chain_id=template_chain_id,
                kalign_binary_path=kalign_binary_path,
                rm_template_sequence=False,
                skip_alignment=infer.skip_template_alignment,
            )
        else:
            raw = infer._make_template_stub_features(
                sequence=seq,
                mask=mask_np,
                rm_template_sequence=False,
            )
        t0 = time.time()
        feats = infer.feature_pipeline.process_features(raw, mode="predict", is_multimer=False)
        times_pipeline.append(time.time() - t0)

    # ---- Stage 3: GPU transfer ----
    times_transfer = []
    for _ in range(n_repeats):
        t0 = time.time()
        batch = tensor_tree_map(lambda x: x.to(device), feats)
        _sync()
        times_transfer.append(time.time() - t0)

    # ---- Stage 4: Template injection + overrides ----
    times_inject = []
    for _ in range(n_repeats):
        batch_copy = {k: v.clone() for k, v in batch.items()}
        t0 = time.time()
        infer._inject_template_dgram_probs(batch_copy, dgram[:, :l, :l, :])
        infer._apply_template_overrides(
            batch_copy,
            mask_template_aatype=False,
            zero_template_torsion_angles=True,
        )
        times_inject.append(time.time() - t0)

    # ---- Stage 5: Model forward ----
    # Full build_batch for model input
    full_batch = infer.build_batch(dgram, rtype, mask)
    # Warmup
    with torch.no_grad():
        infer.model(copy.deepcopy(full_batch))
    _sync()

    times_model = []
    for _ in range(n_repeats):
        _sync()
        t0 = time.time()
        with torch.no_grad():
            infer.model(copy.deepcopy(full_batch))
        _sync()
        times_model.append(time.time() - t0)

    # ---- Stage 6: Full end-to-end (build_batch + forward) ----
    times_e2e = []
    for _ in range(n_repeats):
        _sync()
        t0 = time.time()
        out = infer(dgram, rtype, mask)
        _sync()
        times_e2e.append(time.time() - t0)

    return {
        "raw_features": times_raw,
        "feature_pipeline": times_pipeline,
        "gpu_transfer": times_transfer,
        "template_inject": times_inject,
        "model_forward": times_model,
        "end_to_end": times_e2e,
    }


def print_results(results, label=""):
    if label:
        p(f"\n{'='*60}")
        p(f"  {label}")
        p(f"{'='*60}")

    total_featurize = 0
    for stage, times in results.items():
        avg = sum(times) / len(times)
        mn = min(times)
        mx = max(times)
        if stage != "end_to_end" and stage != "model_forward":
            total_featurize += avg
        p(f"  {stage:20s}: avg={avg:.4f}s  min={mn:.4f}s  max={mx:.4f}s")

    model_avg = sum(results["model_forward"]) / len(results["model_forward"])
    e2e_avg = sum(results["end_to_end"]) / len(results["end_to_end"])
    p(f"  {'---':20s}")
    p(f"  {'featurize_total':20s}: {total_featurize:.4f}s  ({total_featurize/(total_featurize+model_avg)*100:.1f}% of featurize+model)")
    p(f"  {'model_forward':20s}: {model_avg:.4f}s  ({model_avg/(total_featurize+model_avg)*100:.1f}% of featurize+model)")
    p(f"  {'end_to_end':20s}: {e2e_avg:.4f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--with-template", action="store_true",
                        help="Also profile with real mmcif template (kalign alignment)")
    parser.add_argument("--mmcif_path", type=str,
                        default="/home/ubuntu/openfold/tests/test_data/mmcifs/3zee.cif")
    parser.add_argument("--chain_id", type=str, default="A")
    parser.add_argument("--kalign_binary", type=str, default="/usr/bin/kalign")
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--max_recycling_iters", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="model_1_ptm")
    parser.add_argument("--jax_params", type=str,
                        default=os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Device: {device}")
    p(f"Loading model: {args.model_name}...")

    infer = OpenFoldTemplateInference(
        model_name=args.model_name,
        jax_params_path=args.jax_params,
        device=device,
        max_recycling_iters=args.max_recycling_iters,
        use_mlm=False,
    )
    p("Model loaded.\n")

    # Profile distogram-only mode (stub template)
    for sl in args.seq_len:
        results = profile_build_batch_stages(infer, sl, n_repeats=args.n_repeats)
        print_results(results, label=f"Distogram-only template, seq_len={sl}")

    # Profile with real mmcif template (use the template's actual sequence length)
    if args.with_template:
        if not os.path.exists(args.mmcif_path):
            p(f"\nWARNING: mmcif file not found: {args.mmcif_path}, skipping template profile")
            return

        # 3zee chain A is 84 residues; use matching sequence for alignment
        template_seq_len = len(TEMPLATE_3ZEE_SEQ)
        results = profile_build_batch_stages(
            infer, template_seq_len, n_repeats=args.n_repeats,
            template_mmcif_path=args.mmcif_path,
            template_chain_id=args.chain_id,
            kalign_binary_path=args.kalign_binary,
            sequence=TEMPLATE_3ZEE_SEQ,
        )
        print_results(results, label=f"Real template ({args.mmcif_path}), seq_len={template_seq_len}")

    # Profile batched featurization
    p(f"\n{'='*60}")
    p(f"  Batched featurization (4 samples)")
    p(f"{'='*60}")
    for sl in args.seq_len:
        n_samples = 4
        dgrams, rtypes, masks = [], [], []
        for _ in range(n_samples):
            d, r, m = make_dummy_inputs(sl, device=device)
            dgrams.append(d)
            rtypes.append(r)
            masks.append(m)

        # Time build_batch_multi (featurization only)
        times_multi_feat = []
        for _ in range(args.n_repeats):
            t0 = time.time()
            batch = infer.build_batch_multi(dgrams, rtypes, masks)
            times_multi_feat.append(time.time() - t0)

        # Time sequential build_batch
        times_seq_feat = []
        for _ in range(args.n_repeats):
            t0 = time.time()
            for i in range(n_samples):
                infer.build_batch(dgrams[i], rtypes[i], masks[i])
            times_seq_feat.append(time.time() - t0)

        avg_multi = sum(times_multi_feat) / len(times_multi_feat)
        avg_seq = sum(times_seq_feat) / len(times_seq_feat)
        p(f"  seq_len={sl}, {n_samples} samples:")
        p(f"    Sequential build_batch:  avg={avg_seq:.4f}s  ({avg_seq/n_samples:.4f}s/sample)")
        p(f"    build_batch_multi:       avg={avg_multi:.4f}s  ({avg_multi/n_samples:.4f}s/sample)")

    p("\n=== PROFILING COMPLETE ===")


if __name__ == "__main__":
    main()
