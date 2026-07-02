"""Precompute compact artifacts for the ext_lig "smart unknown" training augmentation
(mask_ext_lig_blocky in proteinfoundation/utils/ff_utils/pdb_utils.py).

Consumes the full-corpus ext_lig distribution CSVs (163,106 real PDB training chains,
computed directly from the processed .pt files -- see
proteinfoundation/datasets/ext_lig_utils.py for the label definition):
  - per-chain CSV: one row per chain, column `frac_present` = fraction of that chain's
    residues with ext_lig==1 (present).
  - runs CSV: one row per contiguous ext_lig==1 run, column `run_length`.

Produces two small artifacts under $DATA_PATH/ext_lig_stats/:
  - frac_present.npy: float32[N_chains] -- the raw per-chain frac_present values, sampled
    directly (no parametric fit) to draw a realistic per-example "how much to hide" target.
  - runlen_buckets.npz: adaptive-width histogram of run lengths (bucket_lo, bucket_hi,
    bucket_weight int64/int64/float64 arrays). Raw run lengths collapse to only ~148
    distinct values with a sparse tail (single-digit RAW COUNTS past length ~100, out of
    1.8M runs total) -- resampling the raw histogram would reproduce that tail noise
    verbatim (e.g. length 227 exists but 226 doesn't, purely from finite-sample luck).
    Two DIFFERENT quantities matter here: RAW RUN COUNT is the sample-size/confidence
    signal that decides WHEN to merge lengths together, while BY-RESIDUE MASS
    (length * count, matching the "by-residue" -- a present residue's own patch length --
    framing) is what gets stored as each bucket's final sampling weight. Thresholding on
    mass instead of count barely merges anything, because a handful of rare-but-long runs
    (e.g. length 227, only 3 observations) get an inflated mass (227*3=681) purely from
    being long -- exactly the noise this bucketing exists to smooth over. So: walk lengths
    in increasing order, accumulate RAW COUNT until a bucket reaches MIN_BUCKET_COUNT
    observations, then close it (storing the accumulated by-residue mass as its weight); a
    length whose OWN raw count already exceeds MIN_BUCKET_COUNT gets its own single-length
    bucket. This leaves the well-observed head (short runs, thousands of observations
    each) at full length-1 resolution and substantially smooths the genuinely sparse tail,
    with no parametric family assumed. At sample time: pick a bucket by weight, then draw
    uniformly within [bucket_lo, bucket_hi].

Run on Engaging (DATA_PATH set), CPU-only:
    conda activate cue_openfold
    python script_utils/precompute_ext_lig_augment_stats.py
"""

import argparse
import os

import numpy as np
import pandas as pd

MIN_BUCKET_COUNT = 500  # raw run-count required to close a bucket (confidence threshold)


def build_frac_present_pool(per_chain_csv: str) -> np.ndarray:
    df = pd.read_csv(per_chain_csv)
    frac = df["frac_present"].to_numpy(dtype=np.float32)
    assert np.all((frac >= 0.0) & (frac <= 1.0)), "frac_present out of [0,1] range"
    return frac


def build_runlen_buckets(runs_csv: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two DIFFERENT quantities drive this on purpose:
      - RAW RUN COUNT decides WHEN to merge lengths together -- it's the sample-size /
        statistical-confidence signal (a length seen only 3 times is noisy regardless of
        how long it is).
      - BY-RESIDUE MASS (length * count) is what gets stored as the bucket's sampling
        weight -- it's the correct target quantity ("a present residue's own patch
        length"). Using by-residue mass for BOTH (i.e. thresholding on weight too) barely
        merges anything, because a handful of noisy long-but-rare runs (e.g. length 227,
        only 3 observations) get an inflated weight (227*3=681) purely from being long --
        exactly the noise this bucketing is meant to smooth over, not preserve.
    """
    df = pd.read_csv(runs_csv)
    counts_by_length = df["run_length"].value_counts().sort_index()
    lengths = counts_by_length.index.to_numpy(dtype=np.int64)
    counts = counts_by_length.to_numpy(dtype=np.int64)
    by_residue_mass = lengths * counts

    bucket_lo, bucket_hi, bucket_weight = [], [], []
    acc_start_idx = None
    acc_count = 0
    acc_weight = 0
    for i, (length, count, mass) in enumerate(zip(lengths, counts, by_residue_mass)):
        if acc_start_idx is None and count >= MIN_BUCKET_COUNT:
            bucket_lo.append(int(length))
            bucket_hi.append(int(length))
            bucket_weight.append(int(mass))
            continue
        if acc_start_idx is None:
            acc_start_idx = i
        acc_count += int(count)
        acc_weight += int(mass)
        if acc_count >= MIN_BUCKET_COUNT:
            bucket_lo.append(int(lengths[acc_start_idx]))
            bucket_hi.append(int(length))
            bucket_weight.append(acc_weight)
            acc_start_idx = None
            acc_count = 0
            acc_weight = 0
    if acc_start_idx is not None:
        # Leftover tail never reached the threshold -- emit it as its own (possibly
        # under-threshold) final bucket. Do NOT backward-merge into the previous bucket:
        # if that bucket happens to be a well-supported SINGLE length (e.g. length 100 with
        # thousands of observations), widening it to absorb the sparse leftover would
        # corrupt a fine-resolution bucket into a coarse one covering mostly-unobserved
        # lengths -- worse than just accepting one thin bucket at the extreme tail end.
        bucket_lo.append(int(lengths[acc_start_idx]))
        bucket_hi.append(int(lengths[-1]))
        bucket_weight.append(acc_weight)

    return (
        np.array(bucket_lo, dtype=np.int64),
        np.array(bucket_hi, dtype=np.int64),
        np.array(bucket_weight, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-chain-csv", default=None, help="default: ~/ext_lig_per_chain.csv")
    parser.add_argument("--runs-csv", default=None, help="default: ~/ext_lig_runs.csv")
    parser.add_argument("--out-dir", default=None, help="default: $DATA_PATH/ext_lig_stats")
    args = parser.parse_args()

    data_path = os.environ.get("DATA_PATH")
    per_chain_csv = args.per_chain_csv or os.path.expanduser("~/ext_lig_per_chain.csv")
    runs_csv = args.runs_csv or os.path.expanduser("~/ext_lig_runs.csv")
    if args.out_dir is None and data_path is None:
        parser.error("DATA_PATH env not set; pass --out-dir explicitly.")
    out_dir = args.out_dir or os.path.join(data_path, "ext_lig_stats")
    os.makedirs(out_dir, exist_ok=True)

    frac_present = build_frac_present_pool(per_chain_csv)
    frac_out = os.path.join(out_dir, "frac_present.npy")
    np.save(frac_out, frac_present)
    print(f"frac_present pool: {len(frac_present)} chains, mean={frac_present.mean():.4f} "
          f"median={np.median(frac_present):.4f} std={frac_present.std():.4f} -> {frac_out}")

    bucket_lo, bucket_hi, bucket_weight = build_runlen_buckets(runs_csv)
    runlen_out = os.path.join(out_dir, "runlen_buckets.npz")
    np.savez(runlen_out, bucket_lo=bucket_lo, bucket_hi=bucket_hi, bucket_weight=bucket_weight)
    n_raw_lengths = int(pd.read_csv(runs_csv)["run_length"].nunique())
    print(f"runlen buckets: {len(bucket_lo)} buckets (from {n_raw_lengths} raw distinct lengths), "
          f"range [{bucket_lo.min()}, {bucket_hi.max()}] -> {runlen_out}")
    print("bucket detail (lo, hi, by-residue weight):")
    for lo, hi, w in zip(bucket_lo, bucket_hi, bucket_weight):
        print(f"  [{lo:>4}, {hi:>4}]  weight={w:>10.0f}")


if __name__ == "__main__":
    main()
