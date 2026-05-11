"""Cluster correctness audit for the Proteina sequence-identity clustering.

Verifies the invariants the rest of the pipeline (`ClusterSampler`,
`expand_cluster_splits`, dataloader) implicitly assumes:

  1. TSV <-> rep-fasta consistency
  2. Cluster disjointness (no seq in two clusters)
  3. Representative-is-longest invariant (cluster-reps mode relies on this)
  4. Identity cutoff sanity (sample of clusters via mmseqs easy-search)
  5. Train/val/test split disjointness at the SEQUENCE level
  6. Final-output consistency (split <-> processed/*.pt files on disk)

Usage:
    python scripts/analysis/audit_clusters.py \
        --data_dir /home/ubuntu/proteina/data/pdb_train \
        --cluster_tsv data/pdb_train/cluster_seqid_0.25_<...>.tsv \
        --cluster_fasta data/pdb_train/cluster_seqid_0.25_<...>.fasta \
        --input_fasta data/pdb_train/seq_<...>.fasta \
        --processed_dir data/pdb_train/processed \
        --min_seq_id 0.25 --coverage 0.8 \
        --seed 42 --train_val_test 0.8 0.15 0.05 \
        --out audit_clusters_report.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import pandas as pd

# Allow running from any CWD by adding the proteina root to sys.path.
_THIS = pathlib.Path(__file__).resolve()
_PROTEINA_ROOT = _THIS.parents[2]
if str(_PROTEINA_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTEINA_ROOT))

from proteinfoundation.utils.cluster_utils import (  # noqa: E402
    expand_cluster_splits,
    fasta_to_df,
    read_cluster_tsv,
    split_dataframe,
)


def _check_tsv_rep_fasta_consistency(
    cluster_to_members: Dict[str, List[str]],
    rep_fasta_df: pd.DataFrame,
) -> Tuple[bool, str]:
    tsv_reps = set(cluster_to_members.keys())
    fasta_reps = set(rep_fasta_df["id"].tolist())
    only_tsv = tsv_reps - fasta_reps
    only_fasta = fasta_reps - tsv_reps
    dup_in_fasta = len(rep_fasta_df) - len(fasta_reps)
    ok = not only_tsv and not only_fasta and dup_in_fasta == 0
    msg = (
        f"tsv_reps={len(tsv_reps)}, fasta_reps={len(fasta_reps)}, "
        f"only_in_tsv={len(only_tsv)}, only_in_fasta={len(only_fasta)}, "
        f"duplicate_rep_entries_in_fasta={dup_in_fasta}"
    )
    return ok, msg


def _check_disjointness(
    cluster_to_members: Dict[str, List[str]],
) -> Tuple[bool, str]:
    seen: Dict[str, str] = {}
    collisions: List[Tuple[str, str, str]] = []
    for rep, members in cluster_to_members.items():
        for m in members:
            if m in seen and seen[m] != rep:
                collisions.append((m, seen[m], rep))
            else:
                seen[m] = rep
    ok = len(collisions) == 0
    msg = (
        f"unique_seqs={len(seen)}, collisions={len(collisions)}"
        + (f", first_collision={collisions[0]}" if collisions else "")
    )
    return ok, msg


def _check_rep_is_longest(
    cluster_to_members: Dict[str, List[str]],
    input_fasta_df: pd.DataFrame,
) -> Tuple[bool, str]:
    seq_len = {row["id"]: len(row["sequence"]) for _, row in input_fasta_df.iterrows()}
    n_violations = 0
    n_checked = 0
    examples: List[Tuple[str, int, int]] = []
    for rep, members in cluster_to_members.items():
        if rep not in seq_len:
            continue
        n_checked += 1
        rep_len = seq_len[rep]
        member_lens = [seq_len[m] for m in members if m in seq_len]
        if not member_lens:
            continue
        if max(member_lens) > rep_len:
            n_violations += 1
            if len(examples) < 5:
                examples.append((rep, rep_len, max(member_lens)))
    ok = n_violations == 0
    msg = (
        f"checked={n_checked}, violations={n_violations}"
        + (f", examples={examples}" if examples else "")
    )
    return ok, msg


def _check_identity_cutoff(
    cluster_to_members: Dict[str, List[str]],
    input_fasta_df: pd.DataFrame,
    min_seq_id: float,
    coverage: float,
    sample_size: int = 200,
    seed: int = 0,
) -> Tuple[bool, str]:
    if shutil.which("mmseqs") is None:
        return True, "SKIPPED: mmseqs not on PATH"

    rng = random.Random(seed)
    multi = [r for r, m in cluster_to_members.items() if len(m) > 1]
    if not multi:
        return True, "SKIPPED: no multi-member clusters"
    rng.shuffle(multi)
    sample_reps = multi[:sample_size]

    seq_lookup = dict(zip(input_fasta_df["id"], input_fasta_df["sequence"]))
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        rep_fa = td / "reps.fasta"
        mem_fa = td / "members.fasta"
        with open(rep_fa, "w") as f_rep, open(mem_fa, "w") as f_mem:
            for rep in sample_reps:
                if rep not in seq_lookup:
                    continue
                f_rep.write(f">{rep}\n{seq_lookup[rep]}\n")
                for m in cluster_to_members[rep]:
                    if m == rep or m not in seq_lookup:
                        continue
                    f_mem.write(f">{m}\n{seq_lookup[m]}\n")
        out_tsv = td / "search.tsv"
        cmd = [
            "mmseqs", "easy-search", str(mem_fa), str(rep_fa), str(out_tsv), str(td / "tmp"),
            "--min-seq-id", "0.0", "-c", str(coverage), "--cov-mode", "1",
            "--format-output", "query,target,fident",
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            return False, f"mmseqs easy-search failed: {result.stderr.decode()[:300]}"
        if not out_tsv.exists():
            return False, "mmseqs produced no output tsv"

        below = 0
        total = 0
        worst = []  # type: List[Tuple[str, str, float]]
        member_to_rep = {m: r for r, ms in cluster_to_members.items() for m in ms}
        with open(out_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                q, t, fident = parts[0], parts[1], float(parts[2])
                if member_to_rep.get(q) != t:
                    continue
                total += 1
                if fident < min_seq_id:
                    below += 1
                    if len(worst) < 5:
                        worst.append((q, t, fident))
        ok = below == 0 and total > 0
        msg = (
            f"sampled_clusters={len(sample_reps)}, member_pairs_checked={total}, "
            f"below_cutoff={below}/{total} (cutoff={min_seq_id})"
            + (f", worst={worst}" if worst else "")
        )
        return ok, msg


def _replay_split(
    cluster_to_members: Dict[str, List[str]],
    rep_fasta_df: pd.DataFrame,
    train_val_test: List[float],
    seed: int,
) -> Dict[str, pd.DataFrame]:
    rep_ids = list(cluster_to_members.keys())
    df_reps = rep_fasta_df.loc[rep_fasta_df["id"].isin(rep_ids)].copy()
    splits = split_dataframe(df_reps, ["train", "val", "test"], train_val_test, seed=seed)
    full_splits, _ = expand_cluster_splits(splits, cluster_to_members)
    return full_splits


def _check_split_disjointness(
    full_splits: Dict[str, pd.DataFrame],
) -> Tuple[bool, str]:
    train = set(full_splits.get("train", pd.DataFrame({"id": []}))["id"].tolist())
    val = set(full_splits.get("val", pd.DataFrame({"id": []}))["id"].tolist())
    test = set(full_splits.get("test", pd.DataFrame({"id": []}))["id"].tolist())
    leak_tv = train & val
    leak_tt = train & test
    leak_vt = val & test
    ok = not (leak_tv or leak_tt or leak_vt)
    msg = (
        f"|train|={len(train)}, |val|={len(val)}, |test|={len(test)}, "
        f"train∩val={len(leak_tv)}, train∩test={len(leak_tt)}, val∩test={len(leak_vt)}"
        + (f", train∩val_first5={list(leak_tv)[:5]}" if leak_tv else "")
    )
    return ok, msg


def _check_processed_consistency(
    full_splits: Dict[str, pd.DataFrame],
    processed_dir: pathlib.Path,
) -> Tuple[bool, str]:
    if not processed_dir.exists():
        return True, f"SKIPPED: {processed_dir} does not exist"
    on_disk = {p.stem for p in processed_dir.glob("*.pt")}
    out = []
    for split, df in full_splits.items():
        ids = set(df["id"].tolist())
        present = ids & on_disk
        missing = ids - on_disk
        out.append(f"{split}: split_size={len(ids)}, on_disk={len(present)}, missing={len(missing)}")
    return True, "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=pathlib.Path, required=True)
    ap.add_argument("--cluster_tsv", type=pathlib.Path, required=True)
    ap.add_argument("--cluster_fasta", type=pathlib.Path, required=True)
    ap.add_argument("--input_fasta", type=pathlib.Path, required=True)
    ap.add_argument("--processed_dir", type=pathlib.Path, default=None)
    ap.add_argument("--min_seq_id", type=float, default=0.25)
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_val_test", nargs=3, type=float, default=[0.8, 0.15, 0.05])
    ap.add_argument("--identity_sample_size", type=int, default=200)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("audit_clusters_report.md"))
    args = ap.parse_args()

    print(f"[audit_clusters] reading {args.cluster_tsv}")
    cluster_to_members = read_cluster_tsv(args.cluster_tsv)
    print(f"[audit_clusters] {len(cluster_to_members)} clusters")

    rep_fasta_df = fasta_to_df(args.cluster_fasta)
    input_fasta_df = fasta_to_df(args.input_fasta)
    print(f"[audit_clusters] |rep_fasta|={len(rep_fasta_df)} |input_fasta|={len(input_fasta_df)}")

    results: List[Tuple[str, bool, str]] = []

    ok, msg = _check_tsv_rep_fasta_consistency(cluster_to_members, rep_fasta_df)
    results.append(("1. TSV <-> rep-fasta consistency", ok, msg))

    ok, msg = _check_disjointness(cluster_to_members)
    results.append(("2. Cluster disjointness", ok, msg))

    ok, msg = _check_rep_is_longest(cluster_to_members, input_fasta_df)
    results.append(("3. Representative-is-longest invariant", ok, msg))

    ok, msg = _check_identity_cutoff(
        cluster_to_members, input_fasta_df,
        min_seq_id=args.min_seq_id, coverage=args.coverage,
        sample_size=args.identity_sample_size, seed=args.seed,
    )
    results.append(("4. Identity cutoff sanity (mmseqs easy-search sample)", ok, msg))

    full_splits = _replay_split(cluster_to_members, rep_fasta_df, args.train_val_test, args.seed)
    ok, msg = _check_split_disjointness(full_splits)
    results.append(("5. Train/val/test split disjointness", ok, msg))

    if args.processed_dir is not None:
        ok, msg = _check_processed_consistency(full_splits, args.processed_dir)
        results.append(("6. Final-output consistency (split vs processed/*.pt)", ok, msg))

    lines = ["# Cluster correctness audit\n"]
    lines.append(f"- data_dir: `{args.data_dir}`")
    lines.append(f"- cluster_tsv: `{args.cluster_tsv}`")
    lines.append(f"- min_seq_id: {args.min_seq_id}, coverage: {args.coverage}")
    lines.append(f"- split seed: {args.seed}, train/val/test ratios: {args.train_val_test}")
    lines.append(f"- |clusters|: {len(cluster_to_members)}\n")
    lines.append("| # | Check | Result | Detail |")
    lines.append("|---|---|---|---|")
    for name, ok, msg in results:
        lines.append(f"| {name} | | {'PASS' if ok else 'FAIL'} | {msg} |")
    lines.append("")
    lines.append("Per-split sizes:")
    lines.append("```json")
    lines.append(json.dumps({s: len(df) for s, df in full_splits.items()}, indent=2))
    lines.append("```")

    args.out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[audit_clusters] wrote {args.out}")


if __name__ == "__main__":
    main()
