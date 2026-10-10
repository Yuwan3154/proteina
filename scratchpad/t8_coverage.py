"""T8 coverage check over the split's chains (user 2026-10-10: "run the coverage check"). Read-only.

scan:    one row per chain of a slice: processed .pt present, ConFind map present with shape [L, L], backbone completeness
         (N, CA, C, O from coord_mask, on-disk ATOM_NUMBERING), and stored DSSP labels that sit on incomplete-backbone residues (which
         precompute_dssp_targets' own rule says must be -1).
summary: merge the slice tables; counts per split and old/new; writes the Ca-only and no-ConFind lists.
Usage: python t8_coverage.py scan CHAINS.txt DATA_DIR TASK NTASK OUT.tsv
       python t8_coverage.py summary SPLIT_DIR TSV_GLOB OUT_DIR
"""

import glob
import json
import os
import pathlib
import sys

import pandas as pd
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded

N_, CA_, C_, O_ = 0, 1, 2, 3  # ATOM_NUMBERING (on-disk .pt order; atom37 reorder happens only at load)
COLS = ["stem", "pt", "L", "confind", "n_ca", "n_bb", "n_ca_only", "dssp", "n_lab_on_incomplete"]


def scan(chains, data_dir, task, ntask, out):
    stems = [l.strip() for l in open(chains) if l.strip()][int(task)::int(ntask)]
    d = pathlib.Path(data_dir)
    man = json.load(open(d / "shard_manifest.json"))
    with open(out + ".partial", "w") as f:
        f.write("\t".join(COLS) + "\n")
        for i, s in enumerate(stems):
            p = _processed_path_sharded(d / "processed", s, man)
            if not p.exists():
                f.write(f"{s}\t0\t" + "\t".join(["-1"] * (len(COLS) - 2)) + "\n")
                continue
            g = torch.load(p, map_location="cpu", weights_only=False)
            m = g.coord_mask.bool()
            L = int(m.shape[0])
            cm = getattr(g, "contact_map_confind", None)
            has_cf = int(cm is not None and tuple(cm.shape) == (L, L))
            bb = m[:, N_] & m[:, CA_] & m[:, C_] & m[:, O_]
            ca_only = m[:, CA_] & ~m[:, N_] & ~m[:, C_] & ~m[:, O_]
            dssp = getattr(g, "dssp_target", None)
            assert dssp is None or tuple(dssp.shape) == (L,), f"{s}: dssp_target {tuple(dssp.shape)} vs L={L}"
            n_bad = int(((dssp >= 0) & ~bb).sum()) if dssp is not None else -1
            f.write(f"{s}\t1\t{L}\t{has_cf}\t{int(m[:, CA_].sum())}\t{int(bb.sum())}\t{int(ca_only.sum())}\t"
                    f"{int(dssp is not None)}\t{n_bad}\n")
            if (i + 1) % 2000 == 0:
                print(f"[scan {task}] {i + 1}/{len(stems)}", flush=True)
    os.replace(out + ".partial", out)
    print(f"[scan {task}] done {len(stems)} -> {out}")


def summary(split_dir, tsv_glob, out_dir):
    files = sorted(glob.glob(tsv_glob))
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    want = [l.strip() for l in open(os.path.join(split_dir, "chains_for_templates.txt")) if l.strip()]
    assert len(df) == len(want) and set(df.stem) == set(want), f"{len(df)} rows from {len(files)} files vs {len(want)} chains"
    split = {}
    for s in ("train", "val", "test"):
        for c in open(os.path.join(split_dir, f"{s}_chain_ids.txt")):
            if c.strip():
                assert c.strip() not in split, f"{c.strip()} in two splits"
                split[c.strip()] = s
    df["split"] = df.stem.map(split)
    assert df.split.notna().all() and len(split) == len(df), f"{int(df.split.isna().sum())} chains without a split"
    have = df[df.pt == 1]
    ca_only = have[(have.n_bb == 0) & (have.n_ca > 0)]
    partial = have[(have.n_bb > 0) & (have.n_bb < have.n_ca)]
    no_cf = have[have.confind == 0]
    lines = [
        f"chains {len(df)} (files {len(files)})",
        f"processed .pt missing: {int((df.pt == 0).sum())}",
        f"ConFind map missing or wrong shape: {len(no_cf)} of {len(have)}",
        f"Ca-only (no residue with a full N/CA/C/O backbone): {len(ca_only)} of {len(have)}",
        f"partial backbone (some but not all CA residues complete): {len(partial)} chains, "
        f"{int((partial.n_ca - partial.n_bb).sum())} incomplete residues",
        f"stored DSSP labels on incomplete-backbone residues: {int(have.n_lab_on_incomplete.clip(lower=0).sum())} labels "
        f"in {int((have.n_lab_on_incomplete > 0).sum())} chains ({int((ca_only.n_lab_on_incomplete > 0).sum())} of them Ca-only)",
        f"no stored dssp_target: {int((have.dssp == 0).sum())}",
    ]
    for s in ("train", "val", "test"):
        sub = df[df.split == s]
        lines.append(f"  {s}: {len(sub)} chains; missing pt {int((sub.pt == 0).sum())}; no ConFind "
                     f"{int(((sub.pt == 1) & (sub.confind == 0)).sum())}; Ca-only {int(sub.stem.isin(ca_only.stem).sum())}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coverage_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for name, sub in (("ca_only", ca_only), ("no_confind", no_cf), ("missing_pt", df[df.pt == 0])):
        sub[["stem", "split", "L", "n_ca", "n_bb", "n_lab_on_incomplete"]].to_csv(
            os.path.join(out_dir, f"{name}.tsv"), sep="\t", index=False)
    df.to_csv(os.path.join(out_dir, "coverage_all.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        assert len(sys.argv) == 7, __doc__
        scan(*sys.argv[2:7])
    else:
        assert sys.argv[1] == "summary" and len(sys.argv) == 5, __doc__
        summary(*sys.argv[2:5])
