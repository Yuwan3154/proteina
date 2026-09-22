"""Per-chain reference availability for the tri eval arms, via the REAL TopologyReferenceTransform.

mode=old  (run from the proteina_cmhier checkout, index = $DATA_PATH/pdb_train/topology_index.pt):
    the retrieved non-self reference tri_full384's nonself arm would get (seed 0, same call path as
    model_trainer_base), plus how many different-sequence, run-bearing cluster-mates exist.
mode=new  (run from the proteina_tri checkout, index = synth_index_v4/topology_index_cb8_synth.pt):
    the synthetic template tri_cb8synth_v5's nonself arm would get (tm_range 0.5-0.9, seed 0).

Every queried stem gets a row; a stem absent from the index is written as NOT_IN_INDEX, never dropped.
"""

import argparse
import sys

import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--mode", choices=("old", "new"), required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--csv", required=True, help="maxl384 dataselector csv (id, processed_length)")
ap.add_argument("--stems", required=True, nargs="+", help="files of chain ids (one per line)")
ap.add_argument("--splits_dir", required=True, help="dir with {train,val,test}_chain_ids.txt")
ap.add_argument("--out", required=True)
args = ap.parse_args()

sys.path.insert(0, args.repo)
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform  # noqa: E402

split_of = {}
for sp in ("train", "val", "test"):
    for s in open(f"{args.splits_dir}/{sp}_chain_ids.txt"):
        split_of[s.strip()] = sp
stems = sorted({ln.strip() for f in args.stems for ln in open(f) if ln.strip()})
df = pd.read_csv(args.csv, usecols=["id", "processed_length"])
length = dict(zip(df["id"].astype(str), df["processed_length"]))

kw = dict(index_path=args.index)
if args.mode == "new":
    kw.update(reference_source="synthetic", tm_range=(0.5, 0.9))
tr = TopologyReferenceTransform(**kw)
tr._ensure_loaded()
idx = tr._index
print(f"[index] {len(idx['ids'])} rows, {len(idx['members_offset']) - 1} groups, mode={args.mode}", flush=True)

rows = []
for st in stems:
    row = tr._id_to_row.get(st)
    L = length.get(st)
    rec = dict(stem=st, split384=split_of.get(st, "none"), length=L)
    if row is None or L != L or L is None:
        rec.update(status="NOT_IN_INDEX" if row is None else "NO_LENGTH")
        rows.append(rec)
        continue
    cl = int(idx["cluster_of"][row])
    lo, hi = int(idx["members_offset"][cl]), int(idx["members_offset"][cl + 1])
    members = idx["members_flat"][lo:hi].long()
    if args.mode == "old":
        cand = members[idx["seq_hash"][members] != idx["seq_hash"][row]]
        cand = [int(c) for c in cand if tr._runs_for(int(c))]
        rec.update(n_group=int(members.numel()), n_valid_cand=len(cand),
                   cand_train=sum(split_of.get(str(idx["ids"][c])) == "train" for c in cand),
                   cand_val=sum(split_of.get(str(idx["ids"][c])) == "val" for c in cand),
                   cand_none=sum(str(idx["ids"][c]) not in split_of for c in cand))
    else:
        tm = idx["row_tm"][members].float()
        rec.update(n_group=int(members.numel()), n_valid_cand=int(((tm >= 0.5) & (tm <= 0.9)).sum()))
    got = tr.nonself_reference(st, int(L), seed=0)
    ref = got[1] if got is not None else "NONE"
    rec.update(status="OK" if got is not None else "NO_REF", ref=ref,
               ref_split384=split_of.get(ref, "none") if got is not None else "")
    rows.append(rec)

out = pd.DataFrame(rows)
out.to_csv(args.out, sep="\t", index=False)
print(out["status"].value_counts().to_string(), flush=True)
print(out.groupby(["split384", "status"]).size().to_string(), flush=True)
if args.mode == "old":
    print("[ref split] " + out.loc[out.status == "OK", "ref_split384"].value_counts().to_string(), flush=True)
print(f"[rows] {len(out)} of {len(stems)} stems -> {args.out}", flush=True)
