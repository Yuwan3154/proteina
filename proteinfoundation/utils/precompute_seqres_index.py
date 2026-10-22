"""SEQRES (mmCIF label_seq_id) position of every modelled residue, as a side table keyed by stem (T8b, user 2026-10-20:
"SEQRES position (AF2)"; "We should always use the standardized mmcif numbering").

Processed graphs keep only auth numbering (residue_pdb_idx) and array order; label_seq_id is read by
graphein_utils.read_pdb_to_dataframe and dropped. This re-reads each raw mmCIF exactly as processing did (same biopandas
route, label_asym_id as chain id) and maps every graph residue_id "chain:resname:authnum:ins" to its label_seq_id.

Per chain status (never silent): ok; nonmonotone (label_seq_id not strictly increasing along array order); merged (one graph
residue_id covers several label_seq_id -- biopandas 0.5.1 drops insertion codes, so e.g. 82/82A/82B of one resname collapse);
collide (two graph residues on one label_seq_id, e.g. microheterogeneity); no_label_seq (an ATOM row without label_seq_id);
unmapped (a residue_id with no ATOM row); no_cif. Positions are stored for ok and nonmonotone only.

Usage:
  part:  python -m proteinfoundation.utils.precompute_seqres_index part  --pack PACK --data-dir DATA_PATH/pdb_train \
             --stems STEMS.txt --part K --n-parts N --out-dir PARTS_DIR
  merge: python -m proteinfoundation.utils.precompute_seqres_index merge --stems STEMS.txt --parts-dir PARTS_DIR --out OUT.pt
"""

import argparse
import collections
import io
import json
import mmap
import pathlib
import time

import biopandas
import numpy as np
import pandas as pd
import torch
from biopandas.mmcif import PandasMmcif

from proteinfoundation.datasets.pdb_data import _resolve_raw_sharded

DEFINITION = ("label_seq_id (mmCIF _atom_site, 1-based SEQRES position) of each modelled residue in array order, mapped "
              "via residue_id = label_asym_id:resname:auth_seq_id:ins from the same biopandas read as processing")
STATUSES = ("ok", "nonmonotone", "merged", "collide", "no_label_seq", "unmapped", "no_cif")


def residue_label_seq(path: str) -> tuple:
    """(residue_id -> set of label_seq_id (None if absent), residue_id -> set of real mmCIF insertion codes) over ATOM records,
    keyed like read_pdb_to_dataframe + protein_to_pyg build residue_id (biopandas 0.5.1 blanks the insertion code there)."""
    pmcif = PandasMmcif().read_mmcif(path).get_model(1)
    label_asym, label_seq, ins_code = [], [], []
    for rec in ("ATOM", "HETATM"):
        if rec in pmcif.df and len(pmcif.df[rec]) > 0:
            label_asym.append(pmcif.df[rec]["label_asym_id"].values)
            label_seq.append(pmcif.df[rec]["label_seq_id"].values)
            ins_code.append(pmcif.df[rec]["pdbx_PDB_ins_code"].values)
    conv = pmcif.convert_to_pandas_pdb()
    df = pd.concat([conv.df[r] for r in ("ATOM", "HETATM") if r in conv.df and len(conv.df[r]) > 0]).reset_index(drop=True)
    df["chain_id"] = np.concatenate(label_asym)
    df["label_seq_id"] = np.concatenate(label_seq)
    df["ins_code"] = np.concatenate(ins_code)
    df = df.loc[df.record_name == "ATOM"]
    rid = df["chain_id"] + ":" + df["residue_name"] + ":" + df["residue_number"].astype(str) + ":" + df["insertion"]
    out, ins = collections.defaultdict(set), collections.defaultdict(set)
    for r, s, c in zip(rid.values, df["label_seq_id"].values, df["ins_code"].values):
        out[r].add(None if s is None else int(s))
        # biopandas 0.5.1 maps '?'/'.' to None, then astype(str) turns that into the string 'None'
        ins[r].add("" if c is None or c in ("?", ".", "None") else str(c))
    return out, ins


def chain_positions(residue_ids, mapping) -> tuple:
    """(status, int32 positions or None, number of merged residue_ids) for one chain's residue_id list."""
    sets = [mapping.get(r) for r in residue_ids]
    n_merged = sum(1 for s in sets if s and len(s) > 1)
    if any(not s for s in sets):
        return "unmapped", None, n_merged
    if any(None in s for s in sets):
        return "no_label_seq", None, n_merged
    if n_merged:
        return "merged", None, n_merged
    pos = np.asarray([next(iter(s)) for s in sets], dtype=np.int64)
    if len(np.unique(pos)) != len(pos):
        return "collide", None, 0
    status = "ok" if bool((np.diff(pos) > 0).all()) else "nonmonotone"
    return status, pos.astype(np.int32), 0


def cmd_part(a):
    stems = sorted({l.split()[0] for l in open(a.stems) if l.strip()})
    by_pdb = collections.defaultdict(list)
    for s in stems:
        by_pdb[s.split("_")[0]].append(s)
    pdbs = sorted(by_pdb)[a.part::a.n_parts]
    idx = np.load(a.pack + ".idx.npz", allow_pickle=True)  # stems are an object array (build_pack.py)
    ent = {str(s): (int(o), int(n)) for s, o, n in zip(idx["stems"], idx["offsets"], idx["lengths"])}
    data_dir = pathlib.Path(a.data_dir)
    man = json.load(open(data_dir / "shard_manifest.json"))
    raw_dir = data_dir / "raw"
    fh = open(a.pack, "rb")
    mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    rows = []
    t0 = time.time()
    for k, pdb in enumerate(pdbs):
        path = _resolve_raw_sharded(raw_dir, pdb.lower(), "cif", man) or _resolve_raw_sharded(raw_dir, pdb, "cif", man)
        mapping, ins = residue_label_seq(str(path)) if path is not None else (None, None)
        for stem in by_pdb[pdb]:
            assert stem in ent, f"{stem} not in pack"
            off, n = ent[stem]
            g = torch.load(io.BytesIO(mm[off:off + n]), map_location="cpu", weights_only=False)
            rids = list(g["residue_id"])
            assert len(rids) == int(g["coords"].shape[0]), (stem, len(rids), tuple(g["coords"].shape))
            if mapping is None:
                status, pos, n_merged = "no_cif", None, 0
            else:
                status, pos, n_merged = chain_positions(rids, mapping)
            n_gaps = int((np.diff(pos) > 1).sum()) if status == "ok" else -1
            n_ins = sum(1 for r in rids if any(ins.get(r, ()))) if mapping is not None else -1
            rows.append({"stem": stem, "status": status, "L": len(rids), "pos": pos, "n_gaps": n_gaps,
                         "n_merged": n_merged, "n_ins": n_ins})
        if (k + 1) % 200 == 0:
            print(f"[part {a.part}] {k + 1}/{len(pdbs)} entries {time.time() - t0:.0f} s", flush=True)
    out = pathlib.Path(a.out_dir) / f"part_{a.part:04d}_of_{a.n_parts:04d}.pt"
    torch.save({"rows": rows, "n_entries": len(pdbs), "seconds": time.time() - t0}, out)
    c = collections.Counter(r["status"] for r in rows)
    print(f"[part {a.part}] DONE {len(pdbs)} entries, {len(rows)} chains, {time.time() - t0:.0f} s, status {dict(c)} -> {out}")


def cmd_merge(a):
    stems = sorted({l.split()[0] for l in open(a.stems) if l.strip()})
    parts = sorted(pathlib.Path(a.parts_dir).glob("part_*_of_*.pt"))
    n_parts = {p.name.split("_of_")[1] for p in parts}
    assert len(n_parts) == 1, f"parts from different splits: {n_parts}"
    assert len(parts) == int(next(iter(n_parts))[:-3]), f"{len(parts)} part files, expected {next(iter(n_parts))[:-3]}"
    rows = [r for p in parts for r in torch.load(p, weights_only=False)["rows"]]
    got = [r["stem"] for r in rows]
    assert len(got) == len(set(got)), "a stem appears in two parts"
    assert sorted(got) == stems, f"coverage: {len(set(stems) - set(got))} missing, {len(set(got) - set(stems))} extra"
    rows.sort(key=lambda r: r["stem"])
    offset = np.zeros(len(rows) + 1, dtype=np.int64)
    flat = []
    for i, r in enumerate(rows):
        n = len(r["pos"]) if r["pos"] is not None else 0
        offset[i + 1] = offset[i] + n
        if n:
            flat.append(r["pos"])
    c = collections.Counter(r["status"] for r in rows)
    with_pos = [r for r in rows if r["pos"] is not None]
    audit = {
        "chains": len(rows),
        "status": {s: int(c.get(s, 0)) for s in STATUSES},
        "gapped_ok": int(sum(r["n_gaps"] > 0 for r in with_pos)),
        "merged_residue_ids": int(sum(r["n_merged"] for r in rows)),
        "ok_with_insertion_codes": int(sum(r["status"] == "ok" and r["n_ins"] > 0 for r in rows)),
        "biopandas": biopandas.__version__,
        "pandas": pd.__version__,
    }
    torch.save({
        "def": DEFINITION,
        "stems": [r["stem"] for r in rows],
        "status": [r["status"] for r in rows],
        "offset": torch.from_numpy(offset),
        "label_seq_flat": torch.from_numpy(np.concatenate(flat) if flat else np.zeros(0, dtype=np.int32)),
        "length": torch.tensor([r["L"] for r in rows], dtype=torch.int32),
        "audit": audit,
    }, a.out)
    with open(a.out + ".audit.tsv", "w") as f:
        f.write("stem\tstatus\tL\tn_gaps\tn_merged\tn_ins\n")
        for r in rows:
            f.write(f"{r['stem']}\t{r['status']}\t{r['L']}\t{r['n_gaps']}\t{r['n_merged']}\t{r['n_ins']}\n")
    print(f"[merge] {len(rows)} chains -> {a.out}; audit {audit}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("part")
    p.add_argument("--pack", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--stems", required=True)
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--n-parts", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--stems", required=True)
    m.add_argument("--parts-dir", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"part": cmd_part, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
