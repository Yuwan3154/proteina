"""Re-process the T8b SEQRES repair set with seqres_order=True into a sibling tree (repair plan S2; user 2026-10-21:
positions follow the standardized mmCIF numbering, real insertion codes kept, collide chains dropped, "over 512: Drop it").

Per stem: the real processing path (PDBLightningDataModule._process_single_pdb with seqres_order=True and the settings of
the stored files: store_het False, store_bfactor True, database_tag "pdb", no pre_transform / pre_filter) writes a staging
.pt; then the stored-file post-processing is replayed: ext_lig -> int8 (script_utils/backfill_ext_lig.py), dssp_target =
precompute_dssp_targets._compute_dssp (no proline mask), protein_id copied when the stored pack entry has it. The key set
and per-key dtypes are asserted equal to the stored entry's (minus contact_map_confind) and the graph is written atomically
to <out-root>/processed/<bucket>/<stem>.pt (bucket from --data-dir's shard_manifest.json).

contact_map_confind is left OUT (status column confind=pending, filled by the later GPU step): precompute_confind_maps
skips files that already carry the key, so a placeholder would survive that step silently, while a missing key fails
loudly (train.py guard / transforms.py "contact_map_confind is missing").

Status per stem: written | dropped_collide (side-table status collide; never processed) | dropped_over512 (new
processed_length = coord_mask.any(-1).sum() > 512, the T8 config max_length) | failed:process_returned_none (the entry
point logs the exception and returns None).
Sidecars (<out-root>/sidecars/part_K_of_N.pt), written stems only:
  label_seq  {stem: int32 mmCIF label_seq_id per new residue} (protein_to_pyg(residue_order="label_seq").label_seq_id)
  old_to_new {stem: {"index": int32 [L_old] new residue index (-1 = no match), "chimera": bool, "exact": bool}}: each old
    residue is matched on coordinates among the new residues sharing its chain:resname:authnum (insertion code
    stripped); winner = the candidate whose atom equals the old CA (first present atom when CA is missing); chimera =
    some present old atom is not the winner's (merged residue built from several); exact = winner's coords and
    coord_mask rows equal the old ones.
Usage: python -m proteinfoundation.utils.reprocess_seqres_chains --pack PACK --data-dir DATA_PATH/pdb_train \
           --audit SEQRES_AUDIT.tsv --stems STEMS.txt --part K --n-parts N --out-root OUT_ROOT
"""

import argparse
import collections
import io
import json
import mmap
import os
import pathlib
import time

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import PDBLightningDataModule, _processed_path_sharded, _resolve_raw_sharded
from proteinfoundation.graphein_utils.graphein_utils import protein_to_pyg, read_pdb_to_dataframe
from proteinfoundation.utils.precompute_dssp_targets import _compute_dssp

MAX_LEN = 512  # configs/datasets_config/pdb/pdb_train_contact-confind-synthtopo_S25_max512_t8.yaml max_length
REPROCESS = ("merged", "nonmonotone", "ok_ins")
COLUMNS = ("stem", "category", "status", "L_old", "L_new", "n_ins_new", "label_seq_increasing", "n_unmatched",
           "n_chimera", "n_inexact", "confind")


def load_audit(path) -> dict:
    """stem -> category (side-table status; ok with n_ins > 0 -> ok_ins)."""
    out = {}
    for line in list(open(path))[1:]:
        stem, status, _, _, _, n_ins = line.rstrip("\n").split("\t")
        out[stem] = "ok_ins" if status == "ok" and int(n_ins) > 0 else status
    return out


def open_pack(pack):
    idx = np.load(pack + ".idx.npz", allow_pickle=True)  # stems are an object array (build_pack.py)
    ent = {str(s): (int(o), int(n)) for s, o, n in zip(idx["stems"], idx["offsets"], idx["lengths"])}
    fh = open(pack, "rb")
    return ent, mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ), fh


def load_entry(ent, mm, stem):
    off, n = ent[stem]
    return torch.load(io.BytesIO(mm[off:off + n]), map_location="cpu", weights_only=False)


def schema(g) -> dict:
    return {k: str(v.dtype) if isinstance(v, torch.Tensor) else type(v).__name__ for k, v in g.items()}


def raw_path(raw_dir, pdb, man):
    return _resolve_raw_sharded(raw_dir, pdb.lower(), "cif", man) or _resolve_raw_sharded(raw_dir, pdb, "cif", man)


def reprocess_chain(stem, raw_dir, staging, man, seqres_order, old):
    """(status, graph or None): the real processing path for this one chain + the stored-file post-processing."""
    pdb, chain = stem.split("_", 1)
    fname = PDBLightningDataModule._process_single_pdb(
        (0, pdb, chain), raw_dir=raw_dir, processed_dir=staging, format_type="cif", store_het=False,
        store_bfactor=True, pre_transform=None, pre_filter=None, database_tag="pdb", shard_manifest=man,
        seqres_order=seqres_order)
    if fname is None:
        return "failed:process_returned_none", None
    assert fname == f"{stem}.pt", (fname, stem)
    staged = _processed_path_sharded(staging, stem, man)
    g = torch.load(staged, map_location="cpu", weights_only=False)
    os.remove(staged)
    g.ext_lig = g.ext_lig.to(torch.int8)
    g.dssp_target = _compute_dssp(g)
    if "protein_id" in old.keys():
        g.protein_id = old.protein_id
    return "processed", g


def label_seq_of(df, path, chain, g) -> torch.Tensor:
    """int32 label_seq_id per residue of g, from the same flag-on protein_to_pyg call processing made."""
    ls = protein_to_pyg(df=df.copy(), path=path, chain_selection=chain, keep_insertions=True, store_het=False,
                        store_bfactor=True, fill_value_coords=1e-5, residue_order="label_seq")
    assert list(ls.residue_id) == list(g.residue_id), "label_seq parse disagrees with the processed graph"
    return ls.label_seq_id.to(torch.int32)


def old_to_new(old, new) -> dict:
    """Coordinate match of every old residue to a new residue (see module docstring)."""
    cand = collections.defaultdict(list)
    for j, r in enumerate(new.residue_id):
        cand[r.rsplit(":", 1)[0]].append(j)
    n_old = len(old.residue_id)
    index = np.full(n_old, -1, dtype=np.int32)
    chimera = np.zeros(n_old, dtype=bool)
    exact = np.zeros(n_old, dtype=bool)
    for i, r in enumerate(old.residue_id):
        m = old.coord_mask[i]
        if not bool(m.any()):
            continue
        ref = 1 if bool(m[1]) else int(m.nonzero()[0, 0])
        for j in cand.get(r.rsplit(":", 1)[0], []):
            if bool(new.coord_mask[j, ref]) and torch.equal(new.coords[j, ref], old.coords[i, ref]):
                index[i] = j
                same = (new.coords[j][m] == old.coords[i][m]).all(-1) & new.coord_mask[j][m]
                chimera[i] = not bool(same.all())
                exact[i] = torch.equal(new.coords[j], old.coords[i]) and torch.equal(new.coord_mask[j], m)
                break
    return {"index": index, "chimera": chimera, "exact": exact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    assert 0 <= a.part < a.n_parts, (a.part, a.n_parts)
    data_dir, out_root = pathlib.Path(a.data_dir).resolve(), pathlib.Path(a.out_root).resolve()
    assert out_root != data_dir and data_dir not in out_root.parents, "never write into the shared pdb_train tree"
    cats = load_audit(a.audit)
    stems = sorted({l.split()[0] for l in open(a.stems) if l.strip()})
    bad = [s for s in stems if cats.get(s) not in REPROCESS + ("collide",)]
    assert not bad, f"{len(bad)} stems outside the re-process set, e.g. {bad[:5]}"
    by_pdb = collections.defaultdict(list)
    for s in stems:
        by_pdb[s.split("_")[0]].append(s)
    pdbs = sorted(by_pdb)[a.part::a.n_parts]
    man = json.load(open(data_dir / "shard_manifest.json"))
    raw_dir = data_dir / "raw"
    ent, mm, _fh = open_pack(a.pack)
    staging = out_root / "staging" / f"part_{a.part:04d}" / "processed"
    rows, label_seq, o2n = [], {}, {}
    t0 = time.time()
    for k, pdb in enumerate(pdbs):
        df, path = None, None
        for stem in by_pdb[pdb]:
            old = load_entry(ent, mm, stem)
            row = dict.fromkeys(COLUMNS, "")
            row.update(stem=stem, category=cats[stem], L_old=len(old.residue_id))
            if cats[stem] == "collide":
                row["status"] = "dropped_collide"
                rows.append(row)
                continue
            status, g = reprocess_chain(stem, raw_dir, staging, man, True, old)
            row["status"] = status
            if g is not None:
                want = {kk: v for kk, v in schema(old).items() if kk != "contact_map_confind"}
                assert schema(g) == want, (stem, schema(g), want)
                if df is None:
                    path = str(raw_path(raw_dir, pdb, man))
                    df = read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True)
                ls = label_seq_of(df, path, stem.split("_", 1)[1], g)
                m = old_to_new(old, g)
                row.update(L_new=int(g.coord_mask.any(-1).sum()), n_ins_new=sum(not r.endswith(":") for r in g.residue_id),
                           label_seq_increasing=bool((ls[1:] > ls[:-1]).all()), n_unmatched=int((m["index"] < 0).sum()),
                           n_chimera=int(m["chimera"].sum()), n_inexact=int((~m["exact"]).sum()))
                assert row["label_seq_increasing"] and row["n_unmatched"] == 0, (stem, row)
                if row["L_new"] > MAX_LEN:
                    row["status"] = "dropped_over512"
                else:
                    target = _processed_path_sharded(out_root / "processed", stem, man)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".tmp")
                    torch.save(g, tmp)
                    os.replace(tmp, target)
                    row.update(status="written", confind="pending")
                    label_seq[stem], o2n[stem] = ls, m
            rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"[part {a.part}] {k + 1}/{len(pdbs)} entries {time.time() - t0:.0f} s", flush=True)
    tag = f"part_{a.part:04d}_of_{a.n_parts:04d}"
    for sub in ("sidecars", "status"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    side = out_root / "sidecars" / f"{tag}.pt"
    torch.save({"label_seq": label_seq, "old_to_new": o2n, "n_entries": len(pdbs)}, str(side) + ".tmp")
    os.replace(str(side) + ".tmp", side)
    tsv = out_root / "status" / f"{tag}.tsv"
    with open(str(tsv) + ".tmp", "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    os.replace(str(tsv) + ".tmp", tsv)
    c = collections.Counter(r["status"] for r in rows)
    excl = staging.parent / "auto_excluded_non_canonical.txt"
    print(f"[part {a.part}] DONE {len(pdbs)} entries, {len(rows)} chains, {time.time() - t0:.0f} s, status {dict(c)}; "
          f"auto_excluded entries {len(open(excl).readlines()) if excl.exists() else 0} -> {tsv}")


if __name__ == "__main__":
    main()
