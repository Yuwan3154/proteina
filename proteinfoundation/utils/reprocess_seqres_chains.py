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

Since 2026-10-22 seqres_order=True also resolves altlocs with the AlphaFold3 rule (graphein_utils.resolve_altlocs_af3).

Modes:
  --stems STEMS.txt  the repair set (audit categories merged / nonmonotone / ok_ins / collide); collide stems are
                     dropped unprocessed.
  --all-stems        every stem of the pack index (partitioned by --part / --n-parts); collide stems ARE processed and
                     recorded (whether they enter T8b is open with the user); stems missing from the audit get category
                     not_in_audit. --stems-subset FILE restricts it to the listed stems (each must be in the pack
                     index), e.g. to rerun failed rows; nothing else changes.
  --write-changed-only (either mode) writes a stem's .pt + sidecars only if some field differs from the stored entry.
Every processed stem is compared with its stored pack entry over all fields except contact_map_confind: fields = the
differing field names; rid_exact / rid_mod_ins = residue_id equal exactly / modulo the insertion code; backbone_changed =
residue count or order (residue_id modulo insertion code) or any N/CA/C/O coordinate or mask differs; raw_mtime = mtime of
the raw mmCIF; raw_used = its path; raw_replaced = both the flat raw/<id>.cif(.gz) (which _resolve_raw_sharded checks
first, e.g. a 2026-05-27 re-download) and the bucketed raw/<bucket>/<id>.cif(.gz) exist and their decompressed contents
differ (user: re-process from the current raw file; old_to_new then allows match 4).
Status per stem: written (stems mode) | changed (written; --write-changed-only) | unchanged (--write-changed-only; nothing
written) | dropped_collide (stems mode: side-table status collide; never processed) | dropped_over512 (new
processed_length = coord_mask.any(-1).sum() > 512, the T8 config max_length; checked before unchanged) |
failed:process_returned_none (the entry point logs the exception and returns None) | failed:unmatched (--all-stems: a
non-collide stem with n_unmatched > 0; nothing written; --stems mode asserts instead). No stem is skipped without a row:
rows per part == stems assigned to it (asserted; n_stems in the sidecar), and the status TSV is the source of truth.
Sidecars (<out-root>/sidecars/part_K_of_N.pt):
  label_seq  {stem: int32 mmCIF label_seq_id per new residue} (protein_to_pyg(residue_order="label_seq").label_seq_id),
    for EVERY processed stem (incl. unchanged / dropped_over512 / failed:unmatched)
  old_to_new (written stems only) {stem: {"index": int32 [L_old] new residue index (-1 = no match), "match": int8,
    "chimera": bool, "exact": bool}}: each old residue is matched on coordinates among the new residues sharing its chain:resname:authnum
    (insertion code stripped) on its reference atom (CA; first present atom when CA is missing). match 1 = the old atom
    equals the candidate's resolved atom; else match 2 = every coordinate component of the old atom equals that
    component of some raw altloc row of that atom in the candidate (a whole altloc row the old last-row-wins scatter
    stored, or a per-component chimera of rows); else match 3 = every component equals that component of some raw row
    of that atom among the UNION of all candidates (an old merged residue mixing insertion-code residues, e.g. 129 /
    129A); winner = the candidate supplying the most components, ties -> lowest new index; chimera = True; else, only
    when raw_replaced, match 4 = the new residue with the identical residue_id (none -> still unmatched: the stem fails
    closed); 0 = unmatched (status columns
    n_matched_exact / n_matched_altloc / n_matched_merged / n_matched_id / n_unmatched; n_unmatched must be 0 except for collide stems, whose unchosen component's residue has no
    counterpart). chimera = some present old atom is not the winner's; exact = winner's coords and coord_mask rows
    equal the old ones.
Usage: python -m proteinfoundation.utils.reprocess_seqres_chains --pack PACK --data-dir DATA_PATH/pdb_train \
           --audit SEQRES_AUDIT.tsv (--stems STEMS.txt | --all-stems [--stems-subset FILE]) [--write-changed-only] --part K --n-parts N \
           --out-root OUT_ROOT
"""

import argparse
import collections
import gzip
import hashlib
import io
import json
import mmap
import os
import pathlib
import time

import numpy as np
import pandas as pd
import torch

from proteinfoundation.datasets.pdb_data import (PDBLightningDataModule, _bucket_from_manifest, _processed_path_sharded,
                                                  _resolve_raw_sharded)
from proteinfoundation.graphein_utils.graphein_utils import PROTEIN_ATOMS, protein_to_pyg, read_pdb_to_dataframe
from proteinfoundation.utils.precompute_dssp_targets import _compute_dssp

MAX_LEN = 512  # configs/datasets_config/pdb/pdb_train_contact-confind-synthtopo_S25_max512_t8.yaml max_length
REPROCESS = ("merged", "nonmonotone", "ok_ins")
COLUMNS = ("stem", "category", "status", "L_old", "L_new", "n_ins_new", "label_seq_increasing", "n_unmatched",
           "n_matched_exact", "n_matched_altloc", "n_chimera", "n_inexact", "fields", "rid_exact", "rid_mod_ins",
           "backbone_changed", "raw_mtime", "raw_used", "raw_replaced", "n_matched_merged", "n_matched_id", "confind")
SKIP = ("contact_map_confind",)


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
                        store_bfactor=True, fill_value_coords=1e-5, residue_order="label_seq",
                        resolve_altlocs=True)
    assert list(ls.residue_id) == list(g.residue_id), "label_seq parse disagrees with the processed graph"
    return ls.label_seq_id.to(torch.int32)


def altloc_rows(df, chain) -> dict:
    """residue_id -> atom name -> float32 [n, 3] of EVERY model-1 ATOM row of the chain (all altlocs), keyed like
    protein_to_pyg builds residue_id; df = read_pdb_to_dataframe(mmcif_seqres_fields=True)."""
    a = df.loc[(df["chain_id"] == chain) & (df["record_name"] == "ATOM")]
    rid = a["chain_id"] + ":" + a["residue_name"] + ":" + a["residue_number"].astype(str) + ":" + a["insertion"]
    xyz = a[["x_coord", "y_coord", "z_coord"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r, n, c in zip(rid.tolist(), a["atom_name"].tolist(), xyz):
        out[r][n].append(c)
    return {r: {n: torch.from_numpy(np.stack(v)) for n, v in d.items()} for r, d in out.items()}


def old_to_new(old, new, alt_rows, by_id=False) -> dict:
    """Coordinate match of every old residue to a new residue (see module docstring); by_id enables match 4."""
    cand = collections.defaultdict(list)
    for j, r in enumerate(new.residue_id):
        cand[r.rsplit(":", 1)[0]].append(j)
    rid_index = {r: j for j, r in enumerate(new.residue_id)}
    n_old = len(old.residue_id)
    index = np.full(n_old, -1, dtype=np.int32)
    match = np.zeros(n_old, dtype=np.int8)
    chimera = np.zeros(n_old, dtype=bool)
    exact = np.zeros(n_old, dtype=bool)
    for i, r in enumerate(old.residue_id):
        m = old.coord_mask[i]
        if not bool(m.any()):
            continue
        ref = 1 if bool(m[1]) else int(m.nonzero()[0, 0])
        xyz = old.coords[i, ref]
        js = cand.get(r.rsplit(":", 1)[0], [])
        rows = [alt_rows.get(new.residue_id[j], {}).get(PROTEIN_ATOMS[ref]) for j in js]
        comp = [torch.zeros(3, dtype=torch.bool) if x is None else (x == xyz).any(0) for x in rows]
        hit = [j for j in js if bool(new.coord_mask[j, ref]) and torch.equal(new.coords[j, ref], xyz)]
        kind = 1
        if not hit:
            # every component equals some altloc row's component (covers a whole row and a per-component chimera)
            hit, kind = [j for j, c in zip(js, comp) if bool(c.all())], 2
        if not hit and js and bool(torch.stack(comp).any(0).all()):
            # merged insertion-code residues: components drawn from several candidates' rows (e.g. 129 and 129A)
            k = max(range(len(js)), key=lambda q: (int(comp[q].sum()), -js[q]))
            hit, kind = [js[k]], 3
        if not hit and by_id and r in rid_index:
            hit, kind = [rid_index[r]], 4
        if hit:
            j = hit[0]
            index[i], match[i] = j, kind
            same = (new.coords[j][m] == old.coords[i][m]).all(-1) & new.coord_mask[j][m]
            chimera[i] = kind == 3 or not bool(same.all())
            exact[i] = torch.equal(new.coords[j], old.coords[i]) and torch.equal(new.coord_mask[j], m)
    return {"index": index, "match": match, "chimera": chimera, "exact": exact}


def raw_versions(raw_dir, pdb, man) -> tuple:
    """(flat path or None, bucket path or None) for the id spelling raw_path resolves (lower case first)."""
    for pid in (pdb.lower(), pdb):
        names = [f"{pid}.cif", f"{pid}.cif.gz"]
        flat = next((raw_dir / x for x in names if (raw_dir / x).exists()), None)
        bucket = _bucket_from_manifest(pid, man)
        buck = next((raw_dir / bucket / x for x in names if bucket and (raw_dir / bucket / x).exists()), None)
        if flat is not None or buck is not None:
            return flat, buck
    return None, None


def decompressed_md5(path) -> str:
    h = hashlib.md5()
    with (gzip.open(path, "rb") if str(path).endswith(".gz") else open(path, "rb")) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_equal(a, b) -> bool:
    if not (isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape):
        return False
    if a.dtype.is_floating_point:
        return bool(torch.allclose(a, b, rtol=0.0, atol=0.0, equal_nan=True))
    return torch.equal(a, b)


def diff_fields(old, new) -> tuple:
    """(differing field names, residue_id equal exactly, residue_id equal modulo insertion code); SKIP excluded."""
    ko, kn = set(old.keys()) - set(SKIP), set(new.keys()) - set(SKIP)
    fields = [f"keys:{'|'.join(sorted(ko ^ kn))}"] if ko != kn else []
    for k in sorted(ko & kn):
        a, b = old[k], new[k]
        nan_pair = isinstance(a, float) and isinstance(b, float) and a != a and b != b
        if not (tensor_equal(a, b) if isinstance(a, torch.Tensor) else type(a) is type(b) and (a == b or nan_pair)):
            fields.append(k)
    rid_exact = list(old.residue_id) == list(new.residue_id)
    rid_mod_ins = [r.rsplit(":", 1)[0] for r in old.residue_id] == [r.rsplit(":", 1)[0] for r in new.residue_id]
    return fields, rid_exact, rid_mod_ins


def backbone_changed(old, new, rid_mod_ins) -> bool:
    """Residue count / order (residue_id modulo insertion code) or any N/CA/C/O coordinate or mask differs."""
    if not rid_mod_ins or old.coords.shape != new.coords.shape:
        return True
    return not (tensor_equal(old.coords[:, :4], new.coords[:, :4])
                and tensor_equal(old.coord_mask[:, :4], new.coord_mask[:, :4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--stems", help="stems file (re-process set); or --all-stems")
    ap.add_argument("--all-stems", action="store_true", help="every stem of the pack index; collide stems processed")
    ap.add_argument("--stems-subset", help="with --all-stems: only these pack stems (same --all-stems semantics)")
    ap.add_argument("--write-changed-only", action="store_true", help="write a stem only if a field differs")
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--n-parts", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()
    assert 0 <= a.part < a.n_parts, (a.part, a.n_parts)
    assert (a.stems is None) == a.all_stems, "exactly one of --stems / --all-stems"
    assert a.stems_subset is None or a.all_stems, "--stems-subset needs --all-stems"
    data_dir, out_root = pathlib.Path(a.data_dir).resolve(), pathlib.Path(a.out_root).resolve()
    assert out_root != data_dir and data_dir not in out_root.parents, "never write into the shared pdb_train tree"
    cats = load_audit(a.audit)
    ent, mm, _fh = open_pack(a.pack)
    if a.all_stems and a.stems_subset is not None:
        stems = sorted({l.split()[0] for l in open(a.stems_subset) if l.strip()})
        missing = [s for s in stems if s not in ent]
        assert not missing, f"{len(missing)} --stems-subset stems not in the pack index, e.g. {missing[:5]}"
    elif a.all_stems:
        stems = sorted(ent)
    else:
        stems = sorted({l.split()[0] for l in open(a.stems) if l.strip()})
        bad = [s for s in stems if cats.get(s) not in REPROCESS + ("collide",)]
        assert not bad, f"{len(bad)} stems outside the re-process set, e.g. {bad[:5]}"
    by_pdb = collections.defaultdict(list)
    for s in stems:
        by_pdb[s.split("_")[0]].append(s)
    pdbs = sorted(by_pdb)[a.part::a.n_parts]
    man = json.load(open(data_dir / "shard_manifest.json"))
    raw_dir = data_dir / "raw"
    staging = out_root / "staging" / f"part_{a.part:04d}" / "processed"
    rows, label_seq, o2n = [], {}, {}
    t0 = time.time()
    for k, pdb in enumerate(pdbs):
        df, path = None, None
        flat, buck = raw_versions(raw_dir, pdb, man)
        # the flat raw/<id>.cif.gz shadows the bucketed file the stored pack was built from; re-processing uses the flat
        raw_replaced = flat is not None and buck is not None and decompressed_md5(flat) != decompressed_md5(buck)
        for stem in by_pdb[pdb]:
            old = load_entry(ent, mm, stem)
            row = dict.fromkeys(COLUMNS, "")
            row.update(stem=stem, category=cats.get(stem, "not_in_audit"), L_old=len(old.residue_id))
            raw = raw_path(raw_dir, pdb, man)
            if raw is not None:
                row.update(raw_mtime=repr(os.path.getmtime(raw)), raw_used=str(raw), raw_replaced=raw_replaced)
            if row["category"] == "collide" and not a.all_stems:
                row["status"] = "dropped_collide"
                rows.append(row)
                continue
            status, g = reprocess_chain(stem, raw_dir, staging, man, True, old)
            row["status"] = status
            if g is None:
                rows.append(row)
                continue
            want = {kk: v for kk, v in schema(old).items() if kk != "contact_map_confind"}
            assert schema(g) == want, (stem, schema(g), want)
            fields, rid_exact, rid_mod_ins = diff_fields(old, g)
            row.update(fields=",".join(fields), rid_exact=rid_exact, rid_mod_ins=rid_mod_ins,
                       backbone_changed=backbone_changed(old, g, rid_mod_ins),
                       L_new=int(g.coord_mask.any(-1).sum()), n_ins_new=sum(not r.endswith(":") for r in g.residue_id))
            if df is None:
                path = str(raw)
                df = read_pdb_to_dataframe(path=path, mmcif_seqres_fields=True)
            chain = stem.split("_", 1)[1]
            ls = label_seq_of(df, path, chain, g)
            row["label_seq_increasing"] = bool((ls[1:] > ls[:-1]).all())
            assert row["label_seq_increasing"], (stem, row)
            label_seq[stem] = ls  # every processed stem, written or not: T8b needs SEQRES positions for all chains
            if row["L_new"] > MAX_LEN:
                row["status"] = "dropped_over512"
                rows.append(row)
                continue
            if a.write_changed_only and not fields:
                row["status"] = "unchanged"
                rows.append(row)
                continue
            m = old_to_new(old, g, altloc_rows(df, chain), by_id=raw_replaced)
            row.update(n_unmatched=int((m["match"] == 0).sum()), n_matched_exact=int((m["match"] == 1).sum()),
                       n_matched_altloc=int((m["match"] == 2).sum()), n_matched_merged=int((m["match"] == 3).sum()),
                       n_matched_id=int((m["match"] == 4).sum()), n_chimera=int(m["chimera"].sum()),
                       n_inexact=int((~m["exact"]).sum()))
            # a collide (microheterogeneity) stem loses the residue of the unchosen component: recorded, never failed
            if row["n_unmatched"] > 0 and row["category"] != "collide":
                assert a.all_stems, (stem, row)
                row["status"] = "failed:unmatched"
                rows.append(row)
                continue
            target = _processed_path_sharded(out_root / "processed", stem, man)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            torch.save(g, tmp)
            os.replace(tmp, target)
            row.update(status="changed" if a.write_changed_only else "written", confind="pending")
            o2n[stem] = m
            rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"[part {a.part}] {k + 1}/{len(pdbs)} entries {time.time() - t0:.0f} s", flush=True)
    n_stems = sum(len(by_pdb[p]) for p in pdbs)
    assert len(rows) == n_stems, (len(rows), n_stems)
    tag = f"part_{a.part:04d}_of_{a.n_parts:04d}"
    for sub in ("sidecars", "status"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    side = out_root / "sidecars" / f"{tag}.pt"
    torch.save({"label_seq": label_seq, "old_to_new": o2n, "n_entries": len(pdbs), "n_stems": n_stems}, str(side) + ".tmp")
    os.replace(str(side) + ".tmp", side)
    tsv = out_root / "status" / f"{tag}.tsv"
    with open(str(tsv) + ".tmp", "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    os.replace(str(tsv) + ".tmp", tsv)
    c = collections.Counter(r["status"] for r in rows)
    excl = staging.parent / "auto_excluded_non_canonical.txt"
    print(f"[part {a.part}] DONE {len(pdbs)} entries, {len(rows)} of {n_stems} stems, {time.time() - t0:.0f} s, status {dict(c)}; "
          f"auto_excluded entries {len(open(excl).readlines()) if excl.exists() else 0} -> {tsv}")


if __name__ == "__main__":
    main()
