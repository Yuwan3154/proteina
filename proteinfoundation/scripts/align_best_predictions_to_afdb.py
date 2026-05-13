#!/usr/bin/env python3
"""Align best_predictions PDBs to reference mmCIF/PDB structures; write two-chain PDBs.

Uses Kabsch alignment on matched Cα atoms by residue id. Reference files are read from
``--pdb-root`` (e.g. AlphaFold DB mmCIF named ``AF-{protein_id}-model_v*.cif``).

Run from the ``proteina`` repository root (or any cwd)::

    python proteinfoundation/scripts/align_best_predictions_to_afdb.py \\
        prediction/replica_compare_4-seq_vs_21-seq/compare_replicas_converged_passing.csv \\
        -o /tmp/aligned \\
        --inference-root prediction/inference_seq_cond_sampling_..._21-seq..._045-noise \\
        --pdb-root /path/to/data/afdb/pdb \\
        --protein-id A0A023PXM2-F1

Chain-id handling for mmCIF matches ``proteinfoundation.prediction_pipeline.cif_chain_mapping``
(logic inlined so this script runs without PYTHONPATH hacks).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Chain, Residue, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_cif_path(structure_path: str) -> bool:
    return Path(str(structure_path)).suffix.lower() in {".cif", ".mmcif"}


def cif_label_to_auth_chain(cif_path: str, label_chain: Optional[str]) -> Optional[str]:
    """Map mmCIF label_asym_id to auth_asym_id (same logic as cif_chain_mapping)."""
    if label_chain is None:
        return None
    if not _is_cif_path(cif_path):
        return label_chain

    mmcif = MMCIF2Dict(str(cif_path))
    labels = _as_list(mmcif.get("_atom_site.label_asym_id"))
    auths = _as_list(mmcif.get("_atom_site.auth_asym_id"))
    if not labels or not auths or len(labels) != len(auths):
        return label_chain

    mapped_auths = {auth for label, auth in zip(labels, auths) if label == label_chain}
    if len(mapped_auths) == 1:
        return next(iter(mapped_auths))
    if label_chain in set(auths):
        return label_chain
    return label_chain


def resolve_biopython_chain_for_structure(
    structure_path: str,
    chain: Optional[str],
) -> Optional[str]:
    return cif_label_to_auth_chain(structure_path, chain)


def _resolve_atom(atom):
    if getattr(atom, "is_disordered", lambda: False)():
        return atom.selected_child
    return atom


def collect_ca_by_residue(model, chain_id: str) -> dict:
    """Collect CA coordinates for every polymer residue in chain_id.

    Uses res.id[0] == ' ' (ATOM-record residues) as the only filter so that
    non-standard amino acids (MSE, UNK, etc.) are included, avoiding silent
    truncation for structures that contain modified residues.
    """
    chain = model[chain_id]
    coords = {}
    for res in chain:
        if res.id[0] != " ":
            continue
        if "CA" not in res:
            continue
        ca = _resolve_atom(res["CA"])
        coords[res.get_id()] = np.asarray(ca.get_coord(), dtype=float)
    return coords


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray):
    """Return R, centroid_mobile, centroid_target to map mobile onto target (same as plan)."""
    c_m = mobile.mean(axis=0)
    c_t = target.mean(axis=0)
    p_c = mobile - c_m
    q_c = target - c_t
    h = p_c.T @ q_c
    u, _, vt = np.linalg.svd(h)
    r_mat = vt.T @ u.T
    if np.linalg.det(r_mat) < 0:
        u_flip = u.copy()
        u_flip[:, -1] *= -1.0
        r_mat = vt.T @ u_flip.T
    return r_mat, c_m, c_t


def transform_point(r_mat: np.ndarray, c_m: np.ndarray, c_t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Apply the same rigid motion as used for Kabsch rows (right-multiply rotation)."""
    return (x - c_m) @ r_mat.T + c_t


def read_csv_protein_ids(csv_path: Path, id_column: str) -> list[str]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            pid = row.get(id_column, "").strip()
            if pid:
                out.append(pid)
    return list(dict.fromkeys(out))


def write_two_chain_pdb(
    path: Path,
    ref_model,
    pred_model,
    ref_chain_in: str,
    pred_chain_in: str,
    r_mat: np.ndarray,
    c_m: np.ndarray,
    c_t: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = Structure.Structure("aligned")
    model_out = Model.Model(0)
    merged.add(model_out)
    ch_a = Chain.Chain("A")
    ref_chain = ref_model[ref_chain_in]
    for res in ref_chain:
        if res.id[0] != " ":
            continue
        new_res = Residue.Residue(res.id, res.resname, res.segid)
        for atom in res.get_atoms():
            a = _resolve_atom(atom)
            na = a.copy()
            new_res.add(na)
        ch_a.add(new_res)
    model_out.add(ch_a)
    ch_b = Chain.Chain("B")
    pred_chain = pred_model[pred_chain_in]
    for res in pred_chain:
        if res.id[0] != " ":
            continue
        new_res = Residue.Residue(res.id, res.resname, res.segid)
        for atom in res.get_atoms():
            a = _resolve_atom(atom)
            na = a.copy()
            new_c = transform_point(r_mat, c_m, c_t, np.asarray(a.coord, dtype=float))
            na.set_coord(new_c)
            new_res.add(na)
        ch_b.add(new_res)
    model_out.add(ch_b)
    pdb_io = PDBIO()
    pdb_io.set_structure(merged)
    pdb_io.save(str(path))


def align_one(
    protein_id: str,
    inference_root: Path,
    pdb_root: Path,
    predictions_subdir: str,
    ref_name_template: str,
    out_dir: Path,
    ref_chain: str,
    pred_chain: str,
    min_matched: int,
    save_transform: bool,
) -> Path:
    ref_path = pdb_root / ref_name_template.format(protein_id=protein_id)
    pred_path = inference_root / predictions_subdir / f"{protein_id}.pdb"
    if not ref_path.is_file():
        raise FileNotFoundError(f"Missing reference structure: {ref_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing best prediction: {pred_path}")

    cif_parser = MMCIFParser(QUIET=True)
    pdb_parser = PDBParser(QUIET=True)
    ref_structure = cif_parser.get_structure(protein_id, str(ref_path))
    pred_structure = pdb_parser.get_structure(protein_id, str(pred_path))
    ref_model = ref_structure[0]
    pred_model = pred_structure[0]

    ref_chain_id = resolve_biopython_chain_for_structure(str(ref_path), ref_chain)
    pred_chain_id = resolve_biopython_chain_for_structure(str(pred_path), pred_chain)
    if ref_chain_id not in ref_model:
        raise RuntimeError(f"Chain {ref_chain_id} not found in reference structure {ref_path}")
    if pred_chain_id not in pred_model:
        raise RuntimeError(f"Chain {pred_chain_id} not found in prediction {pred_path}")

    ref_ca = collect_ca_by_residue(ref_model, ref_chain_id)
    pred_ca = collect_ca_by_residue(pred_model, pred_chain_id)
    common = set(ref_ca.keys()) & set(pred_ca.keys())
    if len(common) < min_matched:
        raise RuntimeError(
            f"{protein_id}: insufficient matched CA residues "
            f"({len(common)} < {min_matched}); ref={len(ref_ca)} pred={len(pred_ca)}"
        )

    def sort_key(res_id):
        return (res_id[1], res_id[2])

    ordered = sorted(common, key=sort_key)
    p_stack = np.stack([pred_ca[k] for k in ordered], axis=0)
    q_stack = np.stack([ref_ca[k] for k in ordered], axis=0)
    r_mat, c_m, c_t = kabsch_rotation(p_stack, q_stack)

    out_pdb = out_dir / f"{protein_id}_refA_predB_aligned.pdb"
    write_two_chain_pdb(
        out_pdb,
        ref_model,
        pred_model,
        ref_chain_id,
        pred_chain_id,
        r_mat,
        c_m,
        c_t,
    )

    if save_transform:
        npz_path = out_dir / f"{protein_id}_aligned_transform.npz"
        np.savez(npz_path, R=r_mat, centroid_mobile=c_m, centroid_ref=c_t)

    return out_pdb


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Superimpose prediction PDBs onto reference structures (Kabsch on Cα). "
            "Writes merged two-chain PDBs (A=reference, B=aligned prediction)."
        )
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV file listing protein ids (column set by --csv-id-column).",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for merged PDBs (created if missing).",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        required=True,
        help="Inference run root containing the predictions subdirectory (see --predictions-subdir).",
    )
    parser.add_argument(
        "--pdb-root",
        type=Path,
        required=True,
        help="Directory with reference structure files (e.g. AF-{protein_id}-model_v6.cif).",
    )
    parser.add_argument(
        "--ref-name-template",
        type=str,
        default="AF-{protein_id}-model_v6.cif",
        help="Reference filename under --pdb-root; must contain {protein_id}.",
    )
    parser.add_argument(
        "--predictions-subdir",
        type=str,
        default="best_predictions",
        help="Subdirectory of --inference-root containing <protein_id>.pdb files.",
    )
    parser.add_argument(
        "--csv-id-column",
        type=str,
        default="protein_id",
        help="CSV column name for protein identifiers.",
    )
    parser.add_argument("--ref-chain", type=str, default="A")
    parser.add_argument("--pred-chain", type=str, default="A")
    parser.add_argument("--min-matched", type=int, default=3)
    parser.add_argument(
        "--protein-id",
        type=str,
        default=None,
        help="If set, process only this id (ignores CSV ids).",
    )
    parser.add_argument(
        "--save-transform",
        action="store_true",
        help="Write {protein_id}_aligned_transform.npz (R, centroids) next to PDB.",
    )
    args = parser.parse_args()

    if args.protein_id:
        ids = [args.protein_id.strip()]
    else:
        ids = read_csv_protein_ids(args.csv, args.csv_id_column)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for pid in ids:
        out = align_one(
            pid,
            args.inference_root,
            args.pdb_root,
            args.predictions_subdir,
            args.ref_name_template,
            args.out_dir,
            args.ref_chain,
            args.pred_chain,
            args.min_matched,
            args.save_transform,
        )
        print(out)


if __name__ == "__main__":
    main()
