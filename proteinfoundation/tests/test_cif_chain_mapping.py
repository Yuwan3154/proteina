from pathlib import Path

import pytest

from proteinfoundation.af2rank_evaluation.cif_chain_mapping import (
    cif_label_to_auth_chain,
    resolve_ground_truth_usalign_chain,
)
from proteinfoundation.af2rank_evaluation import proteina_analysis
from proteinfoundation.prediction_pipeline import compare_replicas_topk


def _write_label_a_auth_x_cif(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 0.00 ? 1 ALA X CA 1",
                "#",
            ]
        )
        + "\n"
    )


def test_cif_label_to_auth_chain_maps_label_id(tmp_path):
    cif_path = tmp_path / "1h4a.cif"
    pdb_path = tmp_path / "generated.pdb"
    _write_label_a_auth_x_cif(cif_path)
    pdb_path.write_text("")

    assert cif_label_to_auth_chain(str(cif_path), "A") == "X"
    assert cif_label_to_auth_chain(str(cif_path), "X") == "X"
    assert cif_label_to_auth_chain(str(pdb_path), "A") == "A"
    assert cif_label_to_auth_chain(str(cif_path), None) is None


def test_compare_replicas_resolves_only_gt_chain(tmp_path, monkeypatch):
    gt_path = tmp_path / "1h4a.cif"
    template_path = tmp_path / "template.pdb"
    prediction_path = tmp_path / "prediction.pdb"
    _write_label_a_auth_x_cif(gt_path)
    template_path.write_text("")
    prediction_path.write_text("")

    calls = []

    def fake_find_reference_cif(protein_id, cif_dir):
        return str(gt_path)

    def fake_run_usalign_pair(pdb_a, pdb_b, tmscore_mode, chain1=None, chain2=None):
        calls.append((pdb_a, pdb_b, chain1, chain2))
        return {"TM1": 0.1, "TM2": 0.2, "RMSD": 3.0, "L1": 1.0, "L2": 1.0, "Lali": 1.0}

    monkeypatch.setattr(compare_replicas_topk, "_find_reference_cif", fake_find_reference_cif)
    monkeypatch.setattr(compare_replicas_topk, "_run_usalign_pair", fake_run_usalign_pair)

    top_a = {
        "min_ptm": 0.5,
        "template_path": str(template_path),
        "prediction_path": str(prediction_path),
    }
    top_b = {
        "min_ptm": 0.4,
        "template_path": str(template_path),
        "prediction_path": str(prediction_path),
    }

    out = compare_replicas_topk._compute_gt_metrics("1h4a_A", top_a, top_b, str(tmp_path), 5)

    assert out["gt_missing"] is False
    assert calls == [
        (str(gt_path), str(template_path), "X", None),
        (str(gt_path), str(prediction_path), "X", None),
    ]


def test_reference_dir2_resolves_reference_chain_only(tmp_path, monkeypatch):
    gt_path = tmp_path / "1h4a.cif"
    pred_dir = tmp_path / "preds"
    pred_path = pred_dir / "prediction.pdb"
    pred_dir.mkdir()
    _write_label_a_auth_x_cif(gt_path)
    pred_path.write_text("")

    calls = []

    def fake_run_dir2_rows(chain1_path, folder2, names2, env=None, chain1=None, chain2=None):
        calls.append((chain1_path, folder2, names2, chain1, chain2))
        return [
            {
                "structure_1_name": gt_path.name,
                "structure_2_name": pred_path.name,
                "tms": 0.1,
                "tms2": 0.2,
                "rms": 3.0,
                "gdt": float("nan"),
            }
        ]

    monkeypatch.setattr(proteina_analysis, "run_dir2_rows", fake_run_dir2_rows)

    metrics = proteina_analysis.run_reference_vs_paths_dir2(
        reference_path=str(gt_path),
        target_paths=[str(pred_path)],
        chain1="A",
    )

    assert metrics[pred_path.name]["tms"] == 0.1
    assert calls == [
        (
            str(gt_path.resolve()),
            str(pred_dir.resolve()),
            [pred_path.name],
            "X",
            None,
        )
    ]


def test_resolve_ground_truth_usalign_chain_alias(tmp_path):
    cif_path = tmp_path / "1h4a.cif"
    _write_label_a_auth_x_cif(cif_path)

    assert resolve_ground_truth_usalign_chain(str(cif_path), "A") == "X"


def test_proteinebm_tmscore_pdb_paths_converts_only_reference_side(tmp_path, monkeypatch):
    proteinebm_scorer = pytest.importorskip("proteinfoundation.af2rank_evaluation.proteinebm_scorer")
    gt_path = tmp_path / "1h4a.cif"
    pred_path = tmp_path / "prediction.pdb"
    _write_label_a_auth_x_cif(gt_path)
    pred_path.write_text("")

    calls = []

    def fake_which(name):
        return "USalign" if name == "USalign" else None

    def fake_check_output(cmd, text=True, env=None):
        calls.append(cmd)
        return "#PDBchain1\tPDBchain2\tTM1\tTM2\tRMSD\tID1\tID2\tIDali\tL1\tL2\tLali\nx\ty\t0.1\t0.2\t3.0\t1\t1\t1\t1\t1\t1\n"

    monkeypatch.setattr(proteinebm_scorer.shutil, "which", fake_which)
    monkeypatch.setattr(proteinebm_scorer.subprocess, "check_output", fake_check_output)

    metrics = proteinebm_scorer.tmscore_pdb_paths(
        str(gt_path),
        str(pred_path),
        chain1="A",
        chain1_is_reference=True,
    )

    assert metrics["tms"] == 0.1
    assert calls == [[
        "USalign",
        str(gt_path),
        str(pred_path),
        "-chain1",
        "X",
        "-TMscore",
        "5",
        "-outfmt",
        "2",
    ]]
