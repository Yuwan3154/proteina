import io
import json
import tarfile

import pytest

from proteinfoundation.af2rank_evaluation.protein_tar_utils import (
    ensure_protein_tar,
    list_protein_members,
    pack_protein_dir,
    protein_glob_members,
    protein_relative_path_exists,
    protein_tar_path,
    read_protein_text,
    safe_extract_protein_tar,
)


def test_empty_tar_initialization(tmp_path):
    tar_path = ensure_protein_tar(tmp_path, "protein_A")

    assert tar_path == tmp_path / "protein_A.tar"
    assert tar_path.exists()
    with tarfile.open(tar_path, "r:") as tf:
        assert tf.getmembers() == []


def test_pack_contains_top_level_protein_dir_and_no_compression(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    nested = protein_dir / "analysis"
    nested.mkdir(parents=True)
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (nested / "scores.csv").write_text("x\n1\n")

    assert pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert not protein_dir.exists()
    tar_path = protein_tar_path(tmp_path, protein_id)
    with tarfile.open(tar_path, "r:") as tf:
        names = tf.getnames()
        assert all(name == protein_id or name.startswith(f"{protein_id}/") for name in names)
        assert f"{protein_id}/analysis/scores.csv" in names


def test_restore_round_trip(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    protein_dir.mkdir()
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")

    pack_protein_dir(tmp_path, protein_id, delete_after=True)
    assert safe_extract_protein_tar(tmp_path, protein_id)

    assert (protein_dir / "protein_A_0.pdb").read_text() == "ATOM\n"


def test_path_exists_loose_or_tar(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    protein_dir.mkdir()
    rel_path = "protein_A_0.pdb"
    (protein_dir / rel_path).write_text("ATOM\n")

    assert protein_relative_path_exists(tmp_path, protein_id, rel_path)
    pack_protein_dir(tmp_path, protein_id, delete_after=True)
    assert protein_relative_path_exists(tmp_path, protein_id, rel_path)
    assert not protein_relative_path_exists(tmp_path, protein_id, "missing.pdb")


def test_rejects_unsafe_members(tmp_path):
    protein_id = "protein_A"
    tar_path = protein_tar_path(tmp_path, protein_id)
    data = b"bad"
    info = tarfile.TarInfo("../bad")
    info.size = len(data)
    with tarfile.open(tar_path, "w") as tf:
        tf.addfile(info, io.BytesIO(data))

    with pytest.raises(ValueError):
        safe_extract_protein_tar(tmp_path, protein_id)
    assert not (tmp_path.parent / "bad").exists()


def test_list_members_from_loose_and_tar(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    (protein_dir / "nested").mkdir(parents=True)
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (protein_dir / "nested" / "scores.csv").write_text("x\n1\n")

    assert list_protein_members(tmp_path, protein_id) == {"protein_A_0.pdb", "nested/scores.csv"}
    pack_protein_dir(tmp_path, protein_id, delete_after=True)
    assert not protein_dir.exists()
    assert list_protein_members(tmp_path, protein_id) == {"protein_A_0.pdb", "nested/scores.csv"}


def test_glob_members_from_tar(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    protein_dir.mkdir()
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (protein_dir / "protein_A_1.pdb").write_text("ATOM\n")
    (protein_dir / "other.txt").write_text("x")
    pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert protein_glob_members(tmp_path, protein_id, f"{protein_id}_*.pdb") == [
        "protein_A_0.pdb",
        "protein_A_1.pdb",
    ]


def test_read_text_from_tar(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    (protein_dir / "analysis").mkdir(parents=True)
    (protein_dir / "analysis" / "summary.json").write_text('{"ok": true}\n')
    pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert read_protein_text(tmp_path, protein_id, "analysis/summary.json") == '{"ok": true}\n'
    assert read_protein_text(tmp_path, protein_id, "missing.txt") is None


def test_relative_path_exists_streams_member_names(tmp_path):
    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    protein_dir.mkdir()
    (protein_dir / "present.txt").write_text("yes")
    pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert protein_relative_path_exists(tmp_path, protein_id, "present.txt")
    assert not protein_relative_path_exists(tmp_path, protein_id, "absent.txt")


def test_inference_completeness_from_tar(tmp_path, monkeypatch):
    from proteinfoundation.af2rank_evaluation import parallel_proteina_inference as mod

    root = tmp_path / "root"
    inference_dir = root / "inference" / "cfg"
    protein_id = "protein_A"
    protein_dir = inference_dir / protein_id
    protein_dir.mkdir(parents=True)
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (protein_dir / "protein_A_1.pdb").write_text("ATOM\n")
    pack_protein_dir(inference_dir, protein_id, delete_after=True)
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("id\nprotein_A\n")
    monkeypatch.setattr(mod, "PROTEINA_BASE_DIR", str(root))

    assert mod.find_proteins_needing_inference(
        str(csv_path), "id", "cfg", candidate_proteins=[protein_id], tar_protein_dirs=True
    ) == []
    assert mod.find_proteins_needing_inference(
        str(csv_path), "id", "cfg", candidate_proteins=[protein_id], tar_protein_dirs=True, 
    ) == []


def test_proteinebm_completeness_from_tar(tmp_path, monkeypatch):
    from proteinfoundation.af2rank_evaluation import parallel_proteinebm_scoring as mod

    root = tmp_path / "root"
    inference_dir = root / "inference" / "cfg"
    protein_id = "protein_A"
    protein_dir = inference_dir / protein_id
    analysis_dir = protein_dir / "proteinebm_v2_cathmd_analysis"
    analysis_dir.mkdir(parents=True)
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (analysis_dir / "proteinebm_scores_protein_A.csv").write_text(
        "protein_id,structure_file,structure_path,t,energy\nprotein_A,protein_A_0.pdb,/x/protein_A_0.pdb,1,0.5\n"
    )
    pack_protein_dir(inference_dir, protein_id, delete_after=True)
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("id\nprotein_A\n")
    monkeypatch.setattr(mod, "PROTEINA_BASE_DIR", str(root))

    assert mod.find_proteins_needing_proteinebm(
        str(csv_path), "id", "cfg", candidate_proteins=[protein_id], tar_protein_dirs=True
    ) == []


def test_af2rank_topk_completeness_from_tar(tmp_path):
    from proteinfoundation.prediction_pipeline.run_af2rank_prediction import _all_scored_from_tar_or_dir

    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    pred_dir = protein_dir / "af2rank_on_proteinebm_top_k" / "af2rank_analysis" / "predicted_structures"
    score_dir = pred_dir.parent
    pred_dir.mkdir(parents=True)
    (pred_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (score_dir / "af2rank_scores_protein_A.csv").write_text("structure_file,ptm\nprotein_A_0.pdb,0.9\n")
    pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert _all_scored_from_tar_or_dir(
        tmp_path,
        protein_id,
        "af2rank_on_proteinebm_top_k/af2rank_analysis/af2rank_scores_protein_A.csv",
        "af2rank_on_proteinebm_top_k/af2rank_analysis/predicted_structures",
        {"protein_A_0.pdb"},
    )
    assert not (tmp_path / protein_id).exists()


def test_central_analysis_completeness_from_tar(tmp_path):
    from proteinfoundation.af2rank_evaluation.proteina_analysis import analysis_complete_in_tar_or_dir

    protein_id = "protein_A"
    protein_dir = tmp_path / protein_id
    analysis_dir = protein_dir / "proteina_analysis"
    analysis_dir.mkdir(parents=True)
    (protein_dir / "protein_A_0.pdb").write_text("ATOM\n")
    (analysis_dir / "analysis_summary_protein_A.json").write_text(json.dumps({"n_samples": 1}))
    pack_protein_dir(tmp_path, protein_id, delete_after=True)

    assert analysis_complete_in_tar_or_dir(str(tmp_path), protein_id)
    assert not (tmp_path / protein_id).exists()
