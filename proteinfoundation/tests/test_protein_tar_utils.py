import io
import tarfile

import pytest

from proteinfoundation.af2rank_evaluation.protein_tar_utils import (
    ensure_protein_tar,
    pack_protein_dir,
    protein_relative_path_exists,
    protein_tar_path,
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
