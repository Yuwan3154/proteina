"""Utilities for storing per-protein inference directories as uncompressed tar files."""

from __future__ import annotations

import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Iterable


def protein_tar_path(inference_dir: str | Path, protein_id: str) -> Path:
    return Path(inference_dir) / f"{protein_id}.tar"


def ensure_protein_tar(inference_dir: str | Path, protein_id: str) -> Path:
    inference_path = Path(inference_dir)
    inference_path.mkdir(parents=True, exist_ok=True)
    tar_path = protein_tar_path(inference_path, protein_id)
    if not tar_path.exists():
        with tarfile.open(tar_path, "w"):
            pass
    return tar_path


def _validate_member_name(member_name: str, protein_id: str) -> None:
    member_path = Path(member_name)
    parts = member_path.parts
    if member_path.is_absolute():
        raise ValueError(f"Unsafe absolute tar member: {member_name}")
    if not parts:
        raise ValueError("Unsafe empty tar member")
    if parts[0] != protein_id:
        raise ValueError(f"Tar member must start with {protein_id!r}: {member_name}")
    if any(part == ".." for part in parts):
        raise ValueError(f"Unsafe parent traversal tar member: {member_name}")


def _validate_member(member: tarfile.TarInfo, protein_id: str) -> None:
    _validate_member_name(member.name, protein_id)
    if member.issym() or member.islnk():
        raise ValueError(f"Unsafe link tar member: {member.name}")


def _tar_members(tar_path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(tar_path, "r:") as tf:
        return tf.getmembers()


def safe_extract_protein_tar(inference_dir: str | Path, protein_id: str) -> bool:
    inference_path = Path(inference_dir)
    protein_dir = inference_path / protein_id
    tar_path = protein_tar_path(inference_path, protein_id)
    if protein_dir.exists():
        return False
    if not tar_path.exists():
        return False

    with tarfile.open(tar_path, "r:") as tf:
        members = tf.getmembers()
        for member in members:
            _validate_member(member, protein_id)
        if not members:
            return False
        tf.extractall(inference_path)
    return True


def restore_protein_dirs(inference_dir: str | Path, protein_ids: Iterable[str]) -> dict[str, float | int]:
    start = time.perf_counter()
    initialized = 0
    extracted = 0
    already_present = 0
    empty_or_missing = 0
    inference_path = Path(inference_dir)
    for protein_id in protein_ids:
        tar_path = protein_tar_path(inference_path, protein_id)
        if not tar_path.exists():
            initialized += 1
        ensure_protein_tar(inference_path, protein_id)
        protein_dir = inference_path / protein_id
        if protein_dir.exists():
            already_present += 1
        elif safe_extract_protein_tar(inference_path, protein_id):
            extracted += 1
        else:
            empty_or_missing += 1
    return {
        "initialized": initialized,
        "extracted": extracted,
        "already_present": already_present,
        "empty_or_missing": empty_or_missing,
        "elapsed_seconds": time.perf_counter() - start,
    }


def pack_protein_dir(inference_dir: str | Path, protein_id: str, delete_after: bool = True) -> bool:
    inference_path = Path(inference_dir)
    protein_dir = inference_path / protein_id
    tar_path = protein_tar_path(inference_path, protein_id)
    if not protein_dir.is_dir():
        ensure_protein_tar(inference_path, protein_id)
        return False

    tmp_path = inference_path / f"{protein_id}.tar.tmp.{os.getpid()}"
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w") as tf:
        tf.add(protein_dir, arcname=protein_id, recursive=True)
    os.replace(tmp_path, tar_path)
    if delete_after:
        shutil.rmtree(protein_dir)
    return True


def pack_protein_dirs(
    inference_dir: str | Path,
    protein_ids: Iterable[str],
    delete_after: bool = True,
) -> dict[str, float | int]:
    start = time.perf_counter()
    packed = 0
    skipped_missing = 0
    for protein_id in protein_ids:
        if pack_protein_dir(inference_dir, protein_id, delete_after=delete_after):
            packed += 1
        else:
            skipped_missing += 1
    return {
        "packed": packed,
        "skipped_missing": skipped_missing,
        "elapsed_seconds": time.perf_counter() - start,
    }


def protein_relative_path_exists(inference_dir: str | Path, protein_id: str, relative_path: str | Path) -> bool:
    inference_path = Path(inference_dir)
    loose_path = inference_path / protein_id / relative_path
    if loose_path.exists():
        return True
    tar_path = protein_tar_path(inference_path, protein_id)
    if not tar_path.exists():
        return False
    member_name = str(Path(protein_id) / relative_path)
    with tarfile.open(tar_path, "r:") as tf:
        return member_name in set(tf.getnames())
