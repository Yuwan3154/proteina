"""
USalign -outfmt 2 tabular parsing (no protein_ebm dependency).

Pairwise, -dir, and -dir2 outputs share the same first columns:
``PDBchain1, PDBchain2, TM1, TM2, RMSD, ...``.

Policy:
- ``tms`` means TM1 (normalized to Structure_1 / list1 input)
- ``gdt`` is NaN because GDT-TS is not present in ``-outfmt 2``
"""

from __future__ import annotations

import os
from typing import Dict, List

_COL_NAME1 = 0
_COL_NAME2 = 1
_COL_TM1 = 2
_COL_TM2 = 3
_COL_RMSD = 4


def normalize_usalign_structure_name(raw_name: str) -> str:
    """Normalize a USalign PDBchain field to a join-friendly basename."""
    name = os.path.basename(raw_name.strip())
    if ":" in name:
        name = name.split(":", 1)[0]
    return name


def iter_usalign_outfmt2_rows(text: str) -> List[List[str]]:
    """Return all non-comment tabular data rows as split columns."""
    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if not parts or parts[0].startswith("#"):
            continue
        rows.append(parts)
    return rows


def metrics_from_usalign_outfmt2_row(parts: List[str]) -> Dict[str, float]:
    if len(parts) <= _COL_RMSD:
        raise ValueError(f"USalign -outfmt 2 row too short: {parts!r}")
    return {
        "tms": float(parts[_COL_TM1]),
        "tms2": float(parts[_COL_TM2]),
        "rms": float(parts[_COL_RMSD]),
        "gdt": float("nan"),
    }


def parse_usalign_pair_outfmt2(stdout: str) -> Dict[str, float]:
    """First data row from pairwise ``USalign ... -outfmt 2`` stdout."""
    rows = iter_usalign_outfmt2_rows(stdout)
    if not rows:
        raise ValueError("No USalign -outfmt 2 data row found")
    return metrics_from_usalign_outfmt2_row(rows[0])


def parse_usalign_outfmt2_named_rows(text: str) -> List[Dict[str, object]]:
    """Return parsed ``-outfmt 2`` rows with both raw and normalized names."""
    parsed_rows: List[Dict[str, object]] = []
    for parts in iter_usalign_outfmt2_rows(text):
        metrics = metrics_from_usalign_outfmt2_row(parts)
        parsed_rows.append(
            {
                "structure_1": parts[_COL_NAME1],
                "structure_2": parts[_COL_NAME2],
                "structure_1_name": normalize_usalign_structure_name(parts[_COL_NAME1]),
                "structure_2_name": normalize_usalign_structure_name(parts[_COL_NAME2]),
                **metrics,
            }
        )
    return parsed_rows


def parse_usalign_dir_outfmt2_tm1_scores(text: str) -> List[float]:
    """TM1 column for each data row from ``USalign -dir ... -outfmt 2``."""
    return [float(parts[_COL_TM1]) for parts in iter_usalign_outfmt2_rows(text) if len(parts) > _COL_TM1]
