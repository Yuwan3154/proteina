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
import shutil
from typing import Dict, List, Optional

_COL_NAME1 = 0
_COL_NAME2 = 1
_COL_TM1 = 2
_COL_TM2 = 3
_COL_RMSD = 4

# Fallback executables tried (in order) when no explicit path / $USALIGN_PATH is
# given. ~/.local/bin is the per-user install location used on SLURM compute
# nodes whose non-login PATH omits it.
_USALIGN_FALLBACK_CANDIDATES = (
    "USalign",
    os.path.expanduser("~/.local/bin/USalign"),
    "TMscore",
    "/usr/local/bin/TMscore",
)


def resolve_usalign_exe(usalign_path: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Resolve a USalign/TMscore executable to an absolute path.

    Search order: the explicit ``usalign_path`` argument, then ``$USALIGN_PATH``,
    then a small set of PATH / standard-location fallbacks. This lets an explicit
    path win even when the executable is not on PATH (e.g. installed at
    ``~/.local/bin/USalign`` on a SLURM compute node whose non-login PATH omits it).

    Returns the resolved executable path, or ``None`` if nothing resolves and
    ``required`` is False. When ``required`` is True and nothing resolves, raises
    ``RuntimeError`` with a clear message — fail loudly at startup instead of
    letting callers silently produce all-NaN / tms=0.0 TM-scores deep in scoring.
    """
    candidates = []
    if usalign_path:
        candidates.append(usalign_path)
    env_path = os.environ.get("USALIGN_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend(_USALIGN_FALLBACK_CANDIDATES)

    for cand in candidates:
        if not cand:
            continue
        expanded = os.path.expanduser(cand)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return os.path.abspath(expanded)
        found = shutil.which(expanded)  # bare name -> PATH; abs/rel path -> verify
        if found:
            return found

    if required:
        raise RuntimeError(
            "resolve_usalign_exe(): no USalign/TMscore executable found. Tried "
            f"explicit={usalign_path!r}, $USALIGN_PATH={os.environ.get('USALIGN_PATH')!r}, "
            f"fallbacks={list(_USALIGN_FALLBACK_CANDIDATES)}. "
            "Pass --usalign_path /path/to/USalign or add USalign to PATH; refusing "
            "to proceed and silently produce all-NaN TM-scores."
        )
    return None


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
