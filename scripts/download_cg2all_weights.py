#!/usr/bin/env python3
"""
Pre-download cg2all model weights needed by the proteina pipeline.

Run this once on a machine with internet access before submitting jobs to HPC
nodes that lack outbound connectivity. The weights are stored in the cg2all
package's MODEL_HOME directory inside the active Python environment.

Usage (run with the cue_openfold Python):
    /path/to/cue_openfold/bin/python download_cg2all_weights.py

Models downloaded:
    CalphaBasedModel        — CA-only → all-atom reconstruction (primary)
    CalphaBasedModel-FIX    — fix-atom variant of above
    BackboneModel           — backbone reconstruction
    BackboneModel-FIX       — fix-atom variant of above
"""

import sys
from pathlib import Path

from cg2all.lib.libconfig import MODEL_HOME

# Import the shared downloader from the af2rank_evaluation module.
# Adjust the sys.path so this script can be run from any working directory.
_script_dir = Path(__file__).resolve().parent
_pkg_root = _script_dir.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from proteinfoundation.af2rank_evaluation.cg2all_reconstruct import download_ckpt_file

# Models required by the proteina pipeline.
MODELS_TO_DOWNLOAD = [
    ("CalphaBasedModel", False),
    ("CalphaBasedModel", True),   # -FIX
    ("BackboneModel",    False),
    ("BackboneModel",    True),   # -FIX
]


def main() -> None:
    print(f"cg2all MODEL_HOME: {MODEL_HOME}")
    downloaded = 0
    skipped = 0
    for model_type, fix_atom in MODELS_TO_DOWNLOAD:
        suffix = "-FIX" if fix_atom else ""
        ckpt_path = MODEL_HOME / f"{model_type}{suffix}.ckpt"
        if ckpt_path.exists():
            print(f"  already present: {ckpt_path.name}")
            skipped += 1
        else:
            download_ckpt_file(model_type, ckpt_path, fix_atom=fix_atom)
            if ckpt_path.exists():
                size_mb = ckpt_path.stat().st_size / 1024 / 1024
                print(f"  OK ({size_mb:.1f} MB): {ckpt_path.name}")
                downloaded += 1
            else:
                print(f"  ERROR: download produced no file at {ckpt_path}", file=sys.stderr)
                sys.exit(1)

    print(f"\nDone. {downloaded} downloaded, {skipped} already present.")


if __name__ == "__main__":
    main()
