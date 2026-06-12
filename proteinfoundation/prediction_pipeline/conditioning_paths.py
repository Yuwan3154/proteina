"""Single source of truth for the proteina inference-output directory layout.

The *conditioning mode* selects the per-config subdir that holds a protein's
samples. It is REQUIRED everywhere — the inference writes it, scoring + analysis
read it — with NO silent fallback, so the inference and the downstream steps can
never disagree on which folder the samples live in.

    seq       -> inference/{config}/seq_cond/      (seq_cond_segment if PROTEINA_SEGMENT_MODE=joint)
    seq_cath  -> inference/{config}/seq_cath_cond/
    legacy    -> inference/{config}/legacy/        (explicit only — for reading pre-namespaced runs)

Mode source: inference takes it from --conditioning_mode; scoring/analysis read
PROTEINA_CONDITIONING_MODE (the orchestrator propagates one to the other).
"""
import os

VALID_CONDITIONING_MODES = ("seq", "seq_cath", "legacy")


def conditioning_label(mode):
    """Map a conditioning mode to its inference-output subdir; raise if unset/unknown."""
    if not mode:
        raise ValueError(
            "conditioning mode is required (no silent fallback): pass --conditioning_mode "
            f"or set PROTEINA_CONDITIONING_MODE to one of {VALID_CONDITIONING_MODES}"
        )
    if mode == "seq":
        return "seq_cond_segment" if os.environ.get("PROTEINA_SEGMENT_MODE", "") == "joint" else "seq_cond"
    if mode == "seq_cath":
        return "seq_cath_cond"
    if mode == "legacy":
        return "legacy"
    raise ValueError(f"unknown conditioning mode {mode!r}; expected one of {VALID_CONDITIONING_MODES}")


def conditioning_mode_from_env():
    """Return PROTEINA_CONDITIONING_MODE, raising if unset (no silent fallback)."""
    mode = os.environ.get("PROTEINA_CONDITIONING_MODE", "").strip()
    if not mode:
        raise ValueError(
            "PROTEINA_CONDITIONING_MODE is required (no silent fallback): set it to one of "
            f"{VALID_CONDITIONING_MODES}"
        )
    return mode


def inference_base_dir(base_dir, inference_config, mode):
    """<base_dir>/inference/<inference_config>/<conditioning subdir>."""
    return os.path.join(base_dir, "inference", inference_config, conditioning_label(mode))


def protein_output_dir(base_dir, inference_config, protein_name, mode):
    """Per-protein sample dir: inference_base_dir/<protein_name>."""
    return os.path.join(inference_base_dir(base_dir, inference_config, mode), protein_name)
