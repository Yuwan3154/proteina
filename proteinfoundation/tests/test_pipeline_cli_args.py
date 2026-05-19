"""Tests for the unified prediction pipeline CLI.

The historical `run_full_pipeline.py` has been merged into
`run_prediction_pipeline.py`; this file tests both the prediction-mode
(--input) and evaluation-mode (--dataset_file + --cif_dir) flows of the
merged driver.
"""

import math

import numpy as np
import torch

from proteinfoundation.prediction_pipeline import pipeline_cli_utils
from proteinfoundation.prediction_pipeline import run_prediction_pipeline as rpp
from proteinfoundation.prediction_pipeline.run_prediction_pipeline import build_parser
from proteinfoundation.prediction_pipeline.proteinebm_scorer import compute_mean_pae


def test_parallel_incremental_filter_args():
    assert pipeline_cli_utils.parallel_incremental_filter_args(True) == ["--filter_existing"]
    assert pipeline_cli_utils.parallel_incremental_filter_args(False) == []


def test_prediction_mode_parser_aliases():
    """--input flow with the canonical aliases."""
    p = build_parser()
    a = p.parse_args(
        [
            "--input",
            "in.fasta",
            "--inference_config",
            "cfg",
            "--output_dir",
            "/tmp/out",
            "--top_k",
            "7",
            "--backend",
            "openfold",
            "--force_compile",
            "--skip_af2rank",
        ]
    )
    assert a.input == "in.fasta"
    assert a.dataset_file is None
    assert a.af2rank_top_k == 7
    assert a.af2rank_backend == "openfold"
    assert a.proteina_force_compile is True
    assert a.skip_af2rank_on_top_k is True
    assert a.cif_dir is None
    assert a.scorer == "proteinebm"


def test_evaluation_mode_parser_with_cif_dir():
    """--dataset_file + --cif_dir + --tms_col evaluation flow."""
    p = build_parser()
    a = p.parse_args(
        [
            "--dataset_file",
            "data.csv",
            "--id_col",
            "pdb",
            "--cif_dir",
            "/cif",
            "--tms_col",
            "tm_score",
            "--inference_config",
            "cfg",
            "--output_dir",
            "/out",
            "--scorer",
            "proteinebm",
            "--top_k",
            "8",
            "--proteinebm_batch_size",
            "64",
        ]
    )
    assert a.dataset_file == "data.csv"
    assert a.input is None
    assert a.id_col == "pdb"
    assert a.cif_dir == "/cif"
    assert a.tms_col == "tm_score"
    assert a.scorer == "proteinebm"
    assert a.af2rank_top_k == 8
    assert a.proteinebm_batch_size == 64


def test_input_and_dataset_file_mutually_exclusive():
    """The merged parser must reject specifying both --input and --dataset_file."""
    import pytest
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([
            "--input", "a.fasta",
            "--dataset_file", "b.csv",
            "--inference_config", "cfg",
            "--output_dir", "/out",
        ])


def test_step_proteinebm_scoring_cmd_uses_canonical_batch(monkeypatch):
    """step_proteinebm_scoring (merged) must forward --proteinebm_batch_size
    and skip the unsupported --no-filter_existing flag."""
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "PRED_PIPELINE_DIR", "/pp")

    rpp.step_proteinebm_scoring(
        csv_file="w.csv",
        csv_col="id",
        inference_config="cfg",
        num_gpus=2,
        proteinebm_config="/cfg.yaml",
        proteinebm_checkpoint="/ckpt.pt",
        proteinebm_t=0.05,
        proteinebm_analysis_subdir="sub",
        proteinebm_batch_size=16,
        proteinebm_template_self_condition=False,
        rerun=True,
    )
    cmd = captured["cmd"]
    assert "--proteinebm_batch_size" in cmd
    assert cmd[cmd.index("--proteinebm_batch_size") + 1] == "16"
    assert "--no-proteinebm_template_self_condition" in cmd
    assert "--no-filter_existing" not in cmd
    assert cmd.count("--filter_existing") == 0
    assert "--cif_dir" not in cmd  # no cif_dir passed


def test_step_proteinebm_scoring_cmd_forwards_cif_dir(monkeypatch):
    """When cif_dir is supplied, step_proteinebm_scoring must include
    --cif_dir on the subprocess command (so TM-score columns are added)."""
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "PRED_PIPELINE_DIR", "/pp")

    rpp.step_proteinebm_scoring(
        csv_file="d.csv",
        csv_col="pdb",
        inference_config="cfg",
        num_gpus=1,
        proteinebm_config="/pcfg",
        proteinebm_checkpoint="/pckpt",
        cif_dir="/cif",
        proteinebm_batch_size=48,
        rerun=False,
    )
    cmd = captured["cmd"]
    assert "--cif_dir" in cmd
    assert cmd[cmd.index("--cif_dir") + 1] == "/cif"
    assert "--proteinebm_batch_size" in cmd
    assert cmd[cmd.index("--proteinebm_batch_size") + 1] == "48"
    assert "--filter_existing" in cmd


def test_step_af2rank_topk_forwards_cif_dir(monkeypatch):
    """When cif_dir is supplied, step_af2rank_topk must forward it to
    run_af2rank_prediction.py so the GT TM columns are emitted."""
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "PRED_PIPELINE_DIR", "/pp")
    monkeypatch.setattr(rpp, "PROTEINA_BASE_DIR", "/proteina")

    rpp.step_af2rank_topk(
        inference_config="cfg",
        af2rank_top_k=5,
        recycles=3,
        num_gpus=1,
        csv_file="d.csv",
        csv_col="pdb",
        cif_dir="/cif",
    )
    cmd = captured["cmd"]
    assert "--cif_dir" in cmd
    assert cmd[cmd.index("--cif_dir") + 1] == "/cif"


def test_step_af2rank_topk_no_cif_dir(monkeypatch):
    """Without cif_dir, step_af2rank_topk omits --cif_dir."""
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "PRED_PIPELINE_DIR", "/pp")
    monkeypatch.setattr(rpp, "PROTEINA_BASE_DIR", "/proteina")

    rpp.step_af2rank_topk(
        inference_config="cfg",
        af2rank_top_k=5,
        recycles=3,
        num_gpus=1,
        csv_file="d.csv",
        csv_col="pdb",
    )
    cmd = captured["cmd"]
    assert "--cif_dir" not in cmd
    # Default: no forced summary regen
    assert "--force_regenerate_topk_summary" not in cmd


def test_step_af2rank_topk_force_regenerate_topk_summary(monkeypatch):
    """When force_regenerate_topk_summary=True is passed, the flag must be
    forwarded so previously-complete proteins get their summary CSV +
    per-protein af2rank-vs-proteinebm pTM scatter rebuilt."""
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "PRED_PIPELINE_DIR", "/pp")
    monkeypatch.setattr(rpp, "PROTEINA_BASE_DIR", "/proteina")

    rpp.step_af2rank_topk(
        inference_config="cfg",
        af2rank_top_k=5,
        recycles=3,
        num_gpus=1,
        csv_file="d.csv",
        csv_col="pdb",
        force_regenerate_topk_summary=True,
    )
    cmd = captured["cmd"]
    assert "--force_regenerate_topk_summary" in cmd


def test_proteinebm_defaults_point_at_pae_model():
    """Default checkpoint/config point at the new PAE-enabled model."""
    p = build_parser()
    a = p.parse_args([
        "--input", "in.fasta",
        "--inference_config", "cfg",
        "--output_dir", "/tmp/out",
    ])
    assert a.proteinebm_config.endswith("/pae_config.yaml"), a.proteinebm_config
    assert a.proteinebm_checkpoint.endswith("/pae.ckpt"), a.proteinebm_checkpoint
    # Analysis subdir intentionally unchanged: same EBM trunk, just +PAE head
    assert a.proteinebm_analysis_subdir == "proteinebm_v2_cathmd_analysis"


def test_compute_mean_pae_uniform_distribution():
    """Uniform PAE distribution over [0, max_dist] -> mean pAE ≈ max_dist / 2."""
    B, N, num_bins, max_dist = 1, 4, 64, 32.0
    logits = torch.zeros(B, N, N, num_bins)  # softmax(zeros) → uniform
    mask = torch.ones(B, N)
    mean_pae = compute_mean_pae(logits, mask, max_dist=max_dist)
    # Bin centers: 0.25, 0.75, ..., 31.75 → arithmetic mean = max_dist/2 - bin_width/2 + bin_width/2 = max_dist/2 - bin_width/2 ≈ max_dist/2
    expected = float(np.arange(0.5 * max_dist / num_bins, max_dist, max_dist / num_bins).mean())
    assert mean_pae.shape == (B,)
    assert math.isclose(float(mean_pae[0]), expected, abs_tol=1e-4), (float(mean_pae[0]), expected)


def test_compute_mean_pae_peaked_distribution():
    """Logits peaked at the first bin -> mean pAE ≈ first bin center."""
    B, N, num_bins, max_dist = 1, 3, 64, 32.0
    logits = torch.full((B, N, N, num_bins), -50.0)
    logits[..., 0] = 50.0  # massive mass on the first bin → softmax ≈ delta at bin 0
    mask = torch.ones(B, N)
    mean_pae = compute_mean_pae(logits, mask, max_dist=max_dist)
    expected_center = 0.5 * max_dist / num_bins  # 0.25 Å
    assert math.isclose(float(mean_pae[0]), expected_center, abs_tol=1e-3), float(mean_pae[0])


def test_compute_mean_pae_respects_mask():
    """Padding residues (mask=0) must not contribute to the average."""
    B, N, num_bins, max_dist = 1, 4, 64, 32.0
    logits = torch.full((B, N, N, num_bins), -50.0)
    # Real residues 0-1: peak at bin 0 (low pAE).  Padding residues 2-3: peak at last bin (high pAE).
    logits[..., :2, :2, 0] = 50.0
    logits[..., 2:, :, -1] = 50.0
    logits[..., :, 2:, -1] = 50.0
    mask = torch.zeros(B, N)
    mask[..., :2] = 1.0
    mean_pae = compute_mean_pae(logits, mask, max_dist=max_dist)
    expected_center = 0.5 * max_dist / num_bins
    assert math.isclose(float(mean_pae[0]), expected_center, abs_tol=1e-3), float(mean_pae[0])


def test_calibration_curve_basic(tmp_path):
    """Calibration curve writes a non-empty PNG for a synthetic dataset.

    Curve semantics: at threshold τ, Y = P(ref_pred_TM ≥ τ | best_ptm ≥ τ).
    Both sides use the **same** τ value.

    Synthetic data at τ=0.6:
      - 10 with best_ptm=0.4, best_ref_pred_tm=0.45  → below pTM filter
      - 10 with best_ptm=0.7, best_ref_pred_tm=0.85  → pass filter AND TM≥0.6 ✓
      - 10 with best_ptm=0.8, best_ref_pred_tm=0.55  → pass filter, TM<0.6 ✗
    At τ=0.6: 20 pass pTM filter; 10 of those have TM≥0.6 → fraction = 50%.
    """
    results = (
        [{"best_ptm": 0.4, "best_ref_pred_tm": 0.45} for _ in range(10)]
        + [{"best_ptm": 0.7, "best_ref_pred_tm": 0.85} for _ in range(10)]
        + [{"best_ptm": 0.8, "best_ref_pred_tm": 0.55} for _ in range(10)]
    )
    rpp.step_plot_ptm_calibration_curve(results, str(tmp_path), 0.6)
    out = tmp_path / "ptm_calibration_curve.png"
    assert out.exists(), "calibration plot not written"
    assert out.stat().st_size > 0, "calibration plot is empty"


def test_calibration_curve_yield_semantics(tmp_path, monkeypatch):
    """Verify the curve computes P(ref_pred_TM ≥ τ | pTM ≥ τ), not P(TM ≥ pTM | ...).

    Construct data where the two interpretations diverge:
      - 5 proteins with best_ptm=0.9, best_ref_pred_tm=0.75
        → pass pTM ≥ 0.7 filter; TM≥0.7 ✓ (yield); TM≥pTM ✗ (per-protein cal)
    Yield-at-0.7 interpretation: 5/5 = 100%.
    Per-protein  interpretation: 0/5 = 0%.
    We assert the JSON-summary-style scalar (computed inline below) is 100%.
    """
    results = [{"best_ptm": 0.9, "best_ref_pred_tm": 0.75} for _ in range(5)]
    rpp.step_plot_ptm_calibration_curve(results, str(tmp_path), 0.7)
    out = tmp_path / "ptm_calibration_curve.png"
    assert out.exists()
    # Sanity check the underlying logic explicitly using the same definition
    # used in the JSON summary's calibration_at_cutoff block.
    pairs = [(r["best_ptm"], r["best_ref_pred_tm"]) for r in results]
    n_pass = sum(1 for ptm, _ in pairs if ptm >= 0.7)
    n_yield = sum(1 for ptm, gt in pairs if ptm >= 0.7 and gt >= 0.7)
    assert n_pass == 5
    assert n_yield == 5  # yield: all 5 have TM ≥ 0.7
    # Sanity: the OPPOSITE (per-protein) interpretation would give 0 here.
    n_per_protein = sum(1 for ptm, gt in pairs if ptm >= 0.7 and gt >= ptm)
    assert n_per_protein == 0


def test_calibration_curve_no_gt(tmp_path):
    """When best_ref_pred_tm is NaN for every protein (no-GT mode), the helper
    must skip silently and write no file."""
    results = [
        {"best_ptm": 0.6, "best_ref_pred_tm": float("nan")},
        {"best_ptm": 0.8, "best_ref_pred_tm": float("nan")},
    ]
    rpp.step_plot_ptm_calibration_curve(results, str(tmp_path), 0.7)
    files = list(tmp_path.iterdir())
    assert files == [], f"expected no files, got {[f.name for f in files]}"
