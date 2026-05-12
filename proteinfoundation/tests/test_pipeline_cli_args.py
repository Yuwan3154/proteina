"""Tests for the unified prediction pipeline CLI.

The historical `run_full_pipeline.py` has been merged into
`run_prediction_pipeline.py`; this file tests both the prediction-mode
(--input) and evaluation-mode (--dataset_file + --cif_dir) flows of the
merged driver.
"""

from proteinfoundation.prediction_pipeline import pipeline_cli_utils
from proteinfoundation.prediction_pipeline import run_prediction_pipeline as rpp
from proteinfoundation.prediction_pipeline.run_prediction_pipeline import build_parser


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
