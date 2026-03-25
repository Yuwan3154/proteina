"""Tests for unified CLI between run_full_pipeline and run_prediction_pipeline."""

from proteinfoundation.af2rank_evaluation import pipeline_cli_utils
from proteinfoundation.af2rank_evaluation import run_full_pipeline as rfp_mod
from proteinfoundation.af2rank_evaluation.run_full_pipeline import build_parser as build_full_parser
from proteinfoundation.prediction_pipeline.run_prediction_pipeline import build_parser as build_pred_parser
from proteinfoundation.prediction_pipeline import run_prediction_pipeline as rpp


def test_parallel_incremental_filter_args():
    assert pipeline_cli_utils.parallel_incremental_filter_args(True) == ["--filter_existing"]
    assert pipeline_cli_utils.parallel_incremental_filter_args(False) == []


def test_full_pipeline_parser_aliases_and_proteinebm_batch_size():
    p = build_full_parser()
    a = p.parse_args(
        [
            "--dataset_file",
            "d.csv",
            "--id_column",
            "pdb",
            "--cif_dir",
            "/cif",
            "--inference_config",
            "cfg",
            "--tms_column",
            "tm",
            "--cross_protein_output_dir",
            "/out",
            "--csv_file",
            "legacy.csv",
            "--csv_column",
            "legacy_id",
            "--proteinebm_batch_size",
            "64",
        ]
    )
    assert a.dataset_file == "legacy.csv"
    assert a.id_column == "legacy_id"
    assert a.proteinebm_batch_size == 64


def test_prediction_pipeline_parser_aliases():
    p = build_pred_parser()
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
    assert a.af2rank_top_k == 7
    assert a.af2rank_backend == "openfold"
    assert a.proteina_force_compile is True
    assert a.skip_af2rank_on_top_k is True


def test_prediction_step_proteinebm_cmd_uses_canonical_batch_and_no_invalid_filter_flag(monkeypatch):
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rpp, "run_with_conda_env", fake_run)
    monkeypatch.setattr(rpp, "AF2RANK_EVAL_DIR", "/eval")

    rpp.step_proteinebm_scoring(
        "w.csv",
        "cfg",
        2,
        "/cfg.yaml",
        "/ckpt.pt",
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


def test_full_run_proteinebm_scoring_cmd_includes_batch(monkeypatch):
    captured = {}

    def fake_run(env_name, command_list, cwd=None, direct_python=False):
        captured["cmd"] = command_list
        return True

    monkeypatch.setattr(rfp_mod, "run_with_conda_env", fake_run)

    rfp_mod.run_proteinebm_scoring(
        "d.csv",
        "id",
        "/cif",
        "cfg",
        1,
        "/pcfg",
        "/pckpt",
        proteinebm_batch_size=48,
        rerun=False,
    )
    cmd = captured["cmd"]
    assert "--proteinebm_batch_size" in cmd
    assert cmd[cmd.index("--proteinebm_batch_size") + 1] == "48"
    assert "--filter_existing" in cmd
