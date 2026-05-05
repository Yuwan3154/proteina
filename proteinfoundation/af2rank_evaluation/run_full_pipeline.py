#!/usr/bin/env python3
"""
Complete AF2Rank Evaluation Pipeline

This script runs the complete pipeline:
1. Proteina inference (parallel across GPUs)
2. AF2Rank scoring (parallel across GPUs)
3. (Optional) AF2Rank scoring on ProteinEBM top-k templates (per protein)
4. Cross-protein plots (always)

Usage:
    python run_full_pipeline.py --dataset_file data.csv --id_col pdb --cif_dir /path/to/cif \\
        --inference_config config_name --num_gpus 4 --tms_col tms_single --cross_protein_output_dir out/

Shared flags with run_prediction_pipeline.py include --af2rank_backend, --af2rank_top_k, --proteina_force_compile,
--proteinebm_batch_size, --proteinebm_template_self_condition, and sharding options.
"""

import os
import sys
import argparse
import subprocess
import logging
import time
import shutil
from pathlib import Path

import pandas as pd

from proteinfoundation.af2rank_evaluation.pipeline_cli_utils import parallel_incremental_filter_args
from proteinfoundation.af2rank_evaluation.sharding_utils import (
    add_shard_args,
    build_shard_cli_args,
    lengths_from_csv,
    resolve_shard_args,
    shard_proteins,
    wait_for_completion,
    wait_for_step,
)
from proteinfoundation.af2rank_evaluation.protein_tar_utils import (
    ensure_protein_tar,
    pack_protein_dirs,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def _tar_cli_args(enabled: bool) -> list[str]:
    return ["--tar_protein_dirs"] if enabled else ["--no-tar_protein_dirs"]


def _dynamic_cli_args(enabled: bool, progress_check_workers: int | None) -> list[str]:
    args = ["--dynamic_resharding"] if enabled else ["--no-dynamic_resharding"]
    if progress_check_workers is not None:
        args.extend(["--progress_check_workers", str(progress_check_workers)])
    return args


def _cleanup_shard_sentinels(output_dir: Path, num_shards: int) -> None:
    removed = 0
    for shard_idx in range(num_shards):
        sentinel = output_dir / f".shard_{shard_idx}_of_{num_shards}_complete"
        if sentinel.exists():
            sentinel.unlink()
            removed += 1
    for sentinel in output_dir.glob(".step_*_shard_*_of_*_complete"):
        sentinel.unlink()
        removed += 1
    logger.info("Cleaned up %d shard completion sentinel file(s).", removed)

def detect_conda_executable():
    """Auto-detect conda executable path."""
    # Try common conda executables
    conda_candidates = ['conda', 'mamba', 'micromamba']
    
    for cmd in conda_candidates:
        conda_path = shutil.which(cmd)
        if conda_path:
            logger.info(f"🐍 Detected {cmd} at: {conda_path}")
            return conda_path
    
    # Try common installation paths
    common_paths = [
        '~/miniforge3/bin/conda',
        '~/miniconda3/bin/conda',
        '~/anaconda3/bin/conda',
        '/opt/conda/bin/conda',
        '/usr/local/bin/conda'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            logger.info(f"🐍 Found conda at: {path}")
            return path
    
    logger.error("❌ Could not find conda executable")
    return None

def get_conda_env_path(conda_exe, env_name):
    """Get the full path to a conda environment."""
    try:
        result = subprocess.run([conda_exe, 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        
        for line in result.stdout.split('\n'):
            if env_name in line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2 and parts[0] == env_name:
                    env_path = parts[-1]
                    logger.info(f"🔍 Found {env_name} environment at: {env_path}")
                    return env_path
        
        logger.error(f"❌ Environment '{env_name}' not found in conda env list")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to get conda environment info: {e}")
        return None

_SHARD_ENV_VARS = ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_COUNT", "LLSUB_RANK", "LLSUB_SIZE")


def run_with_conda_env(env_name, command_list, cwd=None, direct_python: bool = False):
    """Run a command. If direct_python, use current Python; else use shell script wrappers.

    SLURM/LLsub sharding env vars are always stripped from child subprocess env so that
    children receive sharding instructions only via explicit CLI args (--shard_index /
    --num_shards), never via env var inheritance from this orchestrator process.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    child_env = {k: v for k, v in os.environ.items() if k not in _SHARD_ENV_VARS}

    if direct_python:
        cmd = [sys.executable] + command_list[1:]
        effective_cwd = cwd if cwd is not None else script_dir
        logger.info(f"🚀 Running with current Python: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, cwd=effective_cwd, check=False, env=child_env)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"❌ Failed to run command: {e}")
            return False

    if env_name == 'proteina':
        wrapper_script = os.path.join(script_dir, 'run_with_proteina_env.sh')
    elif env_name == 'colabdesign':
        wrapper_script = os.path.join(script_dir, 'run_with_colabdesign_env.sh')
    elif env_name == 'proteinebm':
        wrapper_script = os.path.join(script_dir, 'run_with_proteinebm_env.sh')
    else:
        logger.error(f"❌ Unknown environment: {env_name}")
        return False

    if not os.path.exists(wrapper_script):
        logger.error(f"❌ Wrapper script not found: {wrapper_script}")
        return False

    cmd = [wrapper_script] + command_list
    logger.info(f"🚀 Running in {env_name} environment: {' '.join(command_list)}")
    effective_cwd = cwd if cwd is not None else script_dir
    try:
        result = subprocess.run(cmd, cwd=effective_cwd, check=False, env=child_env)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"❌ Failed to run command: {e}")
        return False

def run_proteina_inference(csv_file, csv_col, cif_dir, inference_config, num_gpus, usalign_path=None, force_compile: bool = False, shard_args=None, direct_python: bool = False, rerun: bool = False, tar_protein_dirs: bool = True, dynamic_resharding: bool = True, progress_check_workers: int | None = None):
    """Run the Proteina inference pipeline."""
    logger.info("🧬 Starting Proteina inference pipeline...")

    cmd = [
        'python', 'parallel_proteina_inference.py',
        '--csv_file', csv_file,
        '--csv_col', csv_col,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        *([] if rerun else ['--skip_existing'])
    ]
    if force_compile:
        cmd.append('--force_compile')
    if usalign_path:
        cmd.extend(['--usalign_path', usalign_path])
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))

    return run_with_conda_env('proteina', cmd, direct_python=direct_python)

def run_af2rank_scoring(csv_file, csv_col, cif_dir, inference_config, num_gpus, recycles=3, regenerate_plots=False, backend="colabdesign", use_deepspeed_evoformer_attention=True, use_cuequivariance_attention=True, use_cuequivariance_multiplicative_update=True, shard_args=None, direct_python: bool = False, rerun: bool = False, tar_protein_dirs: bool = True, dynamic_resharding: bool = True, progress_check_workers: int | None = None):
    """Run the AF2Rank scoring pipeline."""
    logger.info(f"⚡ Starting AF2Rank scoring pipeline (backend={backend})...")

    cmd = [
        'python', 'parallel_af2rank_scoring.py',
        '--csv_file', csv_file,
        '--csv_col', csv_col,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--recycles', str(recycles),
        *parallel_incremental_filter_args(not rerun),
        '--af2rank_backend', backend,
    ]
    if regenerate_plots:
        cmd.append('--regenerate_plots')
    if backend == 'openfold':
        if not use_deepspeed_evoformer_attention:
            cmd.append('--no-use_deepspeed_evoformer_attention')
        if not use_cuequivariance_attention:
            cmd.append('--no-use_cuequivariance_attention')
        if not use_cuequivariance_multiplicative_update:
            cmd.append('--no-use_cuequivariance_multiplicative_update')
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))

    env_name = 'proteina' if backend == 'openfold' else 'colabdesign'
    return run_with_conda_env(env_name, cmd, direct_python=direct_python)

def run_proteinebm_scoring(
    csv_file,
    csv_col,
    cif_dir,
    inference_config,
    num_gpus,
    proteinebm_config,
    proteinebm_checkpoint,
    proteinebm_template_self_condition=True,
    proteinebm_analysis_subdir='proteinebm_v2_cathmd_analysis',
    proteinebm_t=0.05,
    proteinebm_batch_size: int = 32,
    num_workers=None,
    shard_args=None,
    direct_python: bool = False,
    rerun: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
):
    """Run the ProteinEBM scoring pipeline."""
    logger.info("💸 Starting ProteinEBM scoring pipeline...")

    cmd = [
        'python', 'parallel_proteinebm_scoring.py',
        '--csv_file', csv_file,
        '--csv_col', csv_col,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        *parallel_incremental_filter_args(not rerun),
        '--proteinebm_config', proteinebm_config,
        '--proteinebm_checkpoint', proteinebm_checkpoint,
        '--proteinebm_analysis_subdir', proteinebm_analysis_subdir,
        '--proteinebm_t', str(proteinebm_t),
        '--proteinebm_batch_size', str(proteinebm_batch_size),
    ]
    if num_workers is not None:
        cmd.extend(['--num_workers', str(num_workers)])
    if not proteinebm_template_self_condition:
        cmd.append('--no-proteinebm_template_self_condition')
    if direct_python:
        cmd.append('--direct_python')
    if shard_args:
        cmd.extend(shard_args)
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))

    return run_with_conda_env('proteinebm', cmd, direct_python=direct_python)

def run_cross_protein_plots(
    inference_dir: str,
    output_dir: str,
    scorer: str,
    dataset_file: str,
    id_col: str,
    tms_col: str,
    af2rank_top_k: int,
    proteinebm_plot_mode: str = "tm",
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    direct_python: bool = False,
) -> bool:
    """Run generate_cross_protein_plots.py for the requested scorer."""
    cmd = [
        "python",
        "generate_cross_protein_plots.py",
        "--inference_dir",
        inference_dir,
        "--output_dir",
        output_dir,
        "--scorer",
        scorer,
        "--dataset_file",
        dataset_file,
        "--id_col",
        id_col,
        "--tms_col",
        tms_col,
        "--proteinebm_analysis_subdir",
        proteinebm_analysis_subdir,
    ]

    if scorer == "proteinebm":
        cmd.extend(["--proteinebm_plot_mode", proteinebm_plot_mode])

    if scorer == "af2rank_on_proteinebm_topk":
        cmd.extend(["--af2rank_top_k", str(int(af2rank_top_k))])

    return run_with_conda_env("proteina", cmd, cwd=os.path.dirname(os.path.abspath(__file__)), direct_python=direct_python)


def run_af2rank_on_proteinebm_topk(
    inference_dir: str,
    dataset_file: str,
    id_col: str,
    tms_col: str,
    top_k: int,
    recycles: int,
    num_gpus: int,
    filter_existing: bool = True,
    proteinebm_analysis_subdir: str = "proteinebm_v2_cathmd_analysis",
    backend: str = "colabdesign",
    use_deepspeed_evoformer_attention: bool = True,
    use_cuequivariance_attention: bool = True,
    use_cuequivariance_multiplicative_update: bool = True,
    shard_args=None,
    direct_python: bool = False,
    cif_dir: str = "",
    output_dir: str = "",
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
) -> bool:
    """Run AF2Rank scoring on the ProteinEBM top-k templates per protein."""
    logger.info(f"🧪 Starting AF2Rank-on-ProteinEBM-topk step (backend={backend})...")

    if top_k <= 0:
        logger.error("top_k must be > 0 for AF2Rank-on-ProteinEBM-topk")
        return False

    cmd = [
        "python",
        "run_af2rank_on_proteinebm_topk.py",
        "--inference_dir",
        inference_dir,
        "--top_k",
        str(int(top_k)),
        "--recycles",
        str(int(recycles)),
        "--num_gpus",
        str(int(num_gpus)),
        "--proteinebm_analysis_subdir",
        proteinebm_analysis_subdir,
        "--af2rank_backend",
        backend,
    ]
    if backend == "openfold":
        if not use_deepspeed_evoformer_attention:
            cmd.append("--no-use_deepspeed_evoformer_attention")
        if not use_cuequivariance_attention:
            cmd.append("--no-use_cuequivariance_attention")
        if not use_cuequivariance_multiplicative_update:
            cmd.append("--no-use_cuequivariance_multiplicative_update")
    if cif_dir:
        cmd.extend(["--cif_dir", cif_dir])
    if dataset_file:
        cmd.extend(
            [
                "--dataset_file",
                dataset_file,
                "--id_col",
                id_col,
                "--tms_col",
                tms_col,
            ]
        )
    if filter_existing:
        cmd.append("--filter_existing")
    else:
        cmd.append("--no-filter_existing")
    if direct_python:
        cmd.append("--direct_python")
    if shard_args:
        cmd.extend(shard_args)
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))

    return run_with_conda_env("proteina", cmd, cwd=os.path.dirname(os.path.abspath(__file__)), direct_python=direct_python)


def run_central_analysis(
    inference_dir: str,
    dataset_file: str,
    id_col: str,
    cif_dir: str,
    proteinebm_analysis_subdir: str,
    num_workers: int | None = None,
    shard_args: list | None = None,
    direct_python: bool = False,
    rerun: bool = False,
    skip_diversity: bool = False,
    tar_protein_dirs: bool = True,
    dynamic_resharding: bool = True,
    progress_check_workers: int | None = None,
) -> bool:
    cmd = [
        "python",
        "proteina_analysis.py",
        "--inference_dir",
        inference_dir,
        "--csv_file",
        dataset_file,
        "--csv_col",
        id_col,
        "--cif_dir",
        cif_dir,
        "--proteinebm_analysis_subdir",
        proteinebm_analysis_subdir,
    ]
    if num_workers is not None:
        cmd.extend(["--num_workers", str(num_workers)])
    if shard_args:
        cmd.extend(shard_args)
    if rerun:
        cmd.append("--rerun")
    if skip_diversity:
        cmd.append("--skip_diversity")
    cmd.extend(_tar_cli_args(tar_protein_dirs))
    cmd.extend(_dynamic_cli_args(dynamic_resharding, progress_check_workers))
    return run_with_conda_env("proteina", cmd, cwd=os.path.dirname(os.path.abspath(__file__)), direct_python=direct_python)


def build_parser() -> argparse.ArgumentParser:
    """CLI for the full AF2Rank evaluation pipeline (shared flag names align with run_prediction_pipeline)."""
    parser = argparse.ArgumentParser(description='Complete AF2Rank Evaluation Pipeline')
    # Prefer consistent naming: dataset_file + id_col.
    # Keep --csv_file/--csv_col as hidden aliases for backward compatibility.
    parser.add_argument('--dataset_file', required=True, help='Path to dataset CSV file with protein ids and reference TM scores')
    parser.add_argument('--id_col', required=True, help='Column name in dataset CSV to use as protein ID (e.g. pdb)')
    parser.add_argument('--csv_file', dest='dataset_file', help=argparse.SUPPRESS)
    parser.add_argument('--csv_col', dest='id_col', help=argparse.SUPPRESS)
    parser.add_argument('--cif_dir', required=True, help='Directory containing CIF files')
    parser.add_argument('--inference_config', required=True, help='Inference configuration name')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--scorer', choices=['af2rank', 'proteinebm'], default='af2rank',
                       help='Which scoring backend to use in step 2')
    parser.add_argument('--recycles', type=int, default=3, help='Number of AF2 recycles for scoring')
    parser.add_argument(
        '--af2rank_top_k',
        type=int,
        default=0,
        help='If > 0 and --scorer=proteinebm, run AF2Rank on the ProteinEBM top-k templates per protein (rank by energy, then AF2Rank ranks by pTM)',
    )
    parser.add_argument(
        '--top_k',
        dest='af2rank_top_k',
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--af2rank_topk_filter_existing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Skip proteins whose af2rank_on_proteinebm_topk outputs already exist.',
    )
    parser.add_argument('--af2rank_backend', choices=['colabdesign', 'openfold'], default='colabdesign',
                       help='AF2Rank backend: colabdesign (JAX) or openfold (PyTorch)')
    parser.add_argument(
        '--backend',
        dest='af2rank_backend',
        choices=['colabdesign', 'openfold'],
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--use_deepspeed_evoformer_attention', action=argparse.BooleanOptionalAction, default=False,
                       help='Use DeepSpeed evoformer attention (openfold backend, default: False)')
    parser.add_argument('--use_cuequivariance_attention', action=argparse.BooleanOptionalAction, default=False,
                       help='Use cuEquivariance attention kernels (openfold backend, default: False)')
    parser.add_argument('--use_cuequivariance_multiplicative_update', action=argparse.BooleanOptionalAction, default=False,
                       help='Use cuEquivariance multiplicative update (openfold backend, default: False)')
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip Proteina inference (only run scoring stage)')
    parser.add_argument('--skip_diversity', action='store_true',
                       help='Skip template-to-template diversity computation in central analysis')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip the central post-scoring TM analysis stage')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Max parallel CPU workers for central analysis and future CPU-parallel steps; '
             'default is clamped os.cpu_count() (1–64).',
    )
    parser.add_argument('--skip_scoring', action='store_true',
                       help='Skip scoring stage (AF2Rank or ProteinEBM depending on --scorer)')
    parser.add_argument('--skip_af2rank_on_top_k', action='store_true',
                       help='Skip AF2Rank-on-ProteinEBM-top-k step even if --af2rank_top_k > 0')
    parser.add_argument(
        '--skip_af2rank',
        dest='skip_af2rank_on_top_k',
        action='store_true',
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument('--regenerate_plots', action='store_true',
                       help='Regenerate AF2Rank plots even if scoring already completed')
    parser.add_argument('--rerun_proteina', action='store_true',
                       help='Force re-run Proteina inference even if outputs already exist')
    parser.add_argument('--rerun_score', action='store_true',
                       help='Force re-run scoring (AF2Rank or ProteinEBM) even if outputs already exist')
    parser.add_argument('--rerun_af2rank_on_top_k', action='store_true',
                       help='Force re-run AF2Rank on ProteinEBM top-k even if outputs already exist')
    parser.add_argument('--rerun_analysis', action='store_true',
                       help='Force re-run central analysis even if analysis summaries already exist')
    parser.add_argument(
        '--proteina_force_compile',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Pass --force_compile to Proteina inference (torch.compile even in eval/no_grad).',
    )
    parser.add_argument(
        '--force_compile',
        dest='proteina_force_compile',
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    add_shard_args(parser)
    parser.add_argument('--shard_poll_interval', type=int, default=60,
                        help='Seconds between polls when shard 0 waits for other shards (default: 60)')
    parser.add_argument('--shard_timeout', type=int, default=86400,
                        help='Max seconds for shard 0 to wait for completion (default: 86400)')
    parser.add_argument('--direct_python', action='store_true',
                        help='Use current Python interpreter for subprocesses instead of shell script wrappers. '
                             'Useful on HPC where conda env activation is slow; requires all deps in current env.')
    parser.add_argument(
        "--tar_protein_dirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-protein inference directories as uncompressed <protein_id>.tar archives (default: True).",
    )
    parser.add_argument(
        "--dynamic_resharding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At each step, filter global unfinished proteins before sharding to reduce idle shards (default: True).",
    )
    parser.add_argument(
        "--progress_check_workers",
        type=int,
        default=None,
        help="Thread workers for per-step progress checks (default: child script uses min(32, cpu_count * 4)).",
    )
    parser.add_argument('--tms_col', required=True,
                       help='Dataset column name to use as reference TM score (required, used for cross-protein plots)')
    parser.add_argument(
        '--cross_protein_output_dir',
        required=True,
        help='Output directory for cross-protein plots',
    )
    parser.add_argument(
        '--proteinebm_cross_protein_plot_mode',
        choices=['tm', 'energy'],
        default='tm',
        help='When --scorer=proteinebm, which ProteinEBM cross-protein plot mode to run in the final plotting step',
    )

    # ProteinEBM scoring options
    parser.add_argument('--proteinebm_checkpoint', default='/home/ubuntu/ProteinEBM/weights/model_1_frozen_1m_md.pt',
                       help='Path to ProteinEBM checkpoint to use for scoring')
    parser.add_argument('--proteinebm_config', default='/home/ubuntu/ProteinEBM/protein_ebm/config/base_pretrain.yaml',
                       help='Path to ProteinEBM base_pretrain.yaml config')
    parser.add_argument('--proteinebm_template_self_condition', action=argparse.BooleanOptionalAction, default=True,
                       help='Use template coordinates for self-conditioning (matches ProteinEBM --template_self_condition)')
    parser.add_argument('--proteinebm_analysis_subdir', default='proteinebm_v2_cathmd_analysis',
                       help='Per-protein subdir for ProteinEBM outputs (default: proteinebm_v2_cathmd_analysis; use proteinebm_analysis for legacy)')
    parser.add_argument('--proteinebm_t', type=float, default=0.05,
                       help='Diffusion time t for ProteinEBM scoring (default: 0.05)')
    parser.add_argument(
        '--proteinebm_batch_size',
        type=int,
        default=32,
        help='Batch size for ProteinEBM inference (default: 32). Passed to parallel_proteinebm_scoring.',
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    shard_index, num_shards = resolve_shard_args(args.shard_index, args.num_shards)
    shard_cli_args = build_shard_cli_args(shard_index, num_shards, len_col=args.len_col)
    if shard_index is not None:
        logger.info(f"Sharding enabled: shard {shard_index} of {num_shards}")

    # Validate inputs
    if not os.path.exists(args.dataset_file):
        logger.error(f"Dataset file not found: {args.dataset_file}")
        sys.exit(1)
    
    if not os.path.exists(args.cif_dir):
        logger.error(f"CIF directory not found: {args.cif_dir}")
        sys.exit(1)
    
    # Check available GPUs
    try:
        gpu_count = int(subprocess.check_output(['nvidia-smi', '--list-gpus']).decode().count('\n'))
    except:
        gpu_count = 1
        logger.warning("Could not detect GPUs, assuming 1 GPU available")
    
    if args.num_gpus > gpu_count:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {gpu_count} available, using {gpu_count}")
        args.num_gpus = gpu_count
    
    start_time = time.time()
    project_root = os.getcwd()
    base_inference_dir = os.path.join(project_root, "inference", args.inference_config)
    dataset_df = pd.read_csv(args.dataset_file)
    protein_ids_for_tar = dataset_df[args.id_col].dropna().astype(str).str.strip().unique().tolist()
    shard_protein_ids_for_tar = list(protein_ids_for_tar)
    if shard_index is not None:
        lengths = lengths_from_csv(args.dataset_file, args.id_col, args.len_col)
        if lengths is not None:
            shard_protein_ids_for_tar = shard_proteins(protein_ids_for_tar, shard_index, num_shards, lengths=lengths)
        else:
            data_dir = os.environ.get("DATA_PATH", os.path.join(project_root, "data"))
            shard_protein_ids_for_tar = shard_proteins(protein_ids_for_tar, shard_index, num_shards, data_dir=data_dir)
    if args.tar_protein_dirs:
        for protein_id in protein_ids_for_tar:
            ensure_protein_tar(base_inference_dir, protein_id)
        logger.info(f"Initialized {len(protein_ids_for_tar)} per-protein tar files under {base_inference_dir}")
    if args.cross_protein_output_dir:
        cross_out_dir = os.path.abspath(os.path.expanduser(args.cross_protein_output_dir))
    else:
        cross_out_dir = os.path.join(base_inference_dir, f"{Path(args.dataset_file).stem}_cross_protein_analysis")
    if shard_index is not None and num_shards is not None:
        os.makedirs(cross_out_dir, exist_ok=True)
        _own_sentinel = Path(cross_out_dir) / f".shard_{shard_index}_of_{num_shards}_complete"
        _own_sentinel.unlink(missing_ok=True)
        for _step_s in Path(cross_out_dir).glob(f".step_*_shard_{shard_index}_of_{num_shards}_complete"):
            _step_s.unlink()
        logger.info(f"Cleared own shard completion sentinel (shard {shard_index}/{num_shards}).")

    logger.info(f"🚀 Starting complete AF2Rank evaluation pipeline")
    logger.info(f"📊 Dataset file: {args.dataset_file}")
    logger.info(f"📂 CIF directory: {args.cif_dir}")
    logger.info(f"⚙️  Inference config: {args.inference_config}")
    logger.info(f"🔥 GPUs: {args.num_gpus}")
    logger.info(f"🧮 Scorer: {args.scorer}")
    logger.info(f"🔄 AF2Rank recycles: {args.recycles}")
    logger.info(f"🐍 Mode: {'direct Python' if args.direct_python else 'shell script wrappers'}")
    logger.info(f"🔑 ID column: {args.id_col}")
    logger.info(f"🔑 TM score column: {args.tms_col}")
    logger.info(f"🔧 AF2Rank backend: {args.af2rank_backend}")
    logger.info(f"📦 ProteinEBM batch size: {args.proteinebm_batch_size}")
    logger.info(f"📦 Tar protein dirs: {args.tar_protein_dirs}")
    logger.info(f"📦 Dynamic re-sharding: {args.dynamic_resharding}")
    from proteinfoundation.af2rank_evaluation.proteina_analysis import resolve_num_workers
    logger.info(f"🔩 num_workers (CPU, analysis etc.): {resolve_num_workers(args.num_workers)}")
    skip_analysis = args.skip_analysis
    
    success = True
    
    # Step 1: Proteina Inference
    if not args.skip_inference:
        logger.info("\n" + "="*60)
        logger.info("STEP 1: PROTEINA INFERENCE")
        logger.info("="*60)
        
        inference_success = run_proteina_inference(
            args.dataset_file,
            args.id_col,
            args.cif_dir,
            args.inference_config,
            args.num_gpus,
            args.usalign_path,
            force_compile=args.proteina_force_compile,
            shard_args=shard_cli_args,
            direct_python=args.direct_python,
            rerun=args.rerun_proteina,
            tar_protein_dirs=args.tar_protein_dirs,
            dynamic_resharding=args.dynamic_resharding,
            progress_check_workers=args.progress_check_workers,
        )
        
        if inference_success:
            logger.info("✅ Proteina inference completed successfully")
        else:
            logger.error("❌ Proteina inference failed")
            success = False
    else:
        logger.info("⏭️  Skipping Proteina inference")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            cross_out_dir, "inference", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # Step 2: Scoring (AF2Rank or ProteinEBM)
    if not args.skip_scoring and success:
        logger.info("\n" + "="*60)
        if args.scorer == 'af2rank':
            logger.info("STEP 2: AF2RANK SCORING")
        else:
            logger.info("STEP 2: PROTEINEBM SCORING")
        logger.info("="*60)
        
        if args.scorer == 'af2rank':
            scoring_success = run_af2rank_scoring(
                args.dataset_file,
                args.id_col,
                args.cif_dir,
                args.inference_config,
                args.num_gpus,
                args.recycles,
                args.regenerate_plots,
                backend=args.af2rank_backend,
                use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
                use_cuequivariance_attention=args.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
                shard_args=shard_cli_args,
                direct_python=args.direct_python,
                rerun=args.rerun_score,
                tar_protein_dirs=args.tar_protein_dirs,
                dynamic_resharding=args.dynamic_resharding,
                progress_check_workers=args.progress_check_workers,
            )
        else:
            scoring_success = run_proteinebm_scoring(
                csv_file=args.dataset_file,
                csv_col=args.id_col,
                cif_dir=args.cif_dir,
                inference_config=args.inference_config,
                num_gpus=args.num_gpus,
                proteinebm_config=args.proteinebm_config,
                proteinebm_checkpoint=args.proteinebm_checkpoint,
                proteinebm_template_self_condition=args.proteinebm_template_self_condition,
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                proteinebm_t=args.proteinebm_t,
                proteinebm_batch_size=args.proteinebm_batch_size,
                num_workers=args.num_workers,
                shard_args=shard_cli_args,
                direct_python=args.direct_python,
                rerun=args.rerun_score,
                tar_protein_dirs=args.tar_protein_dirs,
                dynamic_resharding=args.dynamic_resharding,
                progress_check_workers=args.progress_check_workers,
            )
        
        if scoring_success:
            logger.info(f"✅ Scoring completed successfully ({args.scorer})")
        else:
            logger.error(f"❌ Scoring failed ({args.scorer})")
            success = False
    elif args.skip_scoring:
        logger.info(f"⏭️  Skipping scoring stage ({args.scorer})")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            cross_out_dir, "scoring", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # Step 3: Optional AF2Rank scoring on ProteinEBM top-k templates
    if success and (not args.skip_scoring) and (not args.skip_af2rank_on_top_k) and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: AF2RANK ON PROTEINEBM TOP-K TEMPLATES")
        logger.info("="*60)

        inference_dir = os.path.join(project_root, "inference", args.inference_config)
        topk_success = run_af2rank_on_proteinebm_topk(
            inference_dir=inference_dir,
            dataset_file=args.dataset_file,
            id_col=args.id_col,
            tms_col=args.tms_col,
            top_k=int(args.af2rank_top_k),
            recycles=int(args.recycles),
            num_gpus=int(args.num_gpus),
            filter_existing=bool(args.af2rank_topk_filter_existing) and not args.rerun_af2rank_on_top_k,
            proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
            backend=args.af2rank_backend,
            use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
            use_cuequivariance_attention=args.use_cuequivariance_attention,
            use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
            shard_args=shard_cli_args,
            direct_python=args.direct_python,
            cif_dir=args.cif_dir,
            output_dir=cross_out_dir,
            tar_protein_dirs=args.tar_protein_dirs,
            dynamic_resharding=args.dynamic_resharding,
            progress_check_workers=args.progress_check_workers,
        )

        if topk_success:
            logger.info("✅ AF2Rank-on-ProteinEBM-topk completed successfully")
        else:
            logger.error("❌ AF2Rank-on-ProteinEBM-topk failed")
            success = False
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            cross_out_dir, "af2rank_topk", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    # Step 4: Central analysis (every shard runs its own subset)
    if success:
        logger.info("\n" + "="*60)
        logger.info("STEP 4: CENTRAL ANALYSIS")
        logger.info("="*60)

        inference_dir = os.path.join(project_root, "inference", args.inference_config)

        if success:
            if not skip_analysis:
                analysis_success = run_central_analysis(
                    inference_dir=inference_dir,
                    dataset_file=args.dataset_file,
                    id_col=args.id_col,
                    cif_dir=args.cif_dir,
                    proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                    num_workers=args.num_workers,
                    shard_args=shard_cli_args,
                    direct_python=args.direct_python,
                    rerun=args.rerun_proteina or args.rerun_score or args.rerun_af2rank_on_top_k or args.rerun_analysis,
                    skip_diversity=args.skip_diversity,
                    tar_protein_dirs=args.tar_protein_dirs,
                    dynamic_resharding=args.dynamic_resharding,
                    progress_check_workers=args.progress_check_workers,
                )
                if analysis_success:
                    shard_desc = f" (shard {shard_index})" if shard_index is not None else ""
                    _done_note = "; waiting for other shards..." if shard_index is not None else ""
                    logger.info(f"✅ Central analysis{shard_desc} completed successfully{_done_note}")
                else:
                    logger.error("❌ Central analysis failed")
                    success = False
            else:
                logger.info("⏭️  Skipping central analysis")
    if args.dynamic_resharding and shard_index is not None and num_shards is not None:
        success = wait_for_step(
            cross_out_dir, "analysis", num_shards, shard_index, success,
            poll_interval=args.shard_poll_interval, timeout=args.shard_timeout,
        ) and success

    if args.tar_protein_dirs and shard_index is not None and num_shards is not None:
        stats = pack_protein_dirs(base_inference_dir, shard_protein_ids_for_tar, delete_after=True)
        logger.info(f"Shard-owned tar finalization: {stats}")

    # Write per-shard sentinel so shard 0 can reliably know when all shards' per-protein work
    # is done — even on re-runs where stale per-protein output files may already exist.
    # Sentinels live in cross_out_dir (dataset-specific) rather than base_inference_dir so
    # concurrent runs on different datasets sharing the same inference config don't collide.
    if shard_index is not None and num_shards is not None:
        _sentinel = Path(cross_out_dir) / f".shard_{shard_index}_of_{num_shards}_complete"
        _sentinel.write_text("0" if success else "1")
        logger.info(f"Wrote shard {shard_index} completion sentinel ({'success' if success else 'failure'}).")

    if shard_index is not None and shard_index != 0:
        logger.info("Per-protein work complete (after central analysis), exiting.")
        sys.exit(0 if success else 1)

    if success and shard_index is not None:
        other_shard_indices = list(range(1, num_shards))
        if other_shard_indices:
            def _check_shard_done(shard_idx):
                sentinel = Path(cross_out_dir) / f".shard_{shard_idx}_of_{num_shards}_complete"
                return sentinel.exists()

            if not wait_for_completion(
                other_shard_indices,
                _check_shard_done,
                poll_interval=args.shard_poll_interval,
                timeout=args.shard_timeout,
                item_name="shards",
            ):
                logger.error("Timeout waiting for all shards to complete")
                success = False

            # Report any shards that completed but flagged failure
            failed_shards = []
            for idx in other_shard_indices:
                s = Path(cross_out_dir) / f".shard_{idx}_of_{num_shards}_complete"
                if s.exists() and s.read_text().strip() != "0":
                    failed_shards.append(idx)
            if failed_shards:
                logger.error(f"⚠️  {len(failed_shards)} shard(s) reported failure: {failed_shards}")
                logger.error("Aggregation will proceed but results may be incomplete.")

        # Aggregate AF2Rank-on-topk per-protein results into the cross-protein summary CSV.
        # During the sharded Step 3, each shard suppressed summary writing; now that all
        # per-protein CSVs exist, one aggregation pass (filter_existing=True, no shard_args)
        # reads them and writes the combined summary.
        if success and not args.skip_af2rank_on_top_k and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0:
            logger.info("Aggregating AF2Rank-on-topk per-protein results into cross-protein summary CSV...")
            run_af2rank_on_proteinebm_topk(
                inference_dir=inference_dir,
                dataset_file=args.dataset_file,
                id_col=args.id_col,
                tms_col=args.tms_col,
                top_k=int(args.af2rank_top_k),
                recycles=int(args.recycles),
                num_gpus=int(args.num_gpus),
                filter_existing=True,
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                backend=args.af2rank_backend,
                use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
                use_cuequivariance_attention=args.use_cuequivariance_attention,
                use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
                shard_args=None,
                direct_python=args.direct_python,
                cif_dir=args.cif_dir,
                output_dir=cross_out_dir,
                tar_protein_dirs=args.tar_protein_dirs,
            )

    # Step 5: Cross-protein plots
    if success:
        logger.info("\n" + "="*60)
        logger.info("STEP 5: CROSS-PROTEIN PLOTS")
        logger.info("="*60)

        inference_dir = os.path.join(project_root, "inference", args.inference_config)

        if success:
            plot_success = run_cross_protein_plots(
                inference_dir=inference_dir,
                output_dir=cross_out_dir,
                scorer=args.scorer,
                dataset_file=args.dataset_file,
                id_col=args.id_col,
                tms_col=args.tms_col,
                af2rank_top_k=int(args.af2rank_top_k),
                proteinebm_plot_mode=str(args.proteinebm_cross_protein_plot_mode),
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                direct_python=args.direct_python,
            )
            if not plot_success:
                logger.error("❌ Cross-protein plotting failed")
                success = False

        if success and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0 and not args.skip_af2rank_on_top_k:
            plot_success_2 = run_cross_protein_plots(
                inference_dir=inference_dir,
                output_dir=cross_out_dir,
                scorer="af2rank_on_proteinebm_topk",
                dataset_file=args.dataset_file,
                id_col=args.id_col,
                tms_col=args.tms_col,
                af2rank_top_k=int(args.af2rank_top_k),
                proteinebm_plot_mode=str(args.proteinebm_cross_protein_plot_mode),
                proteinebm_analysis_subdir=args.proteinebm_analysis_subdir,
                direct_python=args.direct_python,
            )
            if not plot_success_2:
                # This commonly occurs if AF2Rank-on-ProteinEBM-top-k hasn't been run yet
                # for this inference_dir/dataset combination. Treat as a warning so the
                # pipeline can still complete after producing the primary plots.
                logger.warning("⚠️  Cross-protein plotting skipped/failed (af2rank_on_proteinebm_topk)")

    if shard_index == 0 and num_shards is not None:
        _cleanup_shard_sentinels(Path(cross_out_dir), num_shards)
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    if success:
        logger.info(f"🎉 Pipeline completed successfully in {total_time:.1f}s")
        # Construct results path dynamically
        results_path = os.path.join(project_root, "inference", args.inference_config)
        logger.info(f"📁 Results should be available at: {results_path}/")
    else:
        logger.error(f"💥 Pipeline failed after {total_time:.1f}s")
    
    logger.info("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
