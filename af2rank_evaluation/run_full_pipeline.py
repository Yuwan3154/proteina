#!/usr/bin/env python3
"""
Complete AF2Rank Evaluation Pipeline

This script runs the complete pipeline:
1. Proteina inference (parallel across GPUs)
2. AF2Rank scoring (parallel across GPUs)
3. (Optional) AF2Rank scoring on ProteinEBM top-k templates (per protein)
4. Cross-protein plots (always)

Usage:
    python run_full_pipeline.py --csv_file data.csv --cif_dir /path/to/cif --inference_config config_name --num_gpus 4
"""

import os
import sys
import argparse
import subprocess
import logging
import time
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

def run_with_conda_env(env_name, command_list, cwd=None):
    """Run a command with conda environment activation using shell script wrappers."""
    # Get the wrapper script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if env_name == 'proteina':
        wrapper_script = os.path.join(script_dir, 'run_with_proteina_env.sh')
    elif env_name == 'colabdesign':
        wrapper_script = os.path.join(script_dir, 'run_with_colabdesign_env.sh')
    elif env_name == 'proteinebm':
        wrapper_script = os.path.join(script_dir, 'run_with_proteinebm_env.sh')
    else:
        logger.error(f"❌ Unknown environment: {env_name}")
        return False
    
    # Check if wrapper script exists
    if not os.path.exists(wrapper_script):
        logger.error(f"❌ Wrapper script not found: {wrapper_script}")
        return False
    
    # Build command with wrapper script
    cmd = [wrapper_script] + command_list
    
    logger.info(f"🚀 Running in {env_name} environment: {' '.join(command_list)}")
    logger.debug(f"Full command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=False)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"❌ Failed to run command: {e}")
        return False

def run_proteina_inference(csv_file, csv_column, cif_dir, inference_config, num_gpus, usalign_path=None, force_compile: bool = False):
    """Run the Proteina inference pipeline."""
    logger.info("🧬 Starting Proteina inference pipeline...")
    
    cmd = [
        'python', 'parallel_proteina_inference.py',
        '--csv_file', csv_file,
        '--csv_column', csv_column,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--skip_existing'  # Always skip existing to avoid re-processing
    ]
    if force_compile:
        cmd.append('--force_compile')
    
    if usalign_path:
        cmd.extend(['--usalign_path', usalign_path])
    
    # Run in proteina environment using shell script wrapper
    return run_with_conda_env('proteina', cmd)

def run_af2rank_scoring(csv_file, csv_column, cif_dir, inference_config, num_gpus, recycles=3, regenerate_plots=False):
    """Run the AF2Rank scoring pipeline."""
    logger.info("⚡ Starting AF2Rank scoring pipeline...")
    
    cmd = [
        'python', 'parallel_af2rank_scoring.py',
        '--csv_file', csv_file,
        '--csv_column', csv_column,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--recycles', str(recycles),
        '--filter_existing'  # Always filter to only score proteins needing AF2Rank
    ]
    
    if regenerate_plots:
        cmd.append('--regenerate_plots')
    
    # Run in colabdesign environment using shell script wrapper
    return run_with_conda_env('colabdesign', cmd)

def run_proteinebm_scoring(csv_file, csv_column, cif_dir, inference_config, num_gpus, proteinebm_config, proteinebm_checkpoint, proteinebm_template_self_condition=True):
    """Run the ProteinEBM scoring pipeline."""
    logger.info("💸 Starting ProteinEBM scoring pipeline...")
    
    cmd = [
        'python', 'parallel_proteinebm_scoring.py',
        '--csv_file', csv_file,
        '--csv_column', csv_column,
        '--cif_dir', cif_dir,
        '--inference_config', inference_config,
        '--num_gpus', str(num_gpus),
        '--filter_existing',  # Always filter to only score proteins needing ProteinEBM
        '--proteinebm_config', proteinebm_config,
        '--proteinebm_checkpoint', proteinebm_checkpoint,
    ]
    
    if not proteinebm_template_self_condition:
        cmd.append('--no-proteinebm_template_self_condition')
    
    # Run in proteinebm environment using shell script wrapper
    return run_with_conda_env('proteinebm', cmd)

def run_cross_protein_plots(
    inference_dir: str,
    output_dir: str,
    scorer: str,
    dataset_file: str,
    id_column: str,
    tms_column: str,
    af2rank_top_k: int,
    proteinebm_plot_mode: str = "tm",
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
        "--id_column",
        id_column,
        "--tms_column",
        tms_column,
    ]

    if scorer == "proteinebm":
        cmd.extend(["--proteinebm_plot_mode", proteinebm_plot_mode])

    if scorer == "af2rank_on_proteinebm_topk":
        cmd.extend(["--af2rank_top_k", str(int(af2rank_top_k))])

    return run_with_conda_env("proteina", cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def run_af2rank_on_proteinebm_topk(
    inference_dir: str,
    dataset_file: str,
    id_column: str,
    tms_column: str,
    top_k: int,
    recycles: int,
    num_gpus: int,
    filter_existing: bool = True,
) -> bool:
    """Run AF2Rank scoring on the ProteinEBM top-k templates per protein."""
    logger.info("🧪 Starting AF2Rank-on-ProteinEBM-topk step...")

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
    ]

    if dataset_file:
        cmd.extend(
            [
                "--dataset_file",
                dataset_file,
                "--id_column",
                id_column,
                "--tms_column",
                tms_column,
            ]
        )

    if filter_existing:
        cmd.append("--filter_existing")
    else:
        cmd.append("--no-filter_existing")

    # This script launches colabdesign subprocesses internally via wrapper script,
    # but the driver itself can run under proteina env.
    return run_with_conda_env("proteina", cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Complete AF2Rank Evaluation Pipeline')
    # Prefer consistent naming: dataset_file + id_column.
    # Keep --csv_file/--csv_column as hidden aliases for backward compatibility.
    parser.add_argument('--dataset_file', required=True, help='Path to dataset CSV file with protein ids and reference TM scores')
    parser.add_argument('--id_column', required=True, help='Column name in dataset CSV to use as protein ID (e.g. pdb)')
    parser.add_argument('--csv_file', dest='dataset_file', help=argparse.SUPPRESS)
    parser.add_argument('--csv_column', dest='id_column', help=argparse.SUPPRESS)
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
        '--af2rank_topk_filter_existing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Skip proteins whose af2rank_on_proteinebm_topk outputs already exist.',
    )
    parser.add_argument('--usalign_path', help='Path to USalign executable')
    parser.add_argument('--skip_inference', action='store_true', 
                       help='Skip Proteina inference (only run scoring stage)')
    parser.add_argument('--skip_scoring', action='store_true',
                       help='Skip scoring stage (AF2Rank or ProteinEBM depending on --scorer)')
    parser.add_argument('--regenerate_plots', action='store_true',
                       help='Regenerate AF2Rank plots even if scoring already completed')
    parser.add_argument(
        '--proteina_force_compile',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Pass --force_compile to Proteina inference (torch.compile even in eval/no_grad).',
    )
    parser.add_argument('--tms_column', required=True,
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
    
    args = parser.parse_args()
    
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    
    logger.info(f"🚀 Starting complete AF2Rank evaluation pipeline")
    logger.info(f"📊 Dataset file: {args.dataset_file}")
    logger.info(f"📂 CIF directory: {args.cif_dir}")
    logger.info(f"⚙️  Inference config: {args.inference_config}")
    logger.info(f"🔥 GPUs: {args.num_gpus}")
    logger.info(f"🧮 Scorer: {args.scorer}")
    logger.info(f"🔄 AF2Rank recycles: {args.recycles}")
    logger.info(f"🐍 Using shell script wrappers for conda environments")
    logger.info(f"🔑 ID column: {args.id_column}")
    logger.info(f"🔑 TM score column: {args.tms_column}")
    
    success = True
    
    # Step 1: Proteina Inference
    if not args.skip_inference:
        logger.info("\n" + "="*60)
        logger.info("STEP 1: PROTEINA INFERENCE")
        logger.info("="*60)
        
        inference_success = run_proteina_inference(
            args.dataset_file,
            args.id_column,
            args.cif_dir, 
            args.inference_config, 
            args.num_gpus,
            args.usalign_path,
            force_compile=args.proteina_force_compile,
        )
        
        if inference_success:
            logger.info("✅ Proteina inference completed successfully")
        else:
            logger.error("❌ Proteina inference failed")
            success = False
    else:
        logger.info("⏭️  Skipping Proteina inference")
    
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
                args.id_column,
                args.cif_dir,
                args.inference_config,
                args.num_gpus,
                args.recycles,
                args.regenerate_plots
            )
        else:
            scoring_success = run_proteinebm_scoring(
                csv_file=args.dataset_file,
                csv_column=args.id_column,
                   cif_dir=args.cif_dir,
                inference_config=args.inference_config,
                num_gpus=args.num_gpus,
                proteinebm_config=args.proteinebm_config,
                proteinebm_checkpoint=args.proteinebm_checkpoint,
                proteinebm_template_self_condition=args.proteinebm_template_self_condition
            )
        
        if scoring_success:
            logger.info(f"✅ Scoring completed successfully ({args.scorer})")
        else:
            logger.error(f"❌ Scoring failed ({args.scorer})")
            success = False
    elif args.skip_scoring:
        logger.info(f"⏭️  Skipping scoring stage ({args.scorer})")

    # Step 3: Optional AF2Rank scoring on ProteinEBM top-k templates
    if success and (not args.skip_scoring) and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: AF2RANK ON PROTEINEBM TOP-K TEMPLATES")
        logger.info("="*60)

        inference_dir = os.path.join(project_root, "inference", args.inference_config)
        topk_success = run_af2rank_on_proteinebm_topk(
            inference_dir=inference_dir,
            dataset_file=args.dataset_file,
            id_column=args.id_column,
            tms_column=args.tms_column,
            top_k=int(args.af2rank_top_k),
            recycles=int(args.recycles),
            num_gpus=int(args.num_gpus),
            filter_existing=bool(args.af2rank_topk_filter_existing),
        )

        if topk_success:
            logger.info("✅ AF2Rank-on-ProteinEBM-topk completed successfully")
        else:
            logger.error("❌ AF2Rank-on-ProteinEBM-topk failed")
            success = False

    # Step 4: Cross-protein plots (always, if previous steps succeeded)
    if success:
        logger.info("\n" + "="*60)
        logger.info("STEP 4: CROSS-PROTEIN PLOTS")
        logger.info("="*60)

        inference_dir = os.path.join(project_root, "inference", args.inference_config)
        if args.cross_protein_output_dir:
            cross_out_dir = os.path.abspath(os.path.expanduser(args.cross_protein_output_dir))
        else:
            cross_out_dir = os.path.join(inference_dir, f"{Path(args.dataset_file).stem}_cross_protein_analysis")

        plot_success = run_cross_protein_plots(
            inference_dir=inference_dir,
            output_dir=cross_out_dir,
            scorer=args.scorer,
            dataset_file=args.dataset_file,
            id_column=args.id_column,
            tms_column=args.tms_column,
            af2rank_top_k=int(args.af2rank_top_k),
            proteinebm_plot_mode=str(args.proteinebm_cross_protein_plot_mode),
        )
        if not plot_success:
            logger.error("❌ Cross-protein plotting failed")
            success = False

        if success and args.scorer == "proteinebm" and int(args.af2rank_top_k) > 0:
            plot_success_2 = run_cross_protein_plots(
                inference_dir=inference_dir,
                output_dir=cross_out_dir,
                scorer="af2rank_on_proteinebm_topk",
                dataset_file=args.dataset_file,
                id_column=args.id_column,
                tms_column=args.tms_column,
                af2rank_top_k=int(args.af2rank_top_k),
                proteinebm_plot_mode=str(args.proteinebm_cross_protein_plot_mode),
            )
            if not plot_success_2:
                # This commonly occurs if AF2Rank-on-ProteinEBM-top-k hasn't been run yet
                # for this inference_dir/dataset combination. Treat as a warning so the
                # pipeline can still complete after producing the primary plots.
                logger.warning("⚠️  Cross-protein plotting skipped/failed (af2rank_on_proteinebm_topk)")
    
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
