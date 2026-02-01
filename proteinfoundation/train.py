# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys
import time

# # Set CUDA environment variables
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# # Disable CUDA graph capture
# os.environ["CUDA_ENABLE_GRAPH_CAPTURE"] = "0"

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import loralib as lora
import torch
torch._dynamo.config.optimize_ddp = False
import wandb
from dotenv import load_dotenv
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.utils.precompute_confind_maps import run_precompute
from proteinfoundation.utils.ema_utils.ema_callback import EMA, EmaModelCheckpoint
from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.metric_utils import (
    transform_global_percentage_to_mask_dropout,
)
from proteinfoundation.utils.seed_callback import SeedCallback
from proteinfoundation.utils.training_analysis_utils import (
    GradAndWeightAnalysisCallback,
    LogEpochTimeCallback,
    LogSetpTimeCallback,
    SkipNanGradCallback,
)
def _maybe_init_cuequivariance():
    try:
        import cuequivariance_torch as cuet  # noqa: F401
        from cueuivariance_ops_torch import init_triton_cache
        init_triton_cache()
        return True
    except Exception as exc:
        log_info(f"Skipping cuequivariance init: {exc}")
        return False

@rank_zero_only
def wandb_login():
    wandb.login()


# Things that should only be done by a single process
@rank_zero_only
def log_info(msg):
    logger.info(msg)


@rank_zero_only
def store_configs(path_configs):
    for cfg, path in path_configs:
        with open(path, "w") as f:
            cfg_aux = OmegaConf.to_container(cfg, resolve=True)
            json.dump(cfg_aux, f, indent=4, sort_keys=True)


@rank_zero_only
def log_configs(path_configs, wandb_logger):
    if wandb_logger is None:
        return
    artifact = wandb.Artifact(wandb_safe_name(f"config_files_{run_name}"), type="config")
    for _, path in path_configs:
        artifact.add_file(path)
    wandb_logger.experiment.log_artifact(artifact)


@rank_zero_only
def create_dir(checkpoint_path_store, parents=True, exist_ok=True):
    Path(checkpoint_path_store).mkdir(parents=parents, exist_ok=exist_ok)


def wandb_safe_name(name: str, max_len: int = 128) -> str:
    """
    W&B API enforces a 128-char limit for certain name fields (run/artifact names).
    Keep names readable but guarantee length via truncation + short hash suffix.
    """
    if name is None:
        return "none"
    name = str(name)
    if len(name) <= max_len:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    # Reserve space for "-" + digest
    keep = max_len - (1 + len(digest))
    if keep < 1:
        return digest[:max_len]
    return f"{name[:keep]}-{digest}"


def _slurm_task_info():
    array_id = os.getenv("SLURM_ARRAY_TASK_ID")
    array_count = os.getenv("SLURM_ARRAY_TASK_COUNT")
    if array_id is not None and array_count is not None:
        return int(array_id), int(array_count)
    proc_id = os.getenv("SLURM_PROCID")
    n_tasks = os.getenv("SLURM_NTASKS")
    if proc_id is not None and n_tasks is not None:
        return int(proc_id), int(n_tasks)
    return 0, 1


def _dataset_csv_path(datamodule) -> Optional[str]:
    if getattr(datamodule, "dataselector", None) is not None:
        file_identifier = datamodule._get_file_identifier(datamodule.dataselector)
    else:
        file_identifier = datamodule.data_dir.name
    return os.path.join(datamodule.data_dir, f"{file_identifier}.csv")


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="training_ca",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--nolog",
        action="store_true",
        help="Avoids checkpoints and wandb logging, mostly for debugging.",
    )
    parser.add_argument(
        "--single", action="store_true", help="Sets single node and single GPU, ignoring config file."
    )
    parser.add_argument(
        "--show_prog_bar",
        action="store_true",
        help="Shows progress bar as training progresses.",
    )
    parser.add_argument(
        "--resume_option",
        type=str,
        default="allow",
        choices=["allow", "never", "must", None],
        help=(
            "Resume option: allow (resume if exists, create if not), never (always create new), "
            "must (must resume existing), None (same as allow when id is set). Default is allow."
        ),
    )
    parser.add_argument(
        "--prepare_data_only",
        action="store_true",
        help="Run dataset preparation (and confind precompute if applicable) then exit.",
    )
    args, overrides = parser.parse_known_args()

    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout
    log_info(f"Avoid wandb and checkpointing: {args.nolog}")
    callbacks = [SeedCallback()]  # Different devices will be assigend different seeds

    # Load experiment config
    config_path = "../configs/experiment_config"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=args.config_name, overrides=overrides)
        if args.single:
            # Rewrite number of GPUs and nodes for local runs or if single flag is used
            cfg_exp.hardware.ngpus_per_node_ = 1
            cfg_exp.hardware.nnodes_ = 1
            cfg_exp.run_name_ = cfg_exp.run_name_ + "_single"
        log_info(f"Exp config {cfg_exp}")

    # Set training precision
    precision = "32"
    if not cfg_exp.force_precision_f32:
        log_info("Using mixed precision")
        torch.set_float32_matmul_precision("medium")
        # precision = "16"
        precision = "bf16-mixed"

    # Set training fold labels dropout rate based on global percentage
    if cfg_exp.training.get("fold_label_sample_ratio") is not None:
        log_info("Setting fold label dropout rate based on fold_label_sample_ratio")
        (
            cfg_exp.training.mask_T_prob,
            cfg_exp.training.mask_A_prob,
            cfg_exp.training.mask_C_prob,
        ) = transform_global_percentage_to_mask_dropout(
            cfg_exp.training.fold_label_sample_ratio
        )
        log_info(
            "Set mask_T_prob: %.3f, mask_A_prob: %.3f, mask_C_prob: %.3f"
            % (
                cfg_exp.training.mask_T_prob,
                cfg_exp.training.mask_A_prob,
                cfg_exp.training.mask_C_prob,
            )
        )



    # Set run name and root directory for the run, used to store things
    run_name = cfg_exp.run_name_
    log_info(f"Job name: {run_name}")
    root_run = os.path.join(
        ".", "store", run_name
    )  # Everything stored in ./store/<run_id>
    log_info(f"Root run: {root_run}")

    # Set checkpoint directory
    checkpoint_path_store = os.path.join(
        root_run, "checkpoints"
    )  # Checkpoints in ./store/run_id/checkpoints/<ckpt-file>
    log_info(f"Checkpoints directory: {checkpoint_path_store}")

    # Check if last checkpoint exists (this is useful if interrupted, it starts from last checkpoint)
    last_ckpt_name = fetch_last_ckpt(checkpoint_path_store)
    last_ckpt_path = (
        os.path.join(checkpoint_path_store, last_ckpt_name)
        if last_ckpt_name is not None
        else None
    )
    log_info(f"Last checkpoint: {last_ckpt_path}")

    # Extract number of cpus from config file
    num_cpus = cfg_exp.hardware.ncpus_per_task_train_
    log_info(
        f"Number of CPUs per task used (will be used for number dataloader number of workers): {num_cpus}"
    )

    # If no checkpoint set seed for correct initialization
    if last_ckpt_path is None:
        log_info(f"Seeding everything to seed {cfg_exp.seed}")
        L.seed_everything(cfg_exp.seed)

    # Load data config
    # Set CIRPIN embedding mode environment variable for dataset config
    if cfg_exp.get("cirpin", {}).get("embedding_mode"):
        os.environ["CIRPIN_EMBEDDING_MODE"] = cfg_exp.cirpin.embedding_mode
        log_info(f"Set CIRPIN_EMBEDDING_MODE={cfg_exp.cirpin.embedding_mode}")
    
    dataset_config_subdir = cfg_exp.get("dataset_config_subdir", None)
    if dataset_config_subdir is not None:
        # if args.dataset_config_subdir:
        config_path = f"../configs/datasets_config/{dataset_config_subdir}"
    else:
        config_path = "../configs/datasets_config/"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp["dataset"])
        cfg_data.datamodule.num_workers = num_cpus  # Overwrite number of cpus
        if cfg_data.get("exclude_id_pkl_path") is not None:
            with open(cfg_data.exclude_id_pkl_path, "rb") as fin:
                exclude_ids = pickle.load(fin)
            if cfg_data.datamodule.dataselector.exclude_ids is not None:
                cfg_data.datamodule.dataselector.exclude_ids += exclude_ids
            else:
                cfg_data.datamodule.dataselector.exclude_ids = exclude_ids
        # overfit_pdb_chains is set in the dataset config (e.g., pdb_train.yaml)
        # and passed to datamodule via hydra instantiation
        if cfg_data.datamodule.get("overfit_pdb_chains") is not None:
            log_info(f"Overfit mode enabled: filtering to {cfg_data.datamodule.overfit_pdb_chains}")
        log_info(f"Data config {cfg_data}")

    # create datamodule containing default train and val dataloader
    datamodule = hydra.utils.instantiate(cfg_data.datamodule)
    
    # Ensure data preparation only happens on rank 0 to avoid duplicate processing
    if hasattr(datamodule, 'prepare_data'):
        datamodule.prepare_data = rank_zero_only(datamodule.prepare_data)

    # Run dataset preparation immediately to ensure CSV exists
    if hasattr(datamodule, "prepare_data"):
        slurm_task_id, slurm_task_count = _slurm_task_info()
        dataset_csv = _dataset_csv_path(datamodule)
        if slurm_task_count > 1 and slurm_task_id != 0:
            wait_sec = int(os.getenv("DATA_PREP_WAIT_SEC", "21600"))
            log_info(
                f"SLURM task {slurm_task_id}/{slurm_task_count}: waiting for dataset CSV {dataset_csv}"
            )
            start_wait = time.time()
            while not os.path.exists(dataset_csv):
                if time.time() - start_wait > wait_sec:
                    raise RuntimeError(
                        f"Timed out waiting for dataset CSV {dataset_csv}"
                    )
                time.sleep(30)
            log_info(f"Dataset CSV found: {dataset_csv}")
        else:
            log_info("Preparing dataset...")
            datamodule.prepare_data()

    # If confind contact maps are requested, precompute and cache them
    transforms_cfg = cfg_data.datamodule.get("transforms")
    transforms_list = (
        OmegaConf.to_container(transforms_cfg, resolve=True)
        if transforms_cfg is not None
        else []
    )
    confind_transform = None
    for t in transforms_list:
        if isinstance(t, dict) and t.get("_target_") == "proteinfoundation.datasets.transforms.ContactMapTransform":
            if t.get("contact_method") == "confind":
                confind_transform = t
                break
    if confind_transform is not None:
        @rank_zero_only
        def _run_confind_precompute():
            log_info("Confind contact maps requested; precomputing raw maps...")
            processed_dir = os.path.join(cfg_data.datamodule.data_dir, "processed")
            rotlib = confind_transform.get("confind_rotamer_lib")
            dataselector_cfg = cfg_data.datamodule.get("dataselector")
            dataselector_dict = (
                OmegaConf.to_container(dataselector_cfg, resolve=True)
                if dataselector_cfg is not None
                else None
            )
            use_slurm = bool(
                os.getenv("SLURM_ARRAY_TASK_ID")
                or os.getenv("SLURM_PROCID")
                or os.getenv("SLURM_NTASKS")
            )
            visible_cores = int(
                os.getenv("SLURM_CPUS_PER_TASK")
                or os.cpu_count()
                or cfg_data.datamodule.num_workers
            )
            workers = max(1, visible_cores // 2)
            log_info(f"Confind precompute workers set to {workers}/{visible_cores} cores")
            omp_threads = int(os.getenv("OMP_NUM_THREADS") or 1)
            if not rotlib:
                rotlib = os.getenv("MSL_ROTLIB")
            if not rotlib:
                candidate = os.path.join(cfg_data.datamodule.data_dir, "rotlibs")
                if os.path.isdir(candidate):
                    rotlib = candidate
            run_precompute(
                processed_dir=Path(processed_dir),
                rotlib=rotlib,
                dataselector=dataselector_dict,
                confind_bin=os.getenv("CONFIND_BIN", "confind"),
                omp_threads=omp_threads,
                workers=workers,
                use_slurm=use_slurm,
                output_csv=str(Path(processed_dir).parent / "confind_precompute.csv"),
                format_type=cfg_data.datamodule.get("format", "cif"),
                store_het=cfg_data.datamodule.get("store_het", False),
                store_bfactor=cfg_data.datamodule.get("store_bfactor", True),
            )

        _run_confind_precompute()

    if args.prepare_data_only:
        log_info("prepare_data_only set; exiting after dataset preparation.")
        sys.exit(0)

    _maybe_init_cuequivariance()

    # Set logger
    wandb_logger = None
    if cfg_exp.log.log_wandb and not args.nolog:
        wandb_resume = args.resume_option
        wandb_logger = WandbLogger(
            entity="kryst3154-massachusetts-institute-of-technology", 
            project=cfg_exp.log.wandb_project, 
            id=wandb_safe_name(run_name),
            name=wandb_safe_name(run_name),
            resume=wandb_resume,
        )
        callbacks.append(LogEpochTimeCallback())
        callbacks.append(LogSetpTimeCallback())

    log_info(f"Using EMA with decay {cfg_exp.ema.decay}")
    callbacks.append(EMA(**cfg_exp.ema))

    # Set checkpointing
    if cfg_exp.log.checkpoint and not args.nolog:
        args_ckpt_last = {
            "dirpath": checkpoint_path_store,
            "save_weights_only": False,
            "filename": "ignore",
            "every_n_train_steps": cfg_exp.log.last_ckpt_every_n_steps,
            "save_last": True,
            "enable_version_counter": False,
            "save_top_k": 0,
        }
        args_ckpt = {
            "dirpath": checkpoint_path_store,
            "save_last": False,
            "save_weights_only": False,
            "filename": "chk_{epoch:08d}_{step:012d}",
            "every_n_train_steps": cfg_exp.log.checkpoint_every_n_steps,
            "monitor": "validation_loss/loss",  # Works for both coordinate and contact map modes
            "save_top_k": 5,
            "mode": "min",
        }
        checkpoint_callback = EmaModelCheckpoint(**args_ckpt)
        checkpoint_callback_last = EmaModelCheckpoint(**args_ckpt_last)

        create_dir(checkpoint_path_store, parents=True, exist_ok=True)
        callbacks.append(checkpoint_callback)
        callbacks.append(checkpoint_callback_last)

        # Save and log config files
        path_configs = [
            (
                cfg_data,
                os.path.join(checkpoint_path_store, f"data_config_{run_name}.json"),
            ),
            (
                cfg_exp,
                os.path.join(checkpoint_path_store, f"exp_config_{run_name}.json"),
            ),
        ]
        store_configs(path_configs)
        log_configs(path_configs, wandb_logger)

    # Gradient and weight stats thoughout training, possibly skip updates with nan in grad
    if cfg_exp.opt.grad_and_weight_analysis:
        callbacks.append(GradAndWeightAnalysisCallback())
    if cfg_exp.opt.skip_nan_grad:
        callbacks.append(SkipNanGradCallback())

    # Define model
    from proteinfoundation.proteinflow.proteina import Proteina
    model = Proteina(cfg_exp, store_dir=root_run)

    # If LoRA is tunred on, replace Linear with LoRA layers
    if cfg_exp.get("lora") and cfg_exp.lora.get("use") and cfg_exp.lora.get("r"):
        replace_lora_layers(
            model, cfg_exp.lora.r, cfg_exp.lora.lora_alpha, cfg_exp.lora.lora_dropout
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg_exp.lora.train_bias)

    # If this is the first run for fine-tuning, load pre-trained checkpoint and don't load optimizer states
    pretrain_ckpt_path = cfg_exp.get("pretrain_ckpt_path", None)
    if last_ckpt_path is None and pretrain_ckpt_path is not None:
        log_info(f"Loading from pre-trained checkpoint path {os.path.expanduser(pretrain_ckpt_path)}")
        ckpt = torch.load(os.path.expanduser(pretrain_ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)

    # Train
    plugins = []
    show_prog_bar = args.show_prog_bar
    trainer = L.Trainer(
        max_epochs=cfg_exp.opt.max_epochs,
        accelerator=cfg_exp.hardware.accelerator,
        devices=cfg_exp.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg_exp.hardware.nnodes_,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=cfg_exp.log.get("log_every_n_steps", 100),
        default_root_dir=root_run,
        check_val_every_n_epoch=None,  # Leave like this
        val_check_interval=cfg_exp.opt.val_check_interval,
        strategy=DDPStrategy(
            process_group_backend=cfg_exp.opt.get("dist_backend", "nccl"),
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # Memory optimization
            static_graph=False
        ) if cfg_exp.opt.dist_strategy == "ddp" else cfg_exp.opt.dist_strategy,
        enable_progress_bar=show_prog_bar,
        plugins=plugins,
        accumulate_grad_batches=cfg_exp.opt.accumulate_grad_batches,
        num_sanity_val_steps=1,
        precision=precision,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model, datamodule, ckpt_path=last_ckpt_path
    )  # If None then it starts from scratch
