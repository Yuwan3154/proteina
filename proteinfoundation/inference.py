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
from typing import List

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
import random
import shutil
import loralib as lora

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
torch.set_float32_matmul_precision("medium")
# Match train.py: raise the Dynamo recompile limit so dynamic-shape eval
# (many sampling lengths) doesn't evict cached graphs (default 8).
torch._dynamo.config.recompile_limit = 32
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from proteinfoundation.utils.lora_utils import replace_lora_layers


from proteinfoundation.metrics.designability import scRMSD
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level, write_prot_to_pdb
from proteinfoundation.utils.openfold_inference import OpenFoldTemplateInference


# Length dataloader for validation and inference
class GenDataset(Dataset):
    """
    Dataset that indicates length of the proteins to generate,
    discretization step size, and number of samples per length,
    empirical (len, cath_code) joint distribution.
    """

    bucket_min_len = 50
    bucket_max_len = 274
    bucket_step_size = 25

    def __init__(self, nres=[110], dt=0.005, nsamples=10, len_cath_codes=None):
        # nres is a list of integers
        # len_cath_codes is a list of [len, List[cath_code]] pairs, representing (len, cath_code) joint distribution
        super(GenDataset, self).__init__()
        self.nres = [int(n) for n in nres]
        self.dt = dt
        if isinstance(nsamples, List):
            assert len(nsamples) == len(nres)
            self.nsamples = nsamples
        elif isinstance(nsamples, int):
            self.nsamples = [nsamples] * len(nres)
        else:
            raise ValueError(f"Unknown type of nsamples {type(nsamples)}")
        self.cath_codes_given_len_bucket = self.bucketize(len_cath_codes)

    def bucketize(self, len_cath_codes):
        """Build length buckets for cath_codes. Record the cath_code distribution given length bucket"""
        if len_cath_codes is None:
            return None
        bucket = list(
            range(self.bucket_min_len, self.bucket_max_len, self.bucket_step_size)
        )
        cath_codes_given_len_bucket = [[] for _ in range(len(bucket))]
        for _len, code in len_cath_codes:
            bucket_idx = (_len - self.bucket_min_len) // self.bucket_step_size
            cath_codes_given_len_bucket[bucket_idx].append(code)
        return cath_codes_given_len_bucket

    def __len__(self):
        return len(self.nres)

    def __getitem__(self, index):
        result = {
            "nres": self.nres[index],
            "dt": self.dt,
            "nsamples": self.nsamples[index],
        }
        if self.cath_codes_given_len_bucket is not None:
            if self.nres[index] <= self.bucket_max_len:
                bucket_idx = (
                    self.nres[index] - self.bucket_min_len
                ) // self.bucket_step_size
            else:
                bucket_idx = -1
            result["cath_code"] = random.choices(
                self.cath_codes_given_len_bucket[bucket_idx], k=self.nsamples[index]
            )
        return result


class GenDatasetWithSeqCath(Dataset):
    def __init__(self, pt, cath_codes, dt=0.005, nsamples_per_len=10, max_nsamples=10):
        super(GenDatasetWithSeqCath, self).__init__()
        self.pt = pt
        self.dt = dt
        self.nsamples_per_len = nsamples_per_len
        self.max_nsamples = max_nsamples
        self.cath_codes = cath_codes

        self.num_batches_per_cath = (nsamples_per_len + max_nsamples - 1) // max_nsamples # Equivalent to math.ceil

    def __len__(self):
        return len(self.cath_codes) * self.num_batches_per_cath
        
    def __getitem__(self, index):
        cath_idx = index // self.num_batches_per_cath
        batch_idx_for_cath = index % self.num_batches_per_cath
        
        current_cath_code = self.cath_codes[cath_idx]

        start_sample_idx = batch_idx_for_cath * self.max_nsamples
        end_sample_idx = start_sample_idx + self.max_nsamples

        if end_sample_idx > self.nsamples_per_len:
            num_samples_for_batch = self.nsamples_per_len - start_sample_idx
        else:
            num_samples_for_batch = self.max_nsamples
            
        cath_code_batch = [[current_cath_code] for _ in range(num_samples_for_batch)]
        residue_type = self.pt.residue_type
        result = {
            "nres": residue_type.shape[-1],
            "dt": self.dt,
            "nsamples": num_samples_for_batch,
            "cath_code": cath_code_batch,
            "residue_type": residue_type,
        }
        for _k in ("residue_pdb_idx", "chain_breaks_per_residue"):
            _v = getattr(self.pt, _k, None)
            if _v is not None:
                result[_k] = _v
        return result


def _tokens_to_palette_image(tokens: np.ndarray) -> Image.Image:
    img = Image.fromarray(tokens.astype(np.uint8), mode="P")
    palette = [0, 0, 0, 255, 0, 0, 160, 160, 160] + [0] * (256 * 3 - 9)
    img.putpalette(palette)
    return img


def split_nlens(nlens_dict, max_nsamples=16, n_replica=1):
    """
    Split nlens into data points (len, nsample) as val dataset and guarantee that
        1. len(val_dataset) should be a multiple of n_replica, to ensure that we don't introduce additional samples for multi-gpu validation
        2. nsample should be the same for all data points if n_replica > 1 (multi-gpu)

    Args:
        nlens_dict: Dict of nlens distribution.
            nlens_dict["length_ranges"] is a set of bin boundaries.
            nlens_dict["length_distribution"] is the numbers of samples in each bin
        max_nsamples: Maximum nsample in each data point
        n_replica: Number of GPUs

    Returns:
        lens_sample (List[int]): List of len for val data points
        nsamples (List[int]): List of nsample for val data points
    """
    lengths_range = nlens_dict["length_ranges"].tolist()
    length_distribution = nlens_dict["length_distribution"].tolist()
    lens_sample, nsamples = [], []
    for length, cnt in zip(lengths_range, length_distribution):
        for i in range(0, cnt, max_nsamples):
            lens_sample.append(length)
            if i + max_nsamples <= cnt:
                nsamples.append(max_nsamples)
            else:
                nsamples.append(cnt - i)

    max_nsamples = max(nsamples)
    for i in range(len(nsamples)):
        nsamples[i] += max_nsamples - nsamples[i]

    while len(lens_sample) % n_replica != 0:
        lens_sample.append(lens_sample[-1])
        nsamples.append(max_nsamples)

    return lens_sample, nsamples


def parse_nlens_cfg(cfg):
    """Parse lengths distribution. Either loading an empirical one or build with arguments in yaml file"""
    if cfg.get("nres_lens_distribution_path") is not None:
        # Sample according to length distribution
        nlens_dict = torch.load(cfg.nres_lens_distribution_path, weights_only=False)
    else:
        # Sample with pre-specified lengths
        if cfg.nres_lens:
            _lens_sample = cfg.nres_lens
        else:
            _lens_sample = [
                int(v)
                for v in np.arange(cfg.min_len, cfg.max_len + 1, cfg.step_len).tolist()
            ]
        nlens_dict = {
            "length_ranges": torch.as_tensor(_lens_sample),
            "length_distribution": torch.as_tensor(
                [cfg.nsamples_per_len] * len(_lens_sample)
            ),
        }
    return nlens_dict


def _normalize_cath_code(code):
    """Normalize a CATH code string to 4 dot-separated levels (pad H with 'x').

    Accepts 'C.A.T' or 'C.A.T.H' (or 'x.x.x.x' as a null sentinel).
    Raises SystemExit on malformed input.
    """
    parts = code.strip().split(".")
    if len(parts) == 3:
        parts.append("x")
    elif len(parts) != 4:
        raise SystemExit(f"--cath_code must have 3 or 4 dot-separated parts; got {code!r}")
    return ".".join(parts)


def parse_len_cath_code(cfg):
    """Load (len, cath_codes) joint distribution. Apply mask according to the guidance cath code level"""
    if cfg.get("len_cath_code_path") is not None:
        logger.info(
            f"Loading empirical (length, cath_code) distribution from {cfg.len_cath_code_path}"
        )
        _len_cath_codes = torch.load(cfg.len_cath_code_path, weights_only=False)
        level = cfg.get("cath_code_level")
        len_cath_codes = []
        for i in range(len(_len_cath_codes)):
            _len, code = _len_cath_codes[i]
            code = mask_cath_code_by_level(code, level="H")
            if level == "A" or level == "C":
                code = mask_cath_code_by_level(code, level="T")
                if level == "C":
                    code = mask_cath_code_by_level(code, level="A")
            len_cath_codes.append((_len, code))
    else:
        logger.info(
            "No empirical (length, cath_code) distribution provided. Use unconditional training."
        )
        len_cath_codes = None
    return len_cath_codes


# ---------------------------------------------------------------------------
# Reusable in-process inference primitives.
#
# These three functions form the public API the parallel runner (and main())
# share so a worker can load the model ONCE and reuse it across many proteins:
#
#   cfg, config_name, cath_override = compose_inference_cfg(args)
#   model, nn_ag, trainer = load_model_for_worker(cfg, force_compile=..., dynamic_shapes=...)
#   result = run_one_protein_in_process(model, nn_ag, trainer, cfg,
#                                       pt_name=..., config_name=..., ...)
#
# main() is now a thin shell that runs them once for a single protein, matching
# the legacy single-protein CLI invocation. The parallel runner can call them
# inside a long-lived worker process and skip the model-load between proteins.
# ---------------------------------------------------------------------------


def compose_inference_cfg(args):
    """Hydra compose + unified-config merge + apply CLI overrides.

    Returns:
        (cfg, config_name, cath_codes_override)

    `cath_codes_override` is None unless --conditioning_mode or --cath_code was set.
    """
    if args.config_subdir is None:
        config_path = "../configs/experiment_config"
    else:
        config_path = f"../configs/experiment_config/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)

        # Resolve unified config (training and inference configs in the same file).
        # DESIGN INVARIANT: there is exactly ONE `run_name_` per config, at the top
        # level. The `inference:` block must NOT define its own `run_name_` — see
        # the comment in the unified YAML.
        if "inference" in cfg and cfg.inference is not None:
            if "run_name_" in cfg.inference:
                raise ValueError(
                    "Inference config block must NOT define its own 'run_name_'. "
                    "There is exactly one 'run_name_' per config, at the top level, "
                    "shared between training and inference."
                )
            from omegaconf import open_dict
            import copy
            cfg_copy = copy.deepcopy(cfg)
            with open_dict(cfg_copy):
                inf_block = cfg_copy.pop("inference")
                cfg = OmegaConf.merge(cfg_copy, inf_block)

        # CLI overrides — applied AFTER the unified-config merge so they actually win.
        if args.max_nsamples is not None:
            cfg = OmegaConf.merge(cfg, {"max_nsamples": args.max_nsamples})
            logger.info(f"Overriding max_nsamples to {args.max_nsamples}")
        if args.nsamples_per_protein is not None:
            cfg = OmegaConf.merge(cfg, {"nsamples_per_len": args.nsamples_per_protein})
            logger.info(f"Overriding nsamples_per_len to {args.nsamples_per_protein}")

        # Conditioning-mode override: drives cfg.fold_cond and the per-protein cath_codes.
        cath_codes_override = None
        if args.conditioning_mode == "seq":
            cfg = OmegaConf.merge(cfg, {"fold_cond": False})
            cath_codes_override = ["x.x.x.x"]
            logger.info("Conditioning mode 'seq': fold_cond=False, cath_codes=['x.x.x.x']")
        elif args.conditioning_mode == "seq_cath":
            cfg = OmegaConf.merge(cfg, {"fold_cond": True})
            if args.cath_code is None:
                raise SystemExit("--conditioning_mode seq_cath requires --cath_code")
            cath_codes_override = [_normalize_cath_code(args.cath_code)]
            logger.info(f"Conditioning mode 'seq_cath': fold_cond=True, cath_codes={cath_codes_override}")
        elif args.cath_code is not None:
            cath_codes_override = [_normalize_cath_code(args.cath_code)]
            logger.info(f"Using --cath_code override (no explicit mode): cath_codes={cath_codes_override}")

        logger.info(f"Inference config {cfg}")

    return cfg, config_name, cath_codes_override


def _resolve_checkpoint(cfg):
    """Resolve the checkpoint file path: explicit (cfg.ckpt_name) or auto via run_name_."""
    ckpt_name = cfg.get("ckpt_name", None)
    checkpoint_mode = cfg.get("checkpoint_mode", "best")

    if ckpt_name is not None and ckpt_name != "":
        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path is None:
            ckpt_path = os.path.join(os.getenv("DATA_PATH"), "weights")
        ckpt_file = os.path.expanduser(os.path.join(ckpt_path, ckpt_name))
        logger.info(f"Using explicit checkpoint {ckpt_file}")
        assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"
        return ckpt_file

    run_name_ = cfg.get("run_name_", None)
    if not run_name_:
        raise ValueError("No run_name_ found in config. Cannot automatically resolve checkpoint without run_name_.")
    ckpt_dir = os.path.join(".", "store", run_name_, "checkpoints")
    if checkpoint_mode == "last":
        from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
        last_ckpt_name = fetch_last_ckpt(ckpt_dir)
        if last_ckpt_name is None:
            raise FileNotFoundError(f"No last checkpoint found in {ckpt_dir}")
        ckpt_file = os.path.join(ckpt_dir, last_ckpt_name)
    else:  # "best"
        from proteinfoundation.utils.fetch_last_ckpt import fetch_best_ckpt
        best_ckpt_name = fetch_best_ckpt(ckpt_dir)
        if best_ckpt_name is None:
            logger.warning(f"No best checkpoint found in {ckpt_dir}, falling back to last checkpoint")
            from proteinfoundation.utils.fetch_last_ckpt import fetch_last_ckpt
            last_ckpt_name = fetch_last_ckpt(ckpt_dir)
            if last_ckpt_name is None:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            ckpt_file = os.path.join(ckpt_dir, last_ckpt_name)
        else:
            ckpt_file = os.path.join(ckpt_dir, best_ckpt_name)
    ckpt_file = os.path.expanduser(ckpt_file)
    logger.info(f"Using auto-resolved checkpoint {ckpt_file} (mode: {checkpoint_mode})")
    return ckpt_file


def load_model_for_worker(cfg, *, force_compile, dynamic_shapes=True, verbose=False, use_cueq=True):
    """Load checkpoint + configure inference + set up trainer. Called ONCE per worker.

    The (model, trainer) pair is REUSABLE across proteins. The first call to
    trainer.predict() triggers torch.compile (when force_compile=True);
    subsequent calls hit the inductor cache (or stay specialized) and run fast.
    With dynamic_shapes=True, a single compiled graph handles all protein
    lengths — no per-length recompile.

    Returns (model, nn_ag, trainer).
    """
    ckpt_file = _resolve_checkpoint(cfg)

    if not cfg.lora.use:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
    else:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
        logger.info("Re-create LoRA layers and reload the weights now")
        replace_lora_layers(model, cfg["lora"]["r"], cfg["lora"]["lora_alpha"], cfg["lora"]["lora_dropout"])
        lora.mark_only_lora_as_trainable(model, bias=cfg["lora"]["train_bias"])
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    nn_ag = None
    if cfg.get("autoguidance_ratio", 0.0) > 0 and cfg.get("guidance_weight", 1.0) != 1.0:
        assert cfg.autoguidance_ckpt_path is not None
        model_ag = Proteina.load_from_checkpoint(cfg.autoguidance_ckpt_path)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)
    pair_update_layers = getattr(model.nn, "pair_update_layers", None)
    if pair_update_layers is not None:
        for layer in pair_update_layers:
            if layer is not None and hasattr(layer, "use_cueq"):
                layer.use_cueq = bool(use_cueq)

    # Attributes the model's predict_step reads on every forward.
    model._force_compile = bool(force_compile)
    model._compile_dynamic = bool(dynamic_shapes)
    model._verbose = bool(verbose)
    # The nn module's forward() reads `_compile_dynamic` when compiling the
    # eval path; propagate so the inference-side torch.compile uses dynamic=True.
    if hasattr(model, "nn") and model.nn is not None:
        model.nn._compile_dynamic = bool(dynamic_shapes)

    trainer = L.Trainer(accelerator="gpu", devices=1)
    logger.info(f"Loaded model on worker (force_compile={force_compile}, dynamic_shapes={dynamic_shapes})")
    return model, nn_ag, trainer


def _compute_root_path(config_name, conditioning_mode, pt_name, seq_cond):
    """Per-protein output dir: inference/{config}/{seq_cond|seq_cath_cond|legacy}/{protein}/"""
    root_path = f"./inference/{config_name}"
    if conditioning_mode == "seq":
        root_path = os.path.join(root_path, "seq_cond")
    elif conditioning_mode == "seq_cath":
        root_path = os.path.join(root_path, "seq_cath_cond")
    else:
        root_path = os.path.join(root_path, "legacy")
    if seq_cond:
        root_path = os.path.join(root_path, pt_name)
    return root_path


def run_one_protein_in_process(model, nn_ag, trainer, cfg, *,
                                pt_name, config_name, conditioning_mode,
                                cath_codes_override):
    """Generate samples for one protein using a pre-loaded model + trainer.

    Returns a dict:
        status: 'OK' | 'SKIP' | 'FAILED'
        n_samples_written: int
        final_max_nsamples: int   (post-OOM-retry batch size, for caller persistence)
        root_path: str
        error / error_type (when FAILED)

    OOM handling: catches RuntimeError matching 'out of memory', clears CUDA cache,
    halves cfg.max_nsamples, rebuilds the dataloader, retries the SAME protein.
    Caller propagates the returned 'final_max_nsamples' to subsequent calls so the
    reduction sticks across proteins.
    """
    root_path = _compute_root_path(config_name, conditioning_mode, pt_name, cfg.seq_cond)

    # Resume detection
    existing_sample_count = 0
    rng_state_path = os.path.join(root_path, "_rng_state.pt")
    if os.path.exists(root_path):
        existing_pdbs = [f for f in os.listdir(root_path) if f.endswith('.pdb')]
        existing_sample_count = len(existing_pdbs)
        if existing_sample_count >= cfg.nsamples_per_len:
            logger.info(f"Output already complete for {pt_name} ({existing_sample_count} PDB files), skipping inference")
            return {"status": "SKIP", "n_samples_written": 0,
                    "final_max_nsamples": int(cfg.max_nsamples),
                    "root_path": root_path}
        if existing_sample_count > 0:
            logger.info(f"Found {existing_sample_count} existing PDB files for {pt_name}, will resume from sample {existing_sample_count}")
    os.makedirs(root_path, exist_ok=True)

    # RNG restore
    if existing_sample_count > 0 and os.path.exists(rng_state_path):
        logger.info(f"Restoring RNG state from {rng_state_path} (resuming from sample {existing_sample_count})")
        saved_rng = torch.load(rng_state_path, weights_only=False)
        torch.set_rng_state(saved_rng["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in saved_rng:
            torch.cuda.set_rng_state(saved_rng["cuda_rng_state"])
        np.random.set_state(saved_rng["numpy_rng_state"])
        random.setstate(saved_rng["python_rng_state"])
    elif existing_sample_count > 0:
        logger.warning(
            f"Resuming from {existing_sample_count} existing samples but no RNG state at {rng_state_path}. "
            f"Using fresh seed {cfg.seed} — new samples will be valid but RNG continuity is not guaranteed."
        )
        L.seed_everything(cfg.seed)
    else:
        logger.info(f"Seeding everything to seed {cfg.seed}")
        L.seed_everything(cfg.seed)

    # Load this protein's PT
    pt = torch.load(os.path.join(cfg.data_dir, "processed", f"{pt_name}.pt"), weights_only=False)
    if cath_codes_override is not None:
        cath_codes = cath_codes_override
    else:
        cath_codes = pd.read_csv(os.path.join(cfg.data_dir, cfg.cath_code_file))["cath_code"].tolist()

    residue_type_tensor = None
    mask_tensor = None
    if pt is not None:
        residue_type_tensor = getattr(pt, "residue_type", None)
        if residue_type_tensor is not None:
            residue_type_tensor = torch.as_tensor(residue_type_tensor).long()
        coord_mask = getattr(pt, "coord_mask", None)
        if coord_mask is not None:
            coord_mask = torch.as_tensor(coord_mask)
            mask_tensor = (coord_mask.sum(dim=-1) > 0).float()
        elif residue_type_tensor is not None:
            mask_tensor = torch.ones_like(residue_type_tensor, dtype=torch.float32)

    remaining_samples = cfg.nsamples_per_len - existing_sample_count

    save_trajectory = bool(cfg.get("save_trajectory", False))
    save_trajectory_gif = bool(cfg.get("save_trajectory_gif", False))
    trajectory_gif_fps = int(cfg.get("trajectory_gif_fps", 30))

    # OOM-retry loop: rebuild dataloader with halved batch on OOM
    while True:
        dataset = GenDatasetWithSeqCath(
            pt=pt,
            cath_codes=cath_codes if cfg.fold_cond else ["x.x.x.x"],
            dt=cfg.dt,
            nsamples_per_len=remaining_samples,
            max_nsamples=cfg.max_nsamples,
        )
        dataloader = DataLoader(dataset, batch_size=1)
        try:
            predictions = trainer.predict(model, dataloader)
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() and "OutOfMemoryError" not in str(e):
                raise
            torch.cuda.empty_cache()
            new_max = max(int(cfg.max_nsamples) // 2, 1)
            if new_max == int(cfg.max_nsamples):
                logger.error(f"OOM for {pt_name} at max_nsamples=1, giving up")
                return {"status": "FAILED", "error_type": "GPU_OOM", "error": str(e),
                        "n_samples_written": 0,
                        "final_max_nsamples": int(cfg.max_nsamples),
                        "root_path": root_path}
            cfg.max_nsamples = new_max
            logger.warning(f"OOM for {pt_name}, retrying with max_nsamples={new_max}")

    # RNG save
    rng_state_to_save = {
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "total_samples_completed": existing_sample_count + remaining_samples,
        "seed": cfg.seed,
    }
    if torch.cuda.is_available():
        rng_state_to_save["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(rng_state_to_save, rng_state_path)
    logger.info(f"Saved RNG state to {rng_state_path} (total samples completed: {cfg.nsamples_per_len})")

    aatype = np.array(pt.residue_type).astype(int) if pt is not None else None

    # Flatten cfg for compute_designability / compute_fid result rows
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())

    # Sample writing loop
    n_samples_written = 0
    samples_per_length = {}
    openfold_infer = None
    openfold_residue_type = None
    openfold_mask = None
    openfold_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for pred in predictions:
        coors_atom37 = pred["coords_atom37"]
        cath_codes_batch = pred["cath_code"]
        contact_map = pred.get("contact_map")
        distogram = pred.get("distogram")
        trajectory_tokens = pred.get("trajectory_tokens")
        if torch.is_tensor(trajectory_tokens):
            trajectory_tokens = trajectory_tokens.cpu()
        batch_size = len(cath_codes_batch)
        for i in range(batch_size):
            current_cath_code = cath_codes_batch[i][0]
            sample_key = (pt_name, current_cath_code)
            if sample_key not in samples_per_length:
                samples_per_length[sample_key] = existing_sample_count
            sample_idx = samples_per_length[sample_key]
            filename_parts = [pt_name]
            if cfg.fold_cond and current_cath_code != "x.x.x.x":
                filename_parts.append(f"cath_{current_cath_code}")
            filename_parts.append(str(sample_idx))
            if coors_atom37 is not None or distogram is not None:
                pdb_fname = "_".join(filename_parts) + ".pdb"
                pdb_path = os.path.join(root_path, pdb_fname)
                if distogram is not None and residue_type_tensor is not None and mask_tensor is not None:
                    if openfold_infer is None:
                        openfold_infer = OpenFoldDistogramOnlyInference(
                            model_name="model_1_ptm",
                            jax_params_path=os.path.expanduser("~/openfold/openfold/resources/params/params_model_1_ptm.npz"),
                            device=openfold_device,
                        )
                        openfold_residue_type = residue_type_tensor.to(openfold_device)
                        openfold_mask = mask_tensor.to(openfold_device)
                    dist_i = distogram[i] if distogram.dim() == 4 else distogram
                    if dist_i.dim() == 3:
                        dist_i = dist_i.unsqueeze(0)
                    with torch.no_grad():
                        out = openfold_infer(
                            dist_i.to(openfold_device),
                            openfold_residue_type.unsqueeze(0),
                            openfold_mask.unsqueeze(0),
                        )
                    atom37 = out["atom37"][0].detach().cpu().numpy()
                    write_prot_to_pdb(atom37, pdb_path, aatype=aatype, overwrite=True, no_indexing=True)
                elif coors_atom37 is not None:
                    write_prot_to_pdb(coors_atom37[i].numpy(), pdb_path, aatype=aatype, overwrite=True, no_indexing=True)
                n_samples_written += 1
            if contact_map is not None:
                contact_fname = "_".join(filename_parts) + "_contact.pt"
                contact_path = os.path.join(root_path, contact_fname)
                torch.save(contact_map[i].cpu(), contact_path)
            if distogram is not None:
                distogram_fname = "_".join(filename_parts) + "_distogram.pt"
                distogram_path = os.path.join(root_path, distogram_fname)
                torch.save(distogram[i].cpu(), distogram_path)
            if trajectory_tokens is not None and (save_trajectory or save_trajectory_gif):
                traj_i = trajectory_tokens[:, i] if trajectory_tokens.dim() == 4 else trajectory_tokens
                if save_trajectory:
                    traj_fname = "_".join(filename_parts) + "_trajectory.pt"
                    traj_path = os.path.join(root_path, traj_fname)
                    torch.save(traj_i, traj_path)
                if save_trajectory_gif:
                    mask_token = int(getattr(getattr(model, "discrete_diffusion", None), "mask_token", 2))
                    tokens_np = traj_i.numpy()
                    if mask_token != 2:
                        tokens_np = tokens_np.copy()
                        tokens_np[tokens_np == mask_token] = 2
                    frames = [_tokens_to_palette_image(frame) for frame in tokens_np]
                    if frames:
                        gif_fname = "_".join(filename_parts) + "_trajectory.gif"
                        gif_path = os.path.join(root_path, gif_fname)
                        duration = max(1, int(1000 / max(1, trajectory_gif_fps)))
                        frames[0].save(
                            gif_path, save_all=True, append_images=frames[1:],
                            duration=duration, loop=0,
                        )
            samples_per_length[sample_key] += 1

    # compute_fid / compute_designability — same logic as legacy main()
    if cfg.compute_fid or cfg.compute_designability:
        if cfg.compute_designability:
            columns += ["id_gen", "pdb_path", "L"]
            if cfg.compute_designability:
                columns += ["_res_scRMSD", "_res_scRMSD_all"]
            results = []
            samples_per_length2 = {}
            for pred in predictions:
                coors_atom37 = pred["coords_atom37"]
                cath_codes_b = pred["cath_code"]
                contact_map = pred.get("contact_map")
                distogram = pred.get("distogram")
                if coors_atom37 is None:
                    continue
                n = coors_atom37.shape[-3]
                if n not in samples_per_length2:
                    samples_per_length2[n] = 0
                for i in range(coors_atom37.shape[0]):
                    if cfg.seq_cond:
                        dir_name = f"n_{n}_{pt_name}_cath_{cath_codes_b[i]}"
                    else:
                        dir_name = f"n_{n}_id_{samples_per_length2[n]}"
                    samples_per_length2[n] += 1
                    sample_root_path = os.path.join(root_path, dir_name)
                    os.makedirs(sample_root_path, exist_ok=False)
                    fname = dir_name + ".pdb"
                    pdb_path = os.path.join(sample_root_path, fname)
                    write_prot_to_pdb(coors_atom37[i].numpy(), pdb_path, overwrite=True, no_indexing=True)
                    if contact_map is not None:
                        torch.save(contact_map[i].cpu(), os.path.join(sample_root_path, dir_name + "_contact.pt"))
                    if distogram is not None:
                        torch.save(distogram[i].cpu(), os.path.join(sample_root_path, dir_name + "_distogram.pt"))
                    res_row = list(flat_dict.values()) + [i, pdb_path, n]
                    if cfg.compute_designability:
                        res_designability = scRMSD(pdb_path, ret_min=False, tmp_path=sample_root_path)
                        res_row += [min(res_designability), res_designability]
                        print(res_designability)
                    results.append(res_row)
            df = pd.DataFrame(results, columns=columns)
        if cfg.compute_fid:
            samples_dir_fid = os.path.join(root_path, "samples_fid")
            os.makedirs(samples_dir_fid, exist_ok=True)
            list_of_pdbs = []
            for pred in predictions:
                coors_atom37 = pred["coords_atom37"]
                contact_map = pred.get("contact_map")
                distogram = pred.get("distogram")
                if coors_atom37 is None:
                    continue
                for i in range(coors_atom37.shape[0]):
                    sample_idx = len(list_of_pdbs)
                    pdb_path = os.path.join(samples_dir_fid, f"{sample_idx}_fid.pdb")
                    write_prot_to_pdb(coors_atom37[i].numpy(), pdb_path, overwrite=True, no_indexing=True)
                    list_of_pdbs.append(pdb_path)
                    if contact_map is not None:
                        torch.save(contact_map[i].cpu(), os.path.join(samples_dir_fid, f"{sample_idx}_fid_contact.pt"))
                    if distogram is not None:
                        torch.save(distogram[i].cpu(), os.path.join(samples_dir_fid, f"{sample_idx}_fid_distogram.pt"))
            res_row = list(flat_dict.values())
            for cfg_mf in cfg.metric_factory:
                if isinstance(model, Proteina):
                    assert cfg_mf.ca_only == True, "Please turn on ca_only for CAFlow model"
                metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
                metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
                for k, v in metrics.items():
                    columns += ["_res_" + k]
                    res_row += [v.cpu().item()]
            df = pd.DataFrame([res_row], columns=columns)
            df = df.drop("metric_factory", axis=1)
        if cfg.compute_fid:
            df.to_csv(os.path.join(root_path, "..", f"results_{config_name}_fid.csv"), index=False)
        else:
            csv_file = os.path.join(root_path, "..", f"results_{config_name}.csv")
            df.to_csv(csv_file, index=False)

    return {"status": "OK",
            "n_samples_written": n_samples_written,
            "final_max_nsamples": int(cfg.max_nsamples),
            "root_path": root_path}


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_base",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=0,
        help="Leave as 0.",
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="pt name containing the protein sequence to generate.",
    )
    parser.add_argument(
        "--force_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force compile the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show sampling progress bars and debug output (suppressed by default for clean logs).",
    )
    parser.add_argument(
        "--use_cueq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cuequivariance triangle updates when available.",
    )
    parser.add_argument(
        "--max_nsamples",
        type=int,
        default=None,
        help="Override max_nsamples (GPU batch size) from config.",
    )
    parser.add_argument(
        "--nsamples_per_protein",
        type=int,
        default=None,
        help="Override nsamples_per_len (total samples per protein) from config.",
    )
    parser.add_argument(
        "--cath_code",
        type=str,
        default=None,
        help="CATH code for this protein, e.g. '3.30.70' or '3.30.70.x' or 'x.x.x.x'. "
             "Overrides cfg.cath_code_file when provided. 3-level codes get padded with '.x' for H.",
    )
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default=None,
        choices=["seq", "seq_cath"],
        help="Which conditioning to apply this run. "
             "'seq' = sequence only (forces cath='x.x.x.x', fold_cond=False). "
             "'seq_cath' = sequence + top-1 CATH (requires --cath_code, sets fold_cond=True). "
             "If omitted, behavior is unchanged from baseline (uses cfg.fold_cond + cfg.cath_code_file).",
    )
    parser.add_argument(
        "--dynamic_shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass dynamic=True to torch.compile so a single compiled graph handles all "
             "protein lengths (no per-length recompile). --no-dynamic_shapes restores static-shape "
             "compilation.",
    )

    args = parser.parse_args()
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )

    # Inference config: use compose_inference_cfg() to flatten the unified
    # `inference:` block into the top-level cfg (so cfg.compute_designability,
    # cfg.nsamples_per_len, cfg.fold_cond, etc. all resolve at top level) and
    # to apply CLI overrides (--max_nsamples, --nsamples_per_protein,
    # --conditioning_mode, --cath_code). Returns cath_codes_override which is
    # plumbed to run_one_protein_in_process below.
    cfg, config_name, cath_codes_override = compose_inference_cfg(args)
    run_name = cfg.run_name_

    assert (
        not cfg.compute_designability or not cfg.compute_fid
    ), "Designability cannot be computed together with FID"

    model, nn_ag, trainer = load_model_for_worker(
        cfg,
        force_compile=args.force_compile,
        dynamic_shapes=args.dynamic_shapes,
        verbose=args.verbose,
        use_cueq=args.use_cueq,
    )

    result = run_one_protein_in_process(
        model, nn_ag, trainer, cfg,
        pt_name=args.pt,
        config_name=config_name,
        conditioning_mode=args.conditioning_mode,
        cath_codes_override=cath_codes_override,
    )
    if result.get("status") == "FAILED":
        sys.exit(1)


if __name__ == "__main__":
    main()
