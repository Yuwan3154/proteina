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
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.utils.cirpin_utils import CirpinEmbeddingLoader


from proteinfoundation.metrics.designability import scRMSD
from proteinfoundation.metrics.metric_factory import (
    GenerationMetricFactory,
    generation_metric_from_list,
)
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import mask_cath_code_by_level, write_prot_to_pdb


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


class GenDatasetWithSeqCathCirpin(Dataset):
    """
    Dataset for sequence + CATH + CIRPIN conditioning inference.
    
    This dataset generates samples for a single protein with CATH fold conditioning
    and CIRPIN conditioning using embeddings from a CIRPIN embeddings file.
    """
    def __init__(self, pt, cath_codes=None, cirpin_emb_file=None, dt=0.005, nsamples_per_len=10, max_nsamples=10):
        super(GenDatasetWithSeqCathCirpin, self).__init__()
        self.pt = pt
        self.dt = dt
        self.nsamples_per_len = nsamples_per_len
        self.max_nsamples = max_nsamples
        self.cath_codes = cath_codes if cath_codes is not None else ["x.x.x.x"]
        self.num_batches_per_cath = (nsamples_per_len + max_nsamples - 1) // max_nsamples # Equivalent to math.ceil

        # Load CIRPIN embeddings if provided
        self.cirpin_loader = None
        if cirpin_emb_file is not None:
            self.cirpin_loader = CirpinEmbeddingLoader(cirpin_emb_file)
            self.cirpin_ids = self.cirpin_loader.get_all_protein_ids()
            logger.info(f"Loaded CIRPIN embeddings for {len(self.cirpin_ids)} proteins")
            if len(self.cath_codes) == 1:
                self.cath_codes = self.cath_codes * len(self.cirpin_ids)

    def __len__(self):
        return len(self.cath_codes) * self.num_batches_per_cath
        
    def __getitem__(self, index):
        cath_idx = index // self.num_batches_per_cath
        batch_idx_for_cath = index % self.num_batches_per_cath
        
        current_cath_code = self.cath_codes[cath_idx]
        current_id = self.cirpin_ids[cath_idx]

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
        
        # Add CIRPIN embeddings if available
        if self.cirpin_loader is not None:
            # Get CIRPIN embedding for this protein
            protein_ids = [current_id for _ in range(num_samples_for_batch)]
            cirpin_emb = self.cirpin_loader.get_embeddings_by_ids(
                protein_ids, 
                fill_missing=True,  # Fill with zeros if not found
                dtype=torch.float32
            )
            result["cirpin_emb"] = cirpin_emb
            result["cirpin_ids"] = protein_ids
        
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
        return result


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
        nlens_dict = torch.load(cfg.nres_lens_distribution_path)
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


def parse_len_cath_code(cfg):
    """Load (len, cath_codes) joint distribution. Apply mask according to the guidance cath code level"""
    if cfg.get("len_cath_code_path") is not None:
        logger.info(
            f"Loading empirical (length, cath_code) distribution from {cfg.len_cath_code_path}"
        )
        _len_cath_codes = torch.load(cfg.len_cath_code_path)
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


if __name__ == "__main__":
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
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
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
    
    args = parser.parse_args()
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../configs/experiment_config"
    else:
        config_path = f"../configs/experiment_config/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")
        run_name = cfg.run_name_

    assert (
        not cfg.compute_designability or not cfg.compute_fid
    ), "Designability cannot be computed together with FID"

    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    if cfg.seq_cond:
        root_path = os.path.join(root_path, args.pt)
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path, exist_ok=True)

    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    if isinstance(ckpt_path, str) and ckpt_path.startswith('~/'):
        ckpt_path = os.path.expanduser(ckpt_path)
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    # Check if using lora and load model
    if not cfg.lora.use:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
    
    else: # If using lora, create lora layers and reload the state_dict
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
        logger.info("Re-create LoRA layers and reload the weights now")
        replace_lora_layers(
            model,
            cfg["lora"]["r"],
            cfg["lora"]["lora_alpha"],
            cfg["lora"]["lora_dropout"],
        )
        lora.mark_only_lora_as_trainable(model, bias=cfg["lora"]["train_bias"])
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Set inference variables and potentially load autoguidance
    nn_ag = None
    if (
        cfg.get("autoguidance_ratio", 0.0) > 0
        and cfg.get("guidance_weight", 1.0) != 1.0
    ):
        assert cfg.autoguidance_ckpt_path is not None
        ckpt_ag_file = cfg.autoguidance_ckpt_path
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)

    # Create inference dataset
    pt = torch.load(f"{cfg.data_dir}/processed/{args.pt}.pt")
    cath_codes = pd.read_csv(f"{cfg.data_dir}/{cfg.cath_code_file}")["cath_code"].tolist()
    assert args.pt is not None, "pt must be provided if seq_cond is True"
    
    # Choose the appropriate dataset class based on CIRPIN conditioning
    if cfg.get("cirpin_cond", False):
        dataset = GenDatasetWithSeqCathCirpin(
            pt=pt,
            cath_codes=cath_codes if cfg.fold_cond else ["x.x.x.x"],
            cirpin_emb_file=cfg.cirpin.cirpin_emb_path,
            dt=cfg.dt,
            nsamples_per_len=cfg.nsamples_per_len,
            max_nsamples=cfg.max_nsamples
        )
    else:
        dataset = GenDatasetWithSeqCath(
            pt=pt,
            cath_codes=cath_codes if cfg.fold_cond else ["x.x.x.x"],
            dt=cfg.dt,
            nsamples_per_len=cfg.nsamples_per_len,
            max_nsamples=cfg.max_nsamples
        )
    # nlens_dict = parse_nlens_cfg(cfg)
    # lens_sample, nsamples = split_nlens(
    #     nlens_dict, max_nsamples=cfg.max_nsamples, n_replica=1
    # )  # Assume running on 1 GPU
    # if cfg.fold_cond:
    #     len_cath_codes = parse_len_cath_code(cfg)
    # else:
    #     len_cath_codes = None
    # dataset = GenDataset(
    #     nres=lens_sample, nsamples=nsamples, dt=cfg.dt, len_cath_codes=len_cath_codes
    # )
    dataloader = DataLoader(dataset, batch_size=1)
    # Note: Batch size should be left as 1, it is not the actual batch size.
    # Each sample returned by this loader is a 3-tuple (L, nsamples, dt) where
    #   - L (int) is the number of residues in the proteins to be samples
    #   - nsamples (int) is the number of proteins to generate (happens in parallel),
    #     so if nsamples=10 it means that it will produce 10 proteins of length L (all sampled in parallel)
    #   - dt (float) step-size used for the ODE integrator
    #   - cath_code (Optional[List[str]]) cath code for conditional generation

    # Flatten config and use it to initialize results dataframes columns
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())

    # Sample the model
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    if pt is not None:
        aatype = np.array(pt.residue_type).astype(int)
    else:
        aatype = None
    
    samples_per_length = {}
    for pred in predictions:
        coors_atom37, cath_codes_batch, cirpin_ids_batch = pred  # [b, n, 37, 3], prediction_step returns atom37

        # Save each generation as a pdb file
        for i in range(coors_atom37.shape[0]):
            # Create directory where everything related to this sample will be stored
            current_cath_code = cath_codes_batch[i][0] # Get the actual CATH code string
            current_cirpin_id = cirpin_ids_batch[i][0] if cirpin_ids_batch[i] is not None else "unknown"

            # Initialize counter for this pt and cath_code if not present
            sample_key = (args.pt, current_cath_code, current_cirpin_id)
            if sample_key not in samples_per_length:
                samples_per_length[sample_key] = 0
            
            sample_idx = samples_per_length[sample_key]
            
            # Create simple filename based on conditioning
            filename_parts = [args.pt]
            if cfg.fold_cond and current_cath_code != "x.x.x.x":
                filename_parts.append(f"cath_{current_cath_code}")
            if cfg.get("cirpin_cond", False) and current_cirpin_id != "unknown":
                filename_parts.append(f"cirpin_{current_cirpin_id}")
            filename_parts.append(str(sample_idx))
            fname = "_".join(filename_parts) + ".pdb"
            pdb_path = os.path.join(
                root_path, fname
            )  # ./inference/conf_{}/n_{}_id_{}

            # Save generated structure as pdb
            write_prot_to_pdb(
                coors_atom37[i].numpy(),
                pdb_path,
                aatype=aatype,
                overwrite=True,
                no_indexing=True,
            )
            samples_per_length[sample_key] += 1
    
    if cfg.compute_fid or cfg.compute_designability:

        # Code for designability and
        # Store samples generated as pdbs and also scRMSD
        if cfg.compute_designability:

            # Add some columns to store per-sample results
            columns += ["id_gen", "pdb_path", "L"]
            if cfg.compute_designability:
                columns += ["_res_scRMSD", "_res_scRMSD_all"]

            results = []
            samples_per_length = {}
            for pred in predictions:
                coors_atom37, cath_codes = pred  # [b, n, 37, 3], prediction_step returns atom37
                n = coors_atom37.shape[-3]
                if n not in samples_per_length:
                    samples_per_length[n] = 0

                # Save each generation as a pdb file
                for i in range(coors_atom37.shape[0]):
                    # Create directory where everything related to this sample will be stored
                    if cfg.seq_cond:
                        dir_name = f"n_{n}_{args.pt}_cath_{cath_codes[i]}"
                    else:
                        dir_name = f"n_{n}_id_{samples_per_length[n]}"
                    samples_per_length[n] += 1
                    sample_root_path = os.path.join(
                        root_path, dir_name
                    )  # ./inference/conf_{}/n_{}_id_{}
                    os.makedirs(sample_root_path, exist_ok=False)

                    # Save generated structure as pdb
                    fname = dir_name + ".pdb"
                    pdb_path = os.path.join(sample_root_path, fname)
                    write_prot_to_pdb(
                        coors_atom37[i].numpy(),
                        pdb_path,
                        overwrite=True,
                        no_indexing=True,
                    )

                    res_row = list(flat_dict.values()) + [i, pdb_path, n]

                    # If needed run designability, storing all intermediate values generated in sample_root_path
                    if cfg.compute_designability:
                        res_designability = scRMSD(
                            pdb_path, ret_min=False, tmp_path=sample_root_path
                        )
                        res_row += [min(res_designability), res_designability]
                        print(res_designability)

                    results.append(res_row)

            # Create the dataframe with results
            df = pd.DataFrame(results, columns=columns)

        # Code for FID
        if cfg.compute_fid:
            # Create directory to store all samples
            samples_dir_fid = os.path.join(root_path, "samples_fid")
            os.makedirs(samples_dir_fid, exist_ok=True)

            # Store samples
            list_of_pdbs = []
            for pred in predictions:
                coors_atom37 = pred  # [b, n, 37, 3], prediction_step returns atom37
                for i in range(coors_atom37.shape[0]):
                    pdb_path = os.path.join(samples_dir_fid, f"{len(list_of_pdbs)}_fid.pdb")
                    write_prot_to_pdb(
                        coors_atom37[i].numpy(),
                        pdb_path,
                        overwrite=True,
                        no_indexing=True,
                    )
                    list_of_pdbs.append(pdb_path)

            # Initialize row with results
            res_row = list(flat_dict.values())

            # Compute metrics and add respective columns and values
            for cfg_mf in cfg.metric_factory:
                if isinstance(model, Proteina):
                    assert cfg_mf.ca_only == True, "Please turn on ca_only for CAFlow model"
                metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
                metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
                for k, v in metrics.items():
                    columns += ["_res_" + k]
                    res_row += [v.cpu().item()]

            # Create dataframe
            df = pd.DataFrame([res_row], columns=columns)
            df = df.drop("metric_factory", axis=1)  # For nicer table

        # Write results to csv file
        if cfg.compute_fid:
            df.to_csv(
                os.path.join(root_path, "..", f"results_{config_name}_fid.csv"), index=False
            )
        else:
            csv_file = os.path.join(root_path, "..", f"results_{config_name}.csv")
            df.to_csv(csv_file, index=False)
