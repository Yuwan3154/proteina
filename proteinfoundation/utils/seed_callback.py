# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import random
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class SeedCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["global_step"] = trainer.global_step
        # Save random states for all generators
        checkpoint["python_random_state"] = random.getstate()
        checkpoint["numpy_random_state"] = np.random.get_state()
        checkpoint["torch_random_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            checkpoint["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        global_step = checkpoint["global_step"]
        
        # Calculate per-rank seed to ensure different but reproducible seeds across ranks
        base_seed = checkpoint.get("base_seed", global_step)
        rank_seed = base_seed + trainer.global_rank
        
        # Seed everything with rank-specific seed
        L.seed_everything(rank_seed)
        
        # Also restore saved random states if available (for exact resumption)
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"]) 
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if "torch_cuda_random_state" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])
            
        print(f"Seeding rank {trainer.global_rank} with seed {rank_seed}")
        
    def on_train_start(self, trainer, pl_module):
        # Ensure different seeds per rank at training start
        if trainer.global_rank is not None:
            base_seed = getattr(pl_module, 'seed', 42)  # Default fallback
            rank_seed = base_seed + trainer.global_rank
            L.seed_everything(rank_seed)
            print(f"Initial seeding rank {trainer.global_rank} with seed {rank_seed}")
