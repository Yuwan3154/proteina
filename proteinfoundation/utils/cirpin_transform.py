# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from loguru import logger
from typing import Optional
from torch_geometric.data import Data

from proteinfoundation.utils.cirpin_utils import CirpinEmbeddingLoader


class CirpinEmbeddingTransform:
    """
    Transform that adds CIRPIN embeddings to protein data objects.
    
    This transform loads CIRPIN embeddings and adds them as a field to the 
    protein data object. It can be used in the training dataset pipeline
    to provide CIRPIN embeddings to the model.
    """
    
    def __init__(self, cirpin_emb_path: str, protein_id_field: str = "id"):
        """
        Initialize the CIRPIN embedding transform.
        
        Args:
            cirpin_emb_path (str): Path to the CIRPIN embeddings .pt file
            protein_id_field (str): Field name in the data object that contains the protein ID
        """
        self.cirpin_loader = CirpinEmbeddingLoader(cirpin_emb_path)
        self.protein_id_field = protein_id_field
        logger.info(f"Initialized CIRPIN transform with {len(self.cirpin_loader)} embeddings")
    
    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to add CIRPIN embeddings to the data object.
        
        Args:
            data (Data): PyTorch Geometric Data object containing protein information
            
        Returns:
            Data: Modified data object with cirpin_emb field added
        """
        # Extract protein ID from the data object
        protein_id = None
        
        if hasattr(data, self.protein_id_field):
            protein_id = getattr(data, self.protein_id_field)
        elif hasattr(data, 'protein_id'):
            protein_id = data.protein_id
        elif hasattr(data, 'pdb_id'):
            protein_id = data.pdb_id
        elif hasattr(data, 'name'):
            protein_id = data.name
        else:
            logger.warning("No protein ID field found in data object, using zero embedding")
        
        # Get CIRPIN embedding for this protein
        if protein_id is not None:
            # Convert to string if needed and clean up
            if isinstance(protein_id, torch.Tensor):
                protein_id = protein_id.item() if protein_id.numel() == 1 else str(protein_id)
            protein_id = str(protein_id)
            
            # Clean protein ID (remove .pt suffix if present)
            clean_protein_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
            
            cirpin_emb = self.cirpin_loader.get_embedding_by_id(clean_protein_id)
            if cirpin_emb is None:
                logger.debug(f"Protein ID '{clean_protein_id}' not found in CIRPIN embeddings, using zero embedding")
                cirpin_emb = torch.zeros(128, dtype=torch.float32)
        else:
            cirpin_emb = torch.zeros(128, dtype=torch.float32)
        
        # Add CIRPIN embedding to the data object
        data.cirpin_emb = cirpin_emb
        
        return data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_embeddings={len(self.cirpin_loader)})"


class CirpinEmbeddingBatchTransform:
    """
    Transform that adds CIRPIN embeddings to batched protein data.
    
    This is used when data is already batched and we need to add CIRPIN
    embeddings for each sample in the batch.
    """
    
    def __init__(self, cirpin_emb_path: str):
        """
        Initialize the batched CIRPIN embedding transform.
        
        Args:
            cirpin_emb_path (str): Path to the CIRPIN embeddings .pt file
        """
        self.cirpin_loader = CirpinEmbeddingLoader(cirpin_emb_path)
        logger.info(f"Initialized batched CIRPIN transform with {len(self.cirpin_loader)} embeddings")
    
    def add_cirpin_to_batch(self, batch: dict, protein_ids: list) -> dict:
        """
        Add CIRPIN embeddings to a batch dictionary.
        
        Args:
            batch (dict): Batch dictionary containing protein data
            protein_ids (list): List of protein IDs for each sample in the batch
            
        Returns:
            dict: Modified batch with cirpin_emb field added
        """
        if len(protein_ids) == 0:
            return batch
        
        # Get CIRPIN embeddings for all proteins in the batch
        cirpin_embeddings = self.cirpin_loader.get_embeddings_by_ids(
            protein_ids,
            fill_missing=True,
            dtype=torch.float32
        )
        
        # Add to batch
        batch["cirpin_emb"] = cirpin_embeddings
        
        return batch
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_embeddings={len(self.cirpin_loader)})"

