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
import torch
import numpy as np
from loguru import logger
from typing import List, Optional, Dict, Union


class CirpinEmbeddingLoader:
    """
    Utility class for loading and managing CIRPIN embeddings.
    
    This class loads CIRPIN embeddings from a .pt file and provides efficient
    lookup by protein ID. It handles the embedding loading logic that was
    previously embedded in the feature factory classes.
    """
    
    def __init__(self, cirpin_emb_path: str):
        """
        Initialize the CIRPIN embedding loader.
        
        Args:
            cirpin_emb_path (str): Path to the .pt file containing CIRPIN embeddings
        """
        self.cirpin_emb_path = os.path.expanduser(cirpin_emb_path)
        self.id_to_index = {}
        self.embeddings = None
        self.protein_ids = None
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load CIRPIN embeddings from the .pt file."""
        if not os.path.exists(self.cirpin_emb_path):
            raise FileNotFoundError(f"CIRPIN embeddings file not found: {self.cirpin_emb_path}")
        
        logger.info(f"Loading CIRPIN embeddings from {self.cirpin_emb_path}")
        cirpin_data = torch.load(self.cirpin_emb_path, map_location='cpu', weights_only=False)
        
        # Validate the format
        if 'ids' not in cirpin_data or 'embeddings' not in cirpin_data:
            raise ValueError("CIRPIN embeddings file must contain 'ids' and 'embeddings' fields")
        
        self.protein_ids = cirpin_data['ids']
        self.embeddings = cirpin_data['embeddings']
        
        # Validate shapes
        if len(self.protein_ids) != self.embeddings.shape[0]:
            raise ValueError(f"Number of IDs ({len(self.protein_ids)}) does not match number of embeddings ({self.embeddings.shape[0]})")
        
        if self.embeddings.shape[1] != 128:
            raise ValueError(f"CIRPIN embeddings must have dimension 128, got {self.embeddings.shape[1]}")
        
        # Create mapping from protein ID to embedding index
        for i, protein_id in enumerate(self.protein_ids):
            # Remove .pt suffix if present for consistent mapping
            clean_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
            self.id_to_index[clean_id] = i
        
        logger.info(f"Loaded CIRPIN embeddings for {len(self.id_to_index)} proteins")
    
    def get_embedding_by_id(self, protein_id: str) -> Optional[torch.Tensor]:
        """
        Get CIRPIN embedding for a single protein ID.
        
        Args:
            protein_id (str): Protein ID to look up
            
        Returns:
            torch.Tensor: CIRPIN embedding of shape [128] or None if not found
        """
        clean_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
        idx = self.id_to_index.get(clean_id)
        
        if idx is not None:
            return self.embeddings[idx].clone()
        else:
            return None
    
    def get_embeddings_by_ids(self, protein_ids: List[str], 
                             fill_missing: bool = True,
                             device: Optional[torch.device] = None,
                             dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Get CIRPIN embeddings for a batch of protein IDs.
        
        Args:
            protein_ids (List[str]): List of protein IDs
            fill_missing (bool): Whether to fill missing embeddings with zeros
            device (torch.device, optional): Device to put the result tensor on
            dtype (torch.dtype, optional): Data type for the result tensor
            
        Returns:
            torch.Tensor: CIRPIN embeddings of shape [batch_size, 128]
        """
        batch_size = len(protein_ids)
        
        # Determine device and dtype
        if device is None:
            device = self.embeddings.device
        if dtype is None:
            dtype = self.embeddings.dtype
        
        # Create result tensor
        result = torch.zeros(batch_size, 128, dtype=dtype, device=device)
        
        # Fill in embeddings for known protein IDs
        for i, protein_id in enumerate(protein_ids):
            embedding = self.get_embedding_by_id(protein_id)
            if embedding is not None:
                result[i] = embedding.to(device=device, dtype=dtype)
            elif not fill_missing:
                logger.warning(f"Protein ID '{protein_id}' not found in CIRPIN embeddings")
        
        return result
    
    def has_protein_id(self, protein_id: str) -> bool:
        """
        Check if a protein ID exists in the loaded embeddings.
        
        Args:
            protein_id (str): Protein ID to check
            
        Returns:
            bool: True if the protein ID exists, False otherwise
        """
        clean_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
        return clean_id in self.id_to_index
    
    def get_all_protein_ids(self) -> List[str]:
        """
        Get all protein IDs in the loaded embeddings.
        
        Returns:
            List[str]: List of all protein IDs
        """
        return self.protein_ids.copy()
    
    def __len__(self) -> int:
        """Return the number of proteins in the embedding collection."""
        return len(self.protein_ids)



