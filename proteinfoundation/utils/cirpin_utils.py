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
from collections import defaultdict
from loguru import logger
from typing import List, Optional, Dict, Union
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


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
        
        rank_zero_info(f"Loading CIRPIN embeddings from {self.cirpin_emb_path}")
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
        
        rank_zero_info(f"Loaded CIRPIN embeddings for {len(self.id_to_index)} proteins")
        
        # Initialize empty CAT code mappings (will be populated from training data if needed)
        self.cat_to_proteins = defaultdict(list)
        self.protein_to_cat = {}
        rank_zero_info("CAT code mappings initialized (will be built from training data when available)")
    
    @staticmethod
    def _extract_cat_code_from_cath(cath_code: str) -> Optional[str]:
        """
        Extract CAT code (first 3 parts) from a full CATH code.
        
        Args:
            cath_code (str): Full CATH code like "3.30.70.20"
            
        Returns:
            str: CAT code like "3.30.70" or None if invalid
        """
        parts = cath_code.split('.')
        if len(parts) >= 3:
            try:
                # Validate that first 3 parts are numeric
                for i in range(3):
                    int(parts[i])
                return '.'.join(parts[:3])
            except ValueError:
                return None
        return None
    
    def build_cat_mappings_from_training_data(self, protein_to_cath_mapping: Dict[str, str]):
        """
        Build CAT code mappings from training data that contains CATH codes.
        
        Args:
            protein_to_cath_mapping (Dict[str, str]): Mapping from protein ID to full CATH code
        """
        self.cat_to_proteins = defaultdict(list)
        self.protein_to_cat = {}
        
        for protein_id, cath_code in protein_to_cath_mapping.items():
            # Extract CAT code using the static method
            cat_code = self._extract_cat_code_from_cath(cath_code)
            
            if cat_code:
                clean_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
                if clean_id in self.id_to_index:  # Only add if we have embeddings for this protein
                    self.protein_to_cat[clean_id] = cat_code
                    self.cat_to_proteins[cat_code].append(clean_id)
        
        rank_zero_info(f"Built CAT code mappings from training data: {len(self.cat_to_proteins)} unique CAT codes")
        self._log_cat_code_statistics()
    
    def _log_cat_code_statistics(self):
        """Log statistics about CAT code distribution."""
        if not self.cat_to_proteins:
            rank_zero_info("No CAT code mappings available yet. Use build_cat_mappings_from_training_data() to enable CAT-based sampling.")
            return
        
        rank_zero_info("CAT code statistics:")
        sorted_cats = sorted(self.cat_to_proteins.items(), key=lambda x: len(x[1]), reverse=True)
        
        for cat_code, proteins in sorted_cats[:10]:  # Log top 10
            rank_zero_info(f"  CAT {cat_code}: {len(proteins)} proteins")
        
        if len(sorted_cats) > 10:
            rank_zero_info(f"  ... and {len(sorted_cats) - 10} more CAT codes")
    
    def get_cat_code_statistics(self) -> Dict[str, int]:
        """
        Get statistics about protein counts per CAT code.
        
        Returns:
            Dict[str, int]: Mapping from CAT code to protein count
        """
        return {cat: len(proteins) for cat, proteins in self.cat_to_proteins.items()}
    
    def sample_embedding_by_cat_code(self, target_cat_code: str) -> Optional[torch.Tensor]:
        """
        Sample a random CIRPIN embedding from proteins with the same CAT code.
        
        Args:
            target_cat_code (str): CAT code to sample from
            
        Returns:
            torch.Tensor: Random CIRPIN embedding from the CAT code, or None if not found
        """
        if target_cat_code not in self.cat_to_proteins:
            return None
        
        # Randomly sample a protein from this CAT code using torch generator for reproducibility
        proteins = self.cat_to_proteins[target_cat_code]
        idx = torch.randint(0, len(proteins), (1,)).item()
        sampled_protein = proteins[idx]
        
        return self.get_embedding_by_id(sampled_protein)
    
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
                rank_zero_warn(f"Protein ID '{protein_id}' not found in CIRPIN embeddings")
        
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



