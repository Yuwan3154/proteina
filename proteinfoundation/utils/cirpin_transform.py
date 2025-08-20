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
    
    Supports two modes:
    - "self": Use the protein's own CIRPIN embedding
    - "sample": Use a random CIRPIN embedding from a protein with the same CAT code
    """
    
    def __init__(self, cirpin_emb_path: str, protein_id_field: str = "id", mode: str = "self"):
        """
        Initialize the CIRPIN embedding transform.
        
        Args:
            cirpin_emb_path (str): Path to the CIRPIN embeddings .pt file
            protein_id_field (str): Field name in the data object that contains the protein ID
            mode (str): "self" or "sample" mode for CIRPIN embedding selection
        """
        self.cirpin_loader = CirpinEmbeddingLoader(cirpin_emb_path)
        self.protein_id_field = protein_id_field
        self.mode = mode
        logger.info(f"Initialized CIRPIN transform with {len(self.cirpin_loader)} embeddings, mode: {mode}")
        
        if mode == "sample":
            logger.info("Sample mode requires CATH codes - will extract from data.cath_code field")
    
    def __call__(self, data: Data) -> Data:
        """
        Apply the transform to add CIRPIN embeddings to the data object.
        
        Args:
            data (Data): PyTorch Geometric Data object containing protein information
            
        Returns:
            Data: Modified data object with cirpin_emb field added
        """
        if self.mode == "self":
            cirpin_emb = self._get_self_embedding(data)
        elif self.mode == "sample":
            cirpin_emb = self._get_sampled_embedding(data)
        else:
            logger.error(f"Unknown CIRPIN mode: {self.mode}, using zero embedding")
            cirpin_emb = torch.zeros(128, dtype=torch.float32)
        
        # Add CIRPIN embedding to the data object
        data.cirpin_emb = cirpin_emb
        
        return data
    
    def _get_self_embedding(self, data: Data) -> torch.Tensor:
        """Get the protein's own CIRPIN embedding."""
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
        
        return cirpin_emb
    
    def _get_sampled_embedding(self, data: Data) -> torch.Tensor:
        """Get a random CIRPIN embedding from proteins with the same CAT code."""
        # Extract CATH code from the data object
        cath_code = None
        
        if hasattr(data, 'cath_code'):
            cath_code = data.cath_code
        elif hasattr(data, 'fold_label'):
            cath_code = data.fold_label
        else:
            logger.warning("No CATH code found in data object for sample mode, falling back to self mode")
            return self._get_self_embedding(data)
        
        if cath_code is not None:
            # Convert to string if needed
            if isinstance(cath_code, torch.Tensor):
                cath_code = cath_code.item() if cath_code.numel() == 1 else str(cath_code)
            cath_code = str(cath_code)
            
            # Extract CAT code using the centralized method
            cat_code = self.cirpin_loader._extract_cat_code_from_cath(cath_code)
            
            if cat_code:
                # Try to sample from this CAT code
                cirpin_emb = self.cirpin_loader.sample_embedding_by_cat_code(cat_code)
                if cirpin_emb is not None:
                    return cirpin_emb
                else:
                    logger.debug(f"No CIRPIN embeddings found for CAT code '{cat_code}', falling back to self mode")
            else:
                logger.debug(f"Could not extract CAT code from CATH code '{cath_code}', falling back to self mode")
        
        # Fallback to self mode if sampling fails
        return self._get_self_embedding(data)
    

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_embeddings={len(self.cirpin_loader)}, mode={self.mode})"


class CirpinEmbeddingTransformWithMapping(CirpinEmbeddingTransform):
    """
    Enhanced CIRPIN transform that builds CAT code mappings from training data.
    
    This transform first scans the training data to build protein ID -> CAT code mappings,
    then uses those mappings for sampling mode.
    """
    
    def __init__(self, cirpin_emb_path: str, protein_id_field: str = "id", mode: str = "self", build_mappings: bool = True):
        """
        Initialize the enhanced CIRPIN embedding transform.
        
        Args:
            cirpin_emb_path (str): Path to the CIRPIN embeddings .pt file
            protein_id_field (str): Field name in the data object that contains the protein ID
            mode (str): "self" or "sample" mode for CIRPIN embedding selection
            build_mappings (bool): Whether to build CAT code mappings from training data
        """
        super().__init__(cirpin_emb_path, protein_id_field, mode)
        self.build_mappings = build_mappings
        self.mappings_built = False
        self.protein_to_cath_cache = {}
        
        if mode == "sample" and build_mappings:
            logger.info("Enhanced CIRPIN transform will build CAT mappings from training data on first pass")
    
    def __call__(self, data: Data) -> Data:
        """
        Apply the transform with automatic mapping building.
        
        Args:
            data (Data): PyTorch Geometric Data object containing protein information
            
        Returns:
            Data: Modified data object with cirpin_emb field added
        """
        # Build mappings on first pass if needed
        if self.mode == "sample" and self.build_mappings and not self.mappings_built:
            self._collect_mapping_data(data)
        
        # Apply the transform
        return super().__call__(data)
    
    def _collect_mapping_data(self, data: Data):
        """Collect protein ID -> CATH code mapping data."""
        # Extract protein ID
        protein_id = None
        if hasattr(data, self.protein_id_field):
            protein_id = getattr(data, self.protein_id_field)
        elif hasattr(data, 'protein_id'):
            protein_id = data.protein_id
        elif hasattr(data, 'pdb_id'):
            protein_id = data.pdb_id
        elif hasattr(data, 'name'):
            protein_id = data.name
        
        # Extract CATH code
        cath_code = None
        if hasattr(data, 'cath_code'):
            cath_code = data.cath_code
        elif hasattr(data, 'fold_label'):
            cath_code = data.fold_label
        
        if protein_id is not None and cath_code is not None:
            # Convert to strings
            if isinstance(protein_id, torch.Tensor):
                protein_id = protein_id.item() if protein_id.numel() == 1 else str(protein_id)
            if isinstance(cath_code, torch.Tensor):
                cath_code = cath_code.item() if cath_code.numel() == 1 else str(cath_code)
            
            protein_id = str(protein_id)
            cath_code = str(cath_code)
            
            # Clean protein ID
            clean_protein_id = protein_id.replace('.pt', '') if protein_id.endswith('.pt') else protein_id
            
            # Store the mapping
            self.protein_to_cath_cache[clean_protein_id] = cath_code
    
    def finalize_mappings(self):
        """Finalize the CAT code mappings after data collection."""
        if self.protein_to_cath_cache:
            logger.info(f"Building CAT mappings from {len(self.protein_to_cath_cache)} collected protein-CATH pairs")
            self.cirpin_loader.build_cat_mappings_from_training_data(self.protein_to_cath_cache)
            self.mappings_built = True
        else:
            logger.warning("No protein-CATH mappings collected, CAT-based sampling may not work")


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

