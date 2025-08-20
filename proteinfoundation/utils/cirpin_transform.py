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
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn, rank_zero_debug

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
        rank_zero_info(f"Initialized CIRPIN transform with {len(self.cirpin_loader)} embeddings, mode: {mode}")
        
        if mode == "sample":
            rank_zero_info("Sample mode requires CATH codes - will extract from data.cath_code field")
    
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
        
        # Add CIRPIN embedding to the data object with fallback approach
        # Use dummy sequence dimension to ensure proper batching, will be squeezed in feature factory
        data.cirpin_emb_fallback = cirpin_emb.unsqueeze(0)  # Shape: [1, 128] - will become [batch, 1, 128] after batching
        
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
            pass  # No warning needed - common case during training
        
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
            # No CATH code found, fall back to self mode
            return self._get_self_embedding(data)
        
        if cath_code is not None and len(cath_code) > 0:
            # Convert to string if needed
            if isinstance(cath_code, torch.Tensor):
                cath_code = cath_code.item() if cath_code.numel() == 1 else str(cath_code)
            # Use torch random generator for reproducibility across DDP
            if len(cath_code) > 1:
                idx = torch.randint(0, len(cath_code), (1,)).item()
                cath_code = cath_code[idx]
            else:
                cath_code = cath_code[0]
            
            # Extract CAT code using the centralized method
            cat_code = self.cirpin_loader._extract_cat_code_from_cath(cath_code)
            
            if cat_code:
                # Try to sample from this CAT code
                cirpin_emb = self.cirpin_loader.sample_embedding_by_cat_code(cat_code)
                if cirpin_emb is not None:
                    return cirpin_emb
                # No embeddings found for this CAT code, fall back to self mode
            # Could not extract CAT code or no embeddings found, fall back to self mode
        
        # Fallback to self mode if sampling fails
        return self._get_self_embedding(data)
    

    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_embeddings={len(self.cirpin_loader)}, mode={self.mode})"


class CirpinEmbeddingTransformWithMapping(CirpinEmbeddingTransform):
    """
    Enhanced CIRPIN transform that uses CATH codes from the CATHLabelTransform.
    
    This transform expects CATH codes to already be provided by the CATHLabelTransform
    and builds CAT code mappings from the available CIRPIN embeddings during initialization.
    """
    
    def __init__(self, cirpin_emb_path: str, protein_id_field: str = "id", mode: str = "self", 
                 cath_data_dir: str = None):
        """
        Initialize the enhanced CIRPIN embedding transform.
        
        Args:
            cirpin_emb_path (str): Path to the CIRPIN embeddings .pt file
            protein_id_field (str): Field name in the data object that contains the protein ID
            mode (str): "self" or "sample" mode for CIRPIN embedding selection
            cath_data_dir (str): Directory containing CATH data files (same as CATHLabelTransform)
        """
        super().__init__(cirpin_emb_path, protein_id_field, mode)
        
        if mode == "sample":
            if cath_data_dir:
                rank_zero_info("Building CAT mappings from CATH database")
                self._build_mappings_from_cath_data(cath_data_dir)
            else:
                rank_zero_info("No CATH data directory provided - sample mode will fall back to self mode")
        else:
            rank_zero_info("CIRPIN transform initialized in self mode")
    
    def _build_mappings_from_cath_data(self, cath_data_dir: str):
        """Build CAT code mappings using CATH database files."""
        try:
            # Import the CATHLabelTransform temporarily to access its parsing logic
            from proteinfoundation.datasets.transforms import CATHLabelTransform
            
            # Create a temporary CATH transform to get the mappings
            temp_cath_transform = CATHLabelTransform(cath_data_dir)
            pdbchain_to_cathid_mapping = temp_cath_transform.pdbchain_to_cathid_mapping
            cathid_to_cathcode_mapping = temp_cath_transform.cathid_to_cathcode_mapping
            
            # Build protein ID to CATH code mapping
            protein_to_cath_mapping = {}
            all_cirpin_proteins = set(self.cirpin_loader.get_all_protein_ids())
            
            for pdb_chain_id, cath_ids in pdbchain_to_cathid_mapping.items():
                # Check if this protein exists in CIRPIN embeddings
                if pdb_chain_id in all_cirpin_proteins:
                    for cath_id in cath_ids:
                        if cath_id in cathid_to_cathcode_mapping:
                            cath_code = cathid_to_cathcode_mapping[cath_id]
                            protein_to_cath_mapping[pdb_chain_id] = cath_code
                            break  # Use the first valid CATH code
            
            # Build CAT mappings from the protein-CATH mapping
            if protein_to_cath_mapping:
                self.cirpin_loader.build_cat_mappings_from_training_data(protein_to_cath_mapping)
                rank_zero_info(f"Built CAT mappings for {len(protein_to_cath_mapping)} proteins from CATH database")
            else:
                rank_zero_warn("No CAT mappings could be built from CATH database")
                
        except Exception as e:
            rank_zero_warn(f"Failed to build CAT mappings from CATH data: {e}")
            rank_zero_info("Sample mode will fall back to self mode")
    
    def __call__(self, data: Data) -> Data:
        """
        Apply the transform using pre-built CAT mappings.
        
        Args:
            data (Data): PyTorch Geometric Data object containing protein information
            
        Returns:
            Data: Modified data object with cirpin_emb field added
        """
        # Apply the transform with the pre-built mappings
        return super().__call__(data)


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
        rank_zero_info(f"Initialized batched CIRPIN transform with {len(self.cirpin_loader)} embeddings")
    
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

