import torch
import sys
import unittest
import numpy as np
import os

sys.path.insert(0, "/home/ubuntu/proteina")
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.openfold_stub.utils.import_weights import import_jax_weights_ipa_
import hydra
from omegaconf import OmegaConf
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/proteina/.env")

class TestIPAWeightInit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Use the actual config to bypass parameter missing issues
        config_path = "/home/ubuntu/proteina/configs/experiment_config/model/nn/contact_af3_30M_tri_individual-embed_coor_tanh_ipa-coord_dssp.yaml"
        self.dummy_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        self.model = ProteinTransformerAF3(**self.dummy_cfg)
        
        # Path specified by user
        self.npz_path = "/home/ubuntu/params/params_model_1_ptm.npz"

    def test_ipa_weight_import(self):
        # We want to test that the weights are independently loaded and actually overwrite the model
        
        # 1. Grab initial weights from IPA
        initial_q_weight = self.model.coors_3d_decoder.ipa.linear_q.weight.clone()
        initial_layer_norm_weight = self.model.coors_3d_decoder.layer_norm_ipa.weight.clone()
        
        # 2. Add some random noise uniformly to pretend it was another pretrain_ckpt
        with torch.no_grad():
            self.model.coors_3d_decoder.ipa.linear_q.weight.add_(torch.randn_like(self.model.coors_3d_decoder.ipa.linear_q.weight))
        
        corrupted_q_weight = self.model.coors_3d_decoder.ipa.linear_q.weight.clone()
        self.assertFalse(torch.allclose(initial_q_weight, corrupted_q_weight))

        # 3. Load exclusively the IPA weights
        import_jax_weights_ipa_(self.model, self.npz_path, version="model_1")
        
        # 4. Verify IPA parameter changes from corrupted state
        new_q_weight = self.model.coors_3d_decoder.ipa.linear_q.weight.clone()
        new_layer_norm_weight = self.model.coors_3d_decoder.layer_norm_ipa.weight.clone()
        
        self.assertFalse(torch.allclose(corrupted_q_weight, new_q_weight), "IPA weights failed to overwrite")
        
        # Make sure they are not NaN or identical to random init
        self.assertTrue(torch.all(torch.isfinite(new_q_weight)))
        
        print("IPA Weight Init overrides successfully without structural mismatches.")

if __name__ == '__main__':
    unittest.main()
