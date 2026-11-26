"""T8b SEQRES positions through their CONSUMERS, at the T8b yaml's values (CPU, toy files in a temp dir).

1. End to end: toy chains with SEQRES gaps wider than the AF2 clip -> the T8b dataset yaml's SeqresIndexTransform +
   TopologyReferenceTransform (drop_prob 0.25, drop_ref_len_range [14, 80], a toy synthetic index with real template
   rows) -> the real DensePaddingDataLoader collate -> ContactMapTriV2 built from the T8b nn yaml. The relpos the model
   embeds equals AF2 clip(p_i - p_j, -32, 32) + 32 on the rebased label_seq_id (query-query and query-template), for
   both served and dropped references.
2. ModelTrainerBase.generate(residue_positions=...): a recording network sees residue_seqres_idx, unchanged, at every
   step, on the discrete (MD4) path and on fm.full_simulation (the T8b path: discrete diffusion off in contactflow).
3. The masked validation arm (_build_self_reference_topology, topology_masked) on a padded batch whose second chain
   has a False mask tail (collate pad -1): its placeholder midpoints span that chain's own SEQRES span.
4. _run_validation_trajectory forwards batch["residue_seqres_idx"][:nsamples] to generate() (fixed-chain path and
   self-reference path) and passes None after a variable-length redraw; _dump_contact_sample writes the positions.
Run: python scratchpad/test_t8b_consumers.py
"""

import hashlib
import os
import pathlib
import sys
import tempfile
import types

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from proteinfoundation.flow_matching.discrete_md4 import MD4DiscreteDiffusion  # noqa: E402
from proteinfoundation.flow_matching.r3n_fm import FlowMatcher  # noqa: E402
from proteinfoundation.nn.contact_map_tri_v2 import ContactMapTriV2  # noqa: E402
from proteinfoundation.openfold_stub.np.residue_constants import restypes  # noqa: E402
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase  # noqa: E402
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="t8b_consumers_"))
# stem -> label_seq_id of the modelled residues (1-based mmCIF numbering); gaps of 47 and 48 exceed the clip of 32
CHAINS = {"4big_A": [1, 2, 3, 50, 51, 52, 100], "4big_B": [5, 6, 7, 8, 9]}
TEMPLATE_RUNS = {"4big_A": [(0, 3), (1, 8), (0, 4), (2, 5), (0, 2)], "4big_B": [(0, 2), (1, 6), (0, 3)]}
MB = ModelTrainerBase


def toy_seq(L):
    return "".join(restypes[i % 20] for i in range(L))


def write_side_table(path):
    stems = sorted(CHAINS)
    offset = np.cumsum([0] + [len(CHAINS[s]) for s in stems]).astype(np.int64)
    torch.save({"def": "toy", "stems": stems, "status": ["ok"] * len(stems), "offset": torch.from_numpy(offset),
                "label_seq_flat": torch.tensor([p for s in stems for p in CHAINS[s]], dtype=torch.int32),
                "length": torch.tensor([len(CHAINS[s]) for s in stems], dtype=torch.int32), "audit": {},
                "seq_sha1": [hashlib.sha1(toy_seq(len(CHAINS[s])).encode()).hexdigest() for s in stems]}, path)


def write_topology_index(path):
    """Synthetic-mode index: rows 0-1 the queries (clusters 0-1), rows 2-3 one template each (TM 0.7)."""
    stems = sorted(CHAINS)
    runs = [[], []] + [TEMPLATE_RUNS[s] for s in stems]
    sizes = [0, 0] + [len(r) for r in runs[2:]]
    he = [torch.zeros(n, n) for n in sizes]
    torch.save({
        "ids": stems + [f"{s}_t0" for s in stems], "element_types": (0, 1, 2),
        "cluster_of": torch.tensor([0, 1, 0, 1]), "members_offset": torch.tensor([0, 1, 2]),
        "members_flat": torch.tensor([2, 3]), "row_tm": torch.tensor([0.0, 0.0, 0.7, 0.7]),
        "runs_offset": torch.tensor(np.cumsum([0] + [len(r) for r in runs])),
        "runs_flat": torch.tensor([list(x) for r in runs for x in r]),
        "he_offset": torch.tensor(np.cumsum([0] + [n * n for n in sizes])), "he_size": torch.tensor(sizes),
        "he_flat": torch.cat([h.reshape(-1) for h in he]),
        "elem_feature_mean": 0.0, "elem_feature_std": 1.0,
    }, path)


def graph(stem):
    L = len(CHAINS[stem])
    return Data(protein_id=stem, coords=torch.zeros(L, 37, 3), coord_mask=torch.ones(L, 37, dtype=torch.bool),
                residue_type=torch.arange(L) % 20)


def compose(path, name):
    with hydra.initialize(path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name=name)
    OmegaConf.set_struct(cfg, False)
    return cfg


def t8b_transforms():
    idx_dir = TMP / "t8b_index"
    idx_dir.mkdir(exist_ok=True)
    write_topology_index(idx_dir / "topology_index_confind_synth_t8b.pt")
    write_side_table(TMP / "seqres.pt")
    ids = TMP / "ids.txt"
    ids.write_text("".join(f"{s}\n" for s in CHAINS))
    os.environ.update(DATA_PATH=str(TMP), SYNTH_INDEX_DIR=str(idx_dir), SEQRES_INDEX_PATH=str(TMP / "seqres.pt"))
    specs = compose("../configs/datasets_config/pdb", "pdb_train_contact-confind-synthtopo_S25_max512_t8b").datamodule.transforms
    by = {t._target_.rsplit(".", 1)[1]: t for t in specs}
    sq = hydra.utils.instantiate(by["SeqresIndexTransform"], required_ids_files=[str(ids)])
    tr = hydra.utils.instantiate(by["TopologyReferenceTransform"])
    return sq, tr


def t8b_model():
    nn_cfg = compose("../configs/experiment_config/model/nn", "contact_map_tri_v2_t8b")
    kw = OmegaConf.to_container(nn_cfg)
    kw.update(use_torch_compile=False, use_torch_compile_sc=False)  # CPU test; compile does not touch positions
    assert kw["query_pos"] == "seqres" and kw["max_rel_pos"] == 32
    return ContactMapTriV2(**kw).eval()


def af2_relpos(p, q):
    return (p[:, None] - q[None, :]).round().long().clamp(-32, 32) + 32


def test_end_to_end():
    torch.manual_seed(0)
    sq, tr = t8b_transforms()
    assert tr.drop_prob == 0.25 and tuple(tr.drop_ref_len_range) == (14, 80) and tr.placeholder_span == "seqres"
    m = t8b_model()
    got = []
    m.z_embed.rel_pos.register_forward_pre_hook(lambda mod, a: got.append(a[0].clone()))
    n_kept = n_dropped = 0
    for rep in range(12):
        data = [tr(sq(graph(s))) for s in sorted(CHAINS)]
        b = next(iter(DensePaddingDataLoader(data, batch_size=2, shuffle=False)))
        b = b if isinstance(b, dict) else b.to_dict()
        mask = b["mask_dict"]["coords"][..., 0, 0]
        B, Lb = mask.shape
        nn_in = {"contact_map_t": torch.zeros(B, Lb, Lb), "mask": mask.float(), "t": torch.full((B,), 0.5),
                 "residue_type": b["residue_type"].clamp(min=0), "residue_seqres_idx": b["residue_seqres_idx"]}
        for k in ("topology_he_tokens", "topology_he_pos_raw", "topology_he_feat", "topology_he_elem_feat"):
            nn_in[k] = b[k]
        got.clear()
        with torch.no_grad():
            m(nn_in)
        rel = got[0]
        Tb = b["topology_he_tokens"].shape[1]
        Lp = rel.shape[1] - m._bucket(Tb, m.pad_topo_buckets)
        for k, stem in enumerate(sorted(CHAINS)):
            p = torch.tensor(CHAINS[stem]) - CHAINS[stem][0]          # rebased label_seq_id, written independently
            L = len(p)
            assert torch.equal(b["residue_seqres_idx"][k, :L], p) and bool((b["residue_seqres_idx"][k, L:] == -1).all())
            T = int((b["topology_he_tokens"][k] > 0).sum())
            h = b["topology_he_pos_raw"][k, :T].float()
            assert torch.equal(rel[k, :L, :L], af2_relpos(p.float(), p.float())), stem
            assert torch.equal(rel[k, :L, Lp:Lp + T], af2_relpos(p.float(), h)), stem
            assert torch.equal(rel[k, Lp:Lp + T, :L], af2_relpos(h, p.float())), stem
            if b["topology_ref_id"][k] == "MASK":
                n_dropped += 1
                assert torch.allclose(h, (torch.arange(T) + 0.5) * (int(p[-1]) + 1) / T)
            else:
                n_kept += 1
                assert T == len(TEMPLATE_RUNS[stem])
        assert int(rel[0, 2, 3]) == 0 and int(rel[0, 3, 2]) == 64, "a 47-residue SEQRES jump saturates the clip"
    assert n_kept > 0 and n_dropped > 0, (n_kept, n_dropped)
    print(f"PASS end to end (T8b yaml transforms + dense collate + T8b nn yaml): relpos == AF2 clip(p_i - p_j, -32, 32) "
          f"+ 32 on the rebased label_seq_id, query-query and query<->template, {n_kept} served + {n_dropped} dropped "
          f"references over 24 chain draws; dropped midpoints span the SEQRES span")


class GenStub:
    """Just the parts of the trainer generate() and its guidance wrapper read; the methods are the real ones."""
    generate = MB.generate
    predict_clean_n_v_w_guidance = MB.predict_clean_n_v_w_guidance
    _eval_forward_module = MB._eval_forward_module
    _nn_out_to_x_clean = MB._nn_out_to_x_clean
    _nn_out_to_c_clean = MB._nn_out_to_c_clean
    _nn_out_to_c_logits = MB._nn_out_to_c_logits
    _draw_fold_label_once = MB._draw_fold_label_once
    _discrete_diffusion_enabled = MB._discrete_diffusion_enabled
    _dssp_diffusion_enabled = MB._dssp_diffusion_enabled
    _pair_mask = MB._pair_mask
    _apply_pair_mask = MB._apply_pair_mask
    _contact_tokens_to_input = MB._contact_tokens_to_input
    _contact_tokens_to_output = MB._contact_tokens_to_output

    def __init__(self, nn, discrete):
        self.nn, self.nn_sc, self._nn_sc_aliased = nn, None, False
        self.cfg_exp = OmegaConf.create({"model": {"target_pred": "c_1", "nn": {}}})
        self.motif_conditioning, self.contact_map_mode, self.device = False, True, torch.device("cpu")
        self.fm = FlowMatcher(modality="contact_map", zero_com=True, scale_ref=1.0)
        self.discrete_diffusion = MD4DiscreteDiffusion(vocab_size=2, timesteps=5) if discrete else None
        self.discrete_diffusion_type = "md4" if discrete else None


def test_generate_forwarding():
    m = t8b_model()
    seen = []
    m.register_forward_pre_hook(lambda mod, a: seen.append(a[0].get("residue_seqres_idx")))
    pos = torch.tensor([[0, 1, 2, 49, 50, 51, 99], [0, 1, 2, 3, 4, -1, -1]])
    mask = pos >= 0
    for discrete in (True, False):
        seen.clear()
        stub = GenStub(m, discrete)
        with torch.no_grad():
            stub.generate(nsamples=2, n=7, dt=0.25, self_cond=True, mask=mask, sc_scale_noise=0.45, sc_scale_score=1.0, residue_positions=pos.clone())
        n_steps = 5 if discrete else 4
        assert len(seen) == n_steps and all(s is not None and torch.equal(s, pos) for s in seen), (discrete, len(seen))
        print(f"PASS generate ({'discrete MD4' if discrete else 'fm.full_simulation'}): residue_seqres_idx reached the "
              f"network unchanged at {len(seen)} of {n_steps} steps")
    seen.clear()
    with pytest.raises(KeyError, match="residue_seqres_idx"):
        with torch.no_grad():
            GenStub(m, False).generate(nsamples=2, n=7, dt=0.25, self_cond=True, mask=mask, sc_scale_noise=0.45, sc_scale_score=1.0)
    print("PASS generate without residue_positions: the seqres model raises KeyError (no silent arange)")


class ValStub:
    _build_self_reference_topology = MB._build_self_reference_topology
    _topology_keys = MB._topology_keys
    _stack_topology = staticmethod(MB._stack_topology)
    _run_validation_trajectory = MB._run_validation_trajectory
    _diag_log = MB._diag_log
    _resolve_validation_length_bounds = MB._resolve_validation_length_bounds
    _use_variable_length_validation_sampling = MB._use_variable_length_validation_sampling
    _discrete_diffusion_enabled = MB._discrete_diffusion_enabled
    _dump_contact_sample = MB._dump_contact_sample
    TOPOLOGY_KEYS = MB.TOPOLOGY_KEYS

    def __init__(self, transform, vs):
        self.transform = transform
        self.cfg_exp = OmegaConf.create({"model": {"nn": {"topology_cond": True, "n_elem_features": 1}},
                                         "validation_sampling": vs,
                                         "training": {"fold_cond": False, "self_cond": True, "zero_sin_pos_emb": False}})
        self.device, self.contact_map_mode, self.discrete_diffusion = torch.device("cpu"), True, None
        self.trainer = types.SimpleNamespace(is_global_zero=True, datamodule=None)
        self.logger = types.SimpleNamespace(experiment=None)
        self.nn = None
        self.calls = []

    def _find_topology_transform(self):
        return self.transform

    def _topology_reference_map(self):
        return None


class StopAfterGenerate(Exception):
    pass


def val_batch():
    pos = torch.tensor([[0, 1, 2, 49, 50, 51, 99], [0, 1, 2, 3, 4, -1, -1]])  # collate pads longs with -1
    return {"mask": pos >= 0, "residue_seqres_idx": pos, "protein_id": sorted(CHAINS),
            "residue_type": torch.zeros(2, 7, dtype=torch.long)}


def test_masked_arm_and_val_forwarding():
    _, tr = t8b_transforms()
    stub = ValStub(tr, {"topology_masked": True, "topology_nonself_seed": 0})
    b = val_batch()
    topo = stub._build_self_reference_topology(b, 2, b["mask"])
    for k, (stem, span) in enumerate(zip(sorted(CHAINS), (100, 5))):
        n = int((topo["topology_he_tokens"][k] > 0).sum())
        h = topo["topology_he_pos_raw"][k, :n]
        assert torch.allclose(h, (torch.arange(n) + 0.5) * span / n), (stem, h)
        ref = tr.masked_reference(stem, int(b["mask"][k].sum()), seed=0, seqres_idx=b["residue_seqres_idx"][k, :int(b["mask"][k].sum())])
        assert torch.equal(h, ref["topology_he_pos_raw"])
    print("PASS masked validation arm on a padded batch (second chain: mask tail False, positions -1): midpoints over "
          "each chain's own SEQRES span (100 and 5), == masked_reference on the valid positions")

    def fake_generate(calls):
        def gen(**kw):
            calls.append(kw)
            raise StopAfterGenerate
        return gen

    cases = (
        ("fixed chain list", {"fixed_chain_list": "x"}, None, True),
        ("self-reference topology", {}, {"topology_he_tokens": torch.ones(2, 3, dtype=torch.long)}, True),
        ("variable-length redraw", {"min_length": 3, "max_length": 5}, None, False),
    )
    for name, vs, topo_ret, forwarded in cases:
        s = ValStub(tr, vs)
        s.generate = fake_generate(s.calls)
        s._build_self_reference_topology = lambda batch, nsamples, mask, t=topo_ret: t
        with pytest.raises(StopAfterGenerate):
            s._run_validation_trajectory(val_batch(), OmegaConf.create(vs), nsamples=2, qualitative_only=False)
        rp = s.calls[0]["residue_positions"]
        assert (torch.equal(rp, val_batch()["residue_seqres_idx"]) if forwarded else rp is None), (name, rp)
        print(f"PASS _run_validation_trajectory [{name}]: residue_positions "
              f"{'= batch residue_seqres_idx[:nsamples]' if forwarded else 'None (a seqres model then raises)'}")

    d = TMP / "dumps"
    for with_pos in (True, False):
        s = ValStub(tr, {})
        s._contact_dump_counter, s._last_sampling_ref_ids, s._align_calls = None, ["r0", "r1"], None
        vb = val_batch()
        if not with_pos:
            vb.pop("residue_seqres_idx")
        s._dump_contact_sample(str(d), vb, 1, torch.zeros(7, 7), torch.zeros(7, 7), vb["mask"][1], {})
        z = np.load(d / f"{sorted(CHAINS)[1]}_s00.npz")
        assert (z["residue_seqres_idx"].tolist() == [0, 1, 2, 3, 4]) if with_pos else ("residue_seqres_idx" not in z)
        os.remove(d / f"{sorted(CHAINS)[1]}_s00.npz")
    print("PASS _dump_contact_sample writes residue_seqres_idx[:L] ([0..4] for the padded chain); absent key -> not written")


if __name__ == "__main__":
    test_end_to_end()
    test_generate_forwarding()
    test_masked_arm_and_val_forwarding()
    print("ALL PASS")
