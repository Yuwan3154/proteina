#!/usr/bin/env python3
"""Parallel 4x A6000 Stage-A evaluation & scoring for ConFind TopoRef `mpnn_chunk9`
(`chk_best_contact_precL_epoch=00000185_step=000000075069-EMA.ckpt`) across:
  1. `xrun` Held-Out Validation Eval Set (88 chains x 8 samples = 704 samples/arm):
     - `self` arm: native self-reference
     - `equal` arm: paired equal-length synthetic template (`v4` / `confind_v1` `ref_id`)
  2. `t8b_eval` / `T7-NEWEST` Test Set (98 chains / 70 clean primary chains x 8 samples = 784 samples):
     - `t7` arm: paired T7-NEWEST synthetic template (`v4` / `confind_v1` `ref_id`)

Writes `tri_gen_eval.py`-compatible dump directories (`<out_dir>/dumps/mpnn_<arm>_c<chunk>/samples.jsonl` +
`<stem>_sKK.npz`) for downstream `stageB_c2c_from_maps.py` and `t8b_eval_cebm.py`, and computes exact paired
Contact P@L, P@L/5, LR P@L/5, F1, and Alignment P@Q (`pq`, `pq_base`, `pq_gain`) with 2,000-rep bootstrap 95% CIs.
"""

import argparse
from collections import defaultdict
import glob
import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace

import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from proteinfoundation.datasets.sse_topology import runs_to_spans  # noqa: E402
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform  # noqa: E402
from proteinfoundation.proteinflow.proteina import Proteina  # noqa: E402

N_BOOT = 2000
K_SAMPLES = 8

XRUN_CHUNKS = {
    0: ["1cbk_A", "1csb_B", "1dm9_A", "1dmx_A", "1dtp_A", "1dxs_A", "1e0b_A", "1jpy_A", "1lxj_A", "1mnm_C", "1nsf_A"],
    1: ["1qgt_A", "1r6f_A", "1ui5_A", "1wwh_A", "1wyz_A", "1xff_A", "1y07_A", "1yoy_A", "1yvk_A", "1zn6_A", "2cvi_A"],
    2: ["2d2s_A", "2fu2_A", "2ibz_I", "2jhe_A", "2nnc_A", "2nyz_D", "2ozn_B", "2pt7_G", "2q0o_C", "2q1k_A", "2qgz_A"],
    3: ["2ret_A", "2waq_K", "3cue_C", "3i2b_A", "3j7a_U", "3kq0_A", "3kyi_B", "3l82_A", "3l8u_A", "3ljc_A", "3lpy_A"],
    4: ["3njr_A", "3nmd_A", "3onr_A", "3q2s_C", "3qw9_A", "3rb5_A", "3rpg_B", "3tek_A", "3v64_C", "3w8h_B", "3whj_A"],
    5: ["3wo4_C", "3wwt_A", "4dhe_A", "4emo_A", "4f2m_E", "4irm_A", "4iuh_A", "4jys_A", "4n40_A", "4oaz_A", "4raj_A"],
    6: ["4rt0_A", "4tma_J", "4wd7_A", "5b64_B", "5byp_A", "5duk_A", "5gt1_A", "5hk0_A", "5ip4_D", "5l7s_A", "5o1w_A"],
    7: ["5o31_Q", "5o5j_T", "5tqb_A", "5uvd_A", "5wft_A", "5x2e_A", "5xyv_C", "6jkg_A", "6jq1_A", "6mt7_A", "6p7l_A"],
}

T7_CHUNKS = {
    0: ["1dmx_A", "1jpy_A", "1npo_A", "1wyz_A", "2qck_A", "2wsh_A", "3djc_A", "3gxa_A", "3hn5_A", "3kq0_A", "3l8u_A", "3ljc_A", "3szq_A", "4emo_A", "4f2m_E", "4iqj_M", "4iuh_A", "4qaq_A", "4xb6_A", "5lnp_A", "5ovu_A", "5tvf_A", "6pfq_C"],
    1: ["1csb_B", "1dm9_A", "1dxs_A", "1ov8_A", "1qgt_A", "1yvk_A", "2anv_A", "2ret_A", "2ynq_A", "3a1m_A", "3ktm_A", "3rb5_A", "3rpg_B", "3wwt_A", "4bnd_A", "4oaz_A", "4p29_A", "4qq0_A", "5bon_A", "5kym_A", "5n83_A", "6jkg_A", "6jq1_A", "6mt7_A", "6o7y_A"],
    2: ["1e0b_A", "1rtu_A", "1uch_A", "1xff_A", "1y07_A", "1yoy_A", "2waq_K", "2y33_A", "3cue_C", "3i2b_A", "3qw9_A", "3u1w_A", "3u6u_A", "3v64_C", "3wyz_A", "4gel_A", "4py6_A", "4uap_A", "4z8q_B", "5epw_A", "5ip4_D", "5jp1_A", "5mql_A", "5o1w_A", "6p7l_A"],
    3: ["1cbk_A", "1nsf_A", "1or4_A", "1ui5_A", "2i3f_A", "2j16_A", "2j2z_B", "2jhe_A", "3i76_A", "3iee_A", "3j7a_U", "3l82_A", "3llq_A", "3zzo_A", "4a1r_A", "4dhe_A", "4he8_A", "4jys_A", "4rt0_A", "4tma_J", "5cj3_A", "5hk0_A", "5jbr_A", "6gr8_B", "6saz_B"],
}

XRUN_EQUAL_MAP = {
    "1cbk_A": "1cbk_A@rw217#31", "1csb_B": "1csb_B@rw226#21", "1dm9_A": "1dm9_A@rw226#21", "1dmx_A": "1dmx_A@rw275#13",
    "1dtp_A": "1dtp_A@rw253#15", "1dxs_A": "1dxs_A@rw361#1", "1e0b_A": "1e0b_A@rw294#15", "1jpy_A": "1jpy_A@rw248#11",
    "1lxj_A": "1lxj_A@rw253#14", "1mnm_C": "1mnm_C@rw185#25", "1nsf_A": "1nsf_A@rw248#26", "1qgt_A": "1qgt_A@rw212#35",
    "1r6f_A": "1r6f_A@rw262#24", "1ui5_A": "1ui5_A@rw248#24", "1wwh_A": "1wwh_A@rw194#31", "1wyz_A": "1wyz_A@rw226#31",
    "1xff_A": "1xff_A@rw248#23", "1y07_A": "1y07_A@rw221#32", "1yoy_A": "1yoy_A@rw235#25", "1yvk_A": "1yvk_A@rw217#35",
    "1zn6_A": "1zn6_A@rw253#15", "2cvi_A": "2cvi_A@rw226#32", "2d2s_A": "2d2s_A@rw262#22", "2fu2_A": "2fu2_A@rw330#8",
    "2ibz_I": "2ibz_I@rw217#30", "2jhe_A": "2jhe_A@rw199#30", "2nnc_A": "2nnc_A@rw253#24", "2nyz_D": "2nyz_D@rw289#13",
    "2ozn_B": "2ozn_B@rw226#25", "2pt7_G": "2pt7_G@rw208#35", "2q0o_C": "2q0o_C@rw235#23", "2q1k_A": "2q1k_A@rw298#17",
    "2qgz_A": "2qgz_A@rw294#18", "2ret_A": "2ret_A@rw149#39", "2waq_K": "2waq_K@rw339#4", "3cue_C": "3cue_C@rw244#25",
    "3i2b_A": "3i2b_A@rw303#11", "3j7a_U": "3j7a_U@rw257#18", "3kq0_A": "3kq0_A@rw271#21", "3kyi_B": "3kyi_B@rw307#15",
    "3l82_A": "3l82_A@rw262#20", "3l8u_A": "3l8u_A@rw298#5", "3ljc_A": "3ljc_A@rw307#5", "3lpy_A": "3lpy_A@rw348#3",
    "3njr_A": "3njr_A@rw375#0", "3nmd_A": "3nmd_A@rw221#34", "3onr_A": "3onr_A@rw257#26", "3q2s_C": "3q2s_C@rw239#30",
    "3qw9_A": "3qw9_A@rw266#20", "3rb5_A": "3rb5_A@rw239#19", "3rpg_B": "3rpg_B@rw208#21", "3tek_A": "3tek_A@rw275#19",
    "3v64_C": "3v64_C@rw262#18", "3w8h_B": "3w8h_B@rw167#45", "3whj_A": "3whj_A@rw217#22", "3wo4_C": "3wo4_C@rw226#28",
    "3wwt_A": "3wwt_A@rw271#19", "4dhe_A": "4dhe_A@rw244#28", "4emo_A": "4emo_A@rw194#31", "4f2m_E": "4f2m_E@rw221#20",
    "4irm_A": "4irm_A@rw221#32", "4iuh_A": "4iuh_A@rw212#21", "4jys_A": "4jys_A@rw230#21", "4n40_A": "4n40_A@rw244#20",
    "4oaz_A": "4oaz_A@rw199#33", "4raj_A": "4raj_A@rw312#13", "4rt0_A": "4rt0_A@rw244#22", "4tma_J": "4tma_J@rw176#32",
    "4wd7_A": "4wd7_A@rw244#27", "5b64_B": "5b64_B@rw221#31", "5byp_A": "5byp_A@rw271#20", "5duk_A": "5duk_A@rw266#22",
    "5gt1_A": "5gt1_A@rw239#17", "5hk0_A": "5hk0_A@rw235#19", "5ip4_D": "5ip4_D@rw217#27", "5l7s_A": "5l7s_A@rw230#30",
    "5o1w_A": "5o1w_A@rw266#10", "5o31_Q": "5o31_Q@rw99#17", "5o5j_T": "5o5j_T@rw257#25", "5tqb_A": "5tqb_A@rw253#20",
    "5uvd_A": "5uvd_A@rw248#23", "5wft_A": "5wft_A@rw212#26", "5x2e_A": "5x2e_A@rw289#19", "5xyv_C": "5xyv_C@rw230#30",
    "6jkg_A": "6jkg_A@rw257#26", "6jq1_A": "6jq1_A@rw239#22", "6mt7_A": "6mt7_A@rw257#23", "6p7l_A": "6p7l_A@rw257#19",
}

T7_MAP = {
    "1dmx_A": "1dmx_A@rw289#10", "1jpy_A": "1jpy_A@rw208#20", "1npo_A": "1npo_A@rw194#20", "1wyz_A": "1wyz_A@rw221#32",
    "2qck_A": "2qck_A@rw266#14", "2wsh_A": "2wsh_A@rw262#20", "3djc_A": "3djc_A@rw303#13", "3gxa_A": "3gxa_A@rw226#30",
    "3hn5_A": "3hn5_A@rw230#25", "3kq0_A": "3kq0_A@rw339#6", "3l8u_A": "3l8u_A@rw275#10", "3ljc_A": "3ljc_A@rw280#11",
    "3szq_A": "3szq_A@rw221#28", "4emo_A": "4emo_A@rw275#13", "4f2m_E": "4f2m_E@rw235#17", "4iqj_M": "4iqj_M@rw180#21",
    "4iuh_A": "4iuh_A@rw199#24", "4qaq_A": "4qaq_A@rw135#43", "4xb6_A": "4xb6_A@rw248#22", "5lnp_A": "5lnp_A@rw230#14",
    "5ovu_A": "5ovu_A@rw199#34", "5tvf_A": "5tvf_A@rw203#15", "6pfq_C": "6pfq_C@rw244#22", "1csb_B": "1csb_B@rw285#8",
    "1dm9_A": "1dm9_A@rw221#22", "1dxs_A": "1dxs_A@rw239#26", "1ov8_A": "1ov8_A@rw253#24", "1qgt_A": "1qgt_A@rw262#24",
    "1yvk_A": "1yvk_A@rw221#34", "2anv_A": "2anv_A@rw271#9", "2ret_A": "2ret_A@rw221#23", "2ynq_A": "2ynq_A@rw248#25",
    "3a1m_A": "3a1m_A@rw185#20", "3ktm_A": "3ktm_A@rw226#17", "3rb5_A": "3rb5_A@rw280#10", "3rpg_B": "3rpg_B@rw298#2",
    "3wwt_A": "3wwt_A@rw271#19", "4bnd_A": "4bnd_A@rw262#25", "4oaz_A": "4oaz_A@rw244#23", "4p29_A": "4p29_A@rw226#28",
    "4qq0_A": "4qq0_A@rw217#27", "5bon_A": "5bon_A@rw248#17", "5kym_A": "5kym_A@rw271#19", "5n83_A": "5n83_A@rw253#19",
    "6jkg_A": "6jkg_A@rw366#2", "6jq1_A": "6jq1_A@rw217#27", "6mt7_A": "6mt7_A@rw262#22", "6o7y_A": "6o7y_A@rw321#6",
    "1e0b_A": "1e0b_A@rw235#27", "1rtu_A": "1rtu_A@rw226#24", "1uch_A": "1uch_A@rw248#25", "1xff_A": "1xff_A@rw248#23",
    "1y07_A": "1y07_A@rw230#30", "1yoy_A": "1yoy_A@rw212#30", "2waq_K": "2waq_K@rw180#33", "2y33_A": "2y33_A@rw307#4",
    "3cue_C": "3cue_C@rw235#27", "3i2b_A": "3i2b_A@rw253#21", "3qw9_A": "3qw9_A@rw230#28", "3u1w_A": "3u1w_A@rw262#17",
    "3u6u_A": "3u6u_A@rw235#31", "3v64_C": "3v64_C@rw221#27", "3wyz_A": "3wyz_A@rw266#24", "4gel_A": "4gel_A@rw253#23",
    "4py6_A": "4py6_A@rw257#14", "4uap_A": "4uap_A@rw275#21", "4z8q_B": "4z8q_B@rw226#20", "5epw_A": "5epw_A@rw212#22",
    "5ip4_D": "5ip4_D@rw348#2", "5jp1_A": "5jp1_A@rw230#32", "5mql_A": "5mql_A@rw266#24", "5o1w_A": "5o1w_A@rw253#13",
    "6p7l_A": "6p7l_A@rw321#5", "1cbk_A": "1cbk_A@rw294#14", "1nsf_A": "1nsf_A@rw253#25", "1or4_A": "1or4_A@rw239#28",
    "1ui5_A": "1ui5_A@rw280#17", "2i3f_A": "2i3f_A@rw244#27", "2j16_A": "2j16_A@rw194#33", "2j2z_B": "2j2z_B@rw212#28",
    "2jhe_A": "2jhe_A@rw239#21", "3i76_A": "3i76_A@rw221#34", "3iee_A": "3iee_A@rw271#20", "3j7a_U": "3j7a_U@rw244#21",
    "3l82_A": "3l82_A@rw194#35", "3llq_A": "3llq_A@rw230#29", "3zzo_A": "3zzo_A@rw199#30", "4a1r_A": "4a1r_A@rw262#20",
    "4dhe_A": "4dhe_A@rw199#37", "4he8_A": "4he8_A@rw271#23", "4jys_A": "4jys_A@rw253#16", "4rt0_A": "4rt0_A@rw230#25",
    "4tma_J": "4tma_J@rw180#31", "5cj3_A": "5cj3_A@rw235#20", "5hk0_A": "5hk0_A@rw212#24", "5jbr_A": "5jbr_A@rw221#23",
    "6gr8_B": "6gr8_B@rw162#20", "6saz_B": "6saz_B@rw253#14",
}

T7_CHANGED_CHAINS = {
    "1or4_A", "1rtu_A", "2anv_A", "2i3f_A", "2qck_A", "2wsh_A", "2ynq_A", "3a1m_A", "3djc_A", "3hn5_A",
    "3i76_A", "3iee_A", "3szq_A", "3u1w_A", "3wyz_A", "4a1r_A", "4bnd_A", "4gel_A", "4qaq_A", "4qq0_A",
    "4uap_A", "5bon_A", "5cj3_A", "5epw_A", "5jbr_A", "5jp1_A", "5ovu_A", "6saz_B",
}


class _NullExp:
    def log(self, *args, **kwargs):
        pass


class _NullLogger:
    @property
    def experiment(self):
        return _NullExp()


def prepare_inputs(out_dir):
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    for k, stems in XRUN_CHUNKS.items():
        for arm in ("self", "equal"):
            with open(os.path.join(chunks_dir, f"{arm}_c{k}_q.txt"), "w") as fh:
                fh.write("\n".join(stems) + "\n")
            with open(os.path.join(chunks_dir, f"{arm}_c{k}.txt"), "w") as fh:
                fh.write("\n".join(s for s in stems for _ in range(K_SAMPLES)) + "\n")
    for k, stems in T7_CHUNKS.items():
        with open(os.path.join(chunks_dir, f"t7_c{k}_q.txt"), "w") as fh:
            fh.write("\n".join(stems) + "\n")
        with open(os.path.join(chunks_dir, f"t7_c{k}.txt"), "w") as fh:
            fh.write("\n".join(s for s in stems for _ in range(K_SAMPLES)) + "\n")

    with open(os.path.join(out_dir, "equal_ref_map.json"), "w") as fh:
        json.dump(XRUN_EQUAL_MAP, fh, indent=2)
    with open(os.path.join(out_dir, "t7_ref_map.json"), "w") as fh:
        json.dump(T7_MAP, fh, indent=2)


def gpu_worker(gpu_id, tasks, config_name, ema_ckpt, out_dir, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    dev = torch.device("cuda:0")
    t0 = time.time()

    with hydra.initialize_config_dir(os.path.join(REPO, "configs", "experiment_config"), version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=config_name)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False

    ds_dir = os.path.join(REPO, "configs", "datasets_config", cfg_exp.dataset_config_subdir)
    with hydra.initialize_config_dir(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    OmegaConf.set_struct(cfg_data, False)

    store_dir = os.path.join(out_dir, f".store_gpu{gpu_id}")
    os.makedirs(store_dir, exist_ok=True)
    model = Proteina(cfg_exp, store_dir=store_dir)
    ck = torch.load(ema_ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"GPU {gpu_id}: missing={missing} unexpected={unexpected}"
    model = model.to(dev).eval()

    dm = hydra.utils.instantiate(cfg_data.datamodule)
    # Speed up PDBDataset.__getitem__ during _fixed_val_batches: skip random training template loading
    # inside TopologyReferenceTransform.forward(), since _build_self_reference_topology builds the exact
    # evaluation reference via transform.self_reference(src, L).
    null_logger = _NullLogger()
    model.trainer = SimpleNamespace(
        datamodule=dm,
        is_global_zero=True,
        world_size=1,
        logger=null_logger,
        loggers=[null_logger],
    )
    topo_tf = model._find_topology_transform()
    assert topo_tf is not None, "TopologyReferenceTransform not found in datamodule.transform"
    topo_tf.drop_prob = 1.0
    topo_tf._ensure_loaded()

    print(f"[GPU {gpu_id}] Model & index loaded in {time.time() - t0:.1f}s; running {len(tasks)} tasks", flush=True)

    for arm, chunk_id in tasks:
        label = f"mpnn_{arm}_c{chunk_id}"
        dump_dir = os.path.join(out_dir, "dumps", label)
        samples_path = os.path.join(dump_dir, "samples.jsonl")
        q_file = os.path.join(out_dir, "chunks", f"{arm}_c{chunk_id}_q.txt")
        c_file = os.path.join(out_dir, "chunks", f"{arm}_c{chunk_id}.txt")
        want_stems = [l.strip() for l in open(q_file) if l.strip()]
        want_total = len(want_stems) * K_SAMPLES

        if os.path.exists(samples_path):
            have = sum(1 for l in open(samples_path) if l.strip())
            if have == want_total:
                print(f"[GPU {gpu_id}] {label} already complete ({have}/{want_total}), skipping", flush=True)
                continue
            for f in glob.glob(os.path.join(dump_dir, "*")):
                os.remove(f)
            os.rmdir(dump_dir)

        t_task = time.time()
        model._fixed_val_batches_cache = None
        model._topology_reference_map_cache = None
        model._contact_dump_counter = {}
        model._validation_contact_results = []

        vs = model.cfg_exp.validation_sampling
        vs.fixed_chain_list = c_file
        vs.topology_nonself = False
        if arm == "self":
            vs.topology_reference_map = None
        elif arm == "equal":
            vs.topology_reference_map = os.path.join(out_dir, "equal_ref_map.json")
        elif arm == "t7":
            vs.topology_reference_map = os.path.join(out_dir, "t7_ref_map.json")
        else:
            raise ValueError(f"Unknown arm: {arm}")
        vs.contact_dump_dir = dump_dir
        vs.dump_align_calls = True

        L.seed_everything(seed)
        fixed_batches = model._fixed_val_batches(vs)
        for fb in fixed_batches:
            model._run_validation_trajectory(
                fb,
                vs,
                nsamples=int(fb["mask"].shape[0]),
                qualitative_only=False,
            )

        rows = [json.loads(l) for l in open(samples_path) if l.strip()]
        assert len(rows) == want_total, f"{label}: expected {want_total} samples, got {len(rows)}"
        pl_vals = [r["metrics"]["contact_precision_at_L"] for r in rows]
        model._fixed_val_batches_cache = None
        torch.cuda.empty_cache()
        print(
            f"[GPU {gpu_id}] DONE {label}: {len(rows)} samples ({len(want_stems)} chains) in {time.time() - t_task:.1f}s "
            f"| chunk P@L mean={np.mean(pl_vals):.4f} median={np.median(pl_vals):.4f}",
            flush=True,
        )


def elements_of(tf, row):
    runs = tf._runs_for(row)
    spans = runs_to_spans(runs)
    keep = [i for i, (t, _) in enumerate(runs) if t in tf.element_types]
    L = sum(n for _, n in runs)
    elem = np.full(L, -1, dtype=np.int64)
    for a, i in enumerate(keep):
        s, e = spans[i]
        elem[s:e] = a
    return elem, np.asarray([runs[i][0] for i in keep])


def p_at_q(score, tgt, rows):
    L, T = score.shape
    aligned = (tgt >= 0) & rows
    q = int(aligned.sum())
    if q == 0:
        return None
    A = np.zeros((L, T), dtype=bool)
    A[np.flatnonzero(aligned), tgt[aligned]] = True
    s = np.where(rows[:, None], score, -np.inf).reshape(-1)
    a = A.reshape(-1)
    q = min(q, s.size)
    thr = np.sort(s)[::-1][q - 1]
    above, tie = s > thr, s == thr
    return float((a[above].sum() + (q - above.sum()) * a[tie].sum() / tie.sum()) / q)


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    lo, hi = np.quantile(b, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def score_all(out_dir, index_path):
    tf = TopologyReferenceTransform(
        index_path=index_path,
        reference_source="synthetic",
        reference_mode="mpnn_chunk9",
        chunk_size=9,
        dssp_def="tmalign_smooth",
        sse_types=(1, 2),
        element_types=(1,),
        elem_features=True,
    )
    tf._ensure_loaded()

    report_lines = []
    for arm, n_chunks in (("self", 8), ("equal", 8), ("t7", 4)):
        per_chain = defaultdict(lambda: defaultdict(list))
        for c in range(n_chunks):
            d = os.path.join(out_dir, "dumps", f"mpnn_{arm}_c{c}")
            spath = os.path.join(d, "samples.jsonl")
            if not os.path.exists(spath):
                continue
            for ln in open(spath):
                if not ln.strip():
                    continue
                r = json.loads(ln)
                z = np.load(os.path.join(d, r["file"]))
                L_q = int(z["L"])
                T = int(z["he_tokens"].shape[0])
                ref = str(z["ref_id"])
                row = tf._id_to_row[ref]
                if arm == "self":
                    elem, _ = elements_of(tf, row)
                    tgt = np.where(elem < T, elem, -1)
                else:
                    tgt = tf._align_for(row, L_q, T).numpy()
                q_pos = np.arange(L_q, dtype=np.float64)
                pos = -np.abs(q_pos[:, None] - z["he_pos_raw"][None, :T].astype(np.float64))
                head = z["align_logits_last"].astype(np.float64)
                allr = np.ones(L_q, dtype=bool)
                pq = p_at_q(head, tgt, allr)
                pq_base = p_at_q(pos, tgt, allr)
                st = r["stem"]
                if pq is not None and pq_base is not None:
                    per_chain[st]["pq"].append(pq)
                    per_chain[st]["pq_base"].append(pq_base)
                    per_chain[st]["pq_gain"].append(pq - pq_base)
                m = r["metrics"]
                per_chain[st]["pl"].append(float(m["contact_precision_at_L"]))
                per_chain[st]["pl5"].append(float(m.get("contact_precision_at_L5", 0.0)))
                per_chain[st]["lr_pl5"].append(float(m.get("contact_long_range_precision_at_L5", 0.0)))
                per_chain[st]["f1"].append(float(m.get("contact_f1", 0.0)))

        stems_all = sorted(per_chain.keys())
        if not stems_all:
            continue

        subsets = [("all", stems_all)]
        if arm == "t7":
            clean_stems = [s for s in stems_all if s not in T7_CHANGED_CHAINS]
            subsets = [("clean_70", clean_stems), ("all_98", stems_all)]

        for sub_name, stems in subsets:
            rng = np.random.default_rng(0)
            report_lines.append(f"\n=== ARM: {arm} ({sub_name}, n_chains={len(stems)}, {K_SAMPLES} samples/chain) ===")
            report_lines.append(f"{'metric':<14} {'chains':>7} {'mean':>8} {'[95% CI]':>18} {'median_of_means':>16}")
            for k in ("pq", "pq_base", "pq_gain", "pl", "pl5", "lr_pl5", "f1"):
                vals = [float(np.mean(per_chain[s][k])) for s in stems if per_chain[s][k]]
                if vals:
                    mu, lo, hi = boot_ci(vals, rng)
                    med = float(np.median(vals))
                    report_lines.append(f"{k:<14} {len(vals):>7} {mu:8.4f}  [{lo:.4f}, {hi:.4f}] {med:16.4f}")

    text = "\n".join(report_lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "stageA_summary.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", default="training_contact_tri_v2_confind_mpnn_chunk9")
    ap.add_argument(
        "--ema_ckpt",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_precL_epoch=00000185_step=000000075069-EMA.ckpt",
    )
    ap.add_argument(
        "--index",
        default="/home/jupyter-chenxi/sse1d_idx/mpnn_chunk9/topology_index_confind_synth_mpnn_chunk9.pt",
    )
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/mpnn_chunk9_eval")
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    prepare_inputs(args.out_dir)

    if not args.score_only:
        all_tasks = (
            [("t7", k) for k in range(4)]
            + [("equal", k) for k in range(8)]
            + [("self", k) for k in range(8)]
        )
        tasks_per_gpu = {g: all_tasks[g :: args.ngpus] for g in range(args.ngpus)}
        ctx = mp.get_context("spawn")
        procs = []
        for g in range(args.ngpus):
            p = ctx.Process(
                target=gpu_worker,
                args=(g, tasks_per_gpu[g], args.config_name, args.ema_ckpt, args.out_dir, args.seed),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"GPU worker exited with code {p.exitcode}")

    score_all(args.out_dir, args.index)


if __name__ == "__main__":
    main()
