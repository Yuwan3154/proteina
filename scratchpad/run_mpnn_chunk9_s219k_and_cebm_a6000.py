"""Evaluate step=218,852 (s219k, all-time #1 validation_sampling/contact_f1_median = 0.5532) and
step=207,842 (s208k, #3 validation_sampling/contact_f1_median = 0.5461, F1 mean = 0.5152, P@L mean = 0.5152)
plus multi-checkpoint ContactEBM pools on A6000_offsite across all 4 RTX A6000 GPUs using dynamic mp.Queue
work-stealing, AND evaluate mpnn_chunk9 + ContactEBM + c2c on the 10-target bad_afdb benchmark
(lad_r250, tmpl, self) against SSE-based cfft / sse_ens24_cebm baselines.

Phases:
  Phase 1: Dynamic-queue Stage-A (tri 50-step ODE rollout, 8 samples/chain) across all 4 GPUs:
           - s219k on T7-NEWEST (4 chunks, 98 chains) + toporef_xrun (equal & self, 16 chunks, 88 chains each)
           - s208k on T7-NEWEST (4 chunks, 98 chains)
  Phase 2: Dynamic-queue ContactEBM (t_eval=0.25, t0=0.30, K=0, minE/pick/final) on T7-NEWEST:
           - s208k_cebm (8-sample), s219k_cebm (8-sample)
           - pool16_189_219_cebm (16-sample: s189k + s219k)
           - pool24_160_189_219_cebm (24-sample: s160k + s189k + s219k)
           - pool24_189_208_219_cebm (24-sample: s189k + s208k + s219k)
           - pool32_160_189_208_219_cebm (32-sample: s160k + s189k + s208k + s219k)
  Phase 3: Dynamic-queue Stage-B c2c folding across all 4 GPUs for all new Stage-A and ContactEBM chunks.
  Phase 4: Dynamic-queue 10-target bad_afdb benchmark (lad_r250, tmpl, self) comparing mpnn_chunk9
           (s189k, s219k, pool16_189_219 + ContactEBM) vs SSE cfft / tmalign_smooth / sse_ens24_cebm.
"""

import argparse
from collections import defaultdict
import glob
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
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
sys.path.insert(0, HERE)

from scratchpad.run_mpnn_chunk9_eval_a6000 import (  # noqa: E402
    K_SAMPLES,
    T7_CHANGED_CHAINS,
    _NullLogger,
    boot_ci,
    prepare_inputs,
)
from scratchpad.run_mpnn_chunk9_s160k_and_cebm_a6000 import (  # noqa: E402
    CEBM_MD5,
    CEBM_TAGS,
    T0,
    T_EVAL,
    TRI_SEED,
    _cebm_pair_mask,
    cebm_worker,
    load_contact_ebm,
    score_clean,
)
from scratchpad.run_mpnn_chunk9_stageB_a6000 import worker_loop as stageB_worker_loop  # noqa: E402
from proteinfoundation.datasets.atom_features import atom14_features  # noqa: E402
from proteinfoundation.datasets.cath_utils import load_cath_mapping  # noqa: E402
from proteinfoundation.datasets.sse_topology import (  # noqa: E402
    DSSP_HELIX,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.nn.af3_diffusion import C2C_INFERENCE_STEPS, SIGMA_DATA  # noqa: E402
from proteinfoundation.nn.mpnn_topology_encoder import (  # noqa: E402
    MPNN_ATOM4_INDICES,
    chunk9_runs,
)
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer  # noqa: E402
from proteinfoundation.proteinflow.proteina import Proteina  # noqa: E402
from proteinfoundation.utils import precompute_synthetic_topology_index as builder  # noqa: E402
from proteinfoundation.utils.c2c_dump import write_atom14_pdb  # noqa: E402
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR  # noqa: E402
from proteinfoundation.utils.topology_from_structure import parse_structure  # noqa: E402

C2C_SEED_BASE = 1234
F2C_MD5 = "e928f9bff6a6020538e8dbc56e928202"


def prepare_bad_afdb_bundle(raw_json_path, out_pt_path, f2c_ckpt, f2c_root="/home/jupyter-chenxi"):
    """Build bad_afdb_10targets_inputs.pt on GPU 0 from raw PDB texts using Frame2ConFind."""
    if os.path.exists(out_pt_path) and os.path.getsize(out_pt_path) > 0:
        return
    if not os.path.exists(raw_json_path):
        return
    os.environ.setdefault("FRAME2SEQ_HOME", os.path.join(f2c_root, "Frame2seq"))
    for p in (f2c_root, os.path.join(f2c_root, "Frame2seq")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    with open(f2c_ckpt, "rb") as fh:
        md5 = hashlib.file_digest(fh, "md5").hexdigest()
    assert md5 == F2C_MD5, f"Frame2ConFind ckpt md5 {md5} != {F2C_MD5}"
    from Frame2ConFind.inference.api import Frame2ConFindPredictor

    pred = Frame2ConFindPredictor(checkpoint=f2c_ckpt, amp_dtype="fp32", compile_model=False)
    raw = json.load(open(raw_json_path))
    cache = {}

    def _featurize_pdb_text(pdb_text):
        h = hashlib.sha1(pdb_text.encode()).hexdigest()
        if h in cache:
            return cache[h]
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as tf:
            tf.write(pdb_text)
            tpath = tf.name
        try:
            g = parse_structure(tpath, chain="A")
        finally:
            os.remove(tpath)
        coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
        amask = (g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].float() > 0.5).bool()
        full = torch.where(amask[..., None], coords, torch.zeros_like(coords))
        aatype = g.residue_type.long()
        L_s = int(amask.shape[0])
        x, mask = builder._f2c_inputs(full[None], amask)
        probs = pred.predict_batch(x, mask[None])[:, :L_s, :L_s].float().cpu()
        cm = (probs[0] >= builder.CONFIND_THRESHOLD).to(torch.uint8)
        cache[h] = (full, amask, aatype, cm)
        return cache[h]

    bundle = {"selected_pids": raw["selected_pids"], "targets": {}}
    for pid in raw["selected_pids"]:
        tr = raw["targets"][pid]
        fq, mq, aaq, cmq = _featurize_pdb_text(tr["native_pdb_text"])
        arms_out = {}
        for arm, ar in tr["arms"].items():
            ft, mt, aat, cmt = _featurize_pdb_text(ar["tmpl_pdb_text"])
            arms_out[arm] = {
                "ref_stem": ar["ref_stem"],
                "full_t0": ft,
                "amask_t0": mt,
                "aatype_t0": aat,
                "cm_tmpl0": cmt,
                "baselines": ar.get("baselines", {}),
            }
        bundle["targets"][pid] = {
            "pid": pid,
            "ci": int(tr["ci"]),
            "Lq": int(mq.shape[0]),
            "full_q": fq,
            "amask_q": mq,
            "aatype_q": aaq,
            "cm_nat": cmq,
            "native_pdb_text": tr["native_pdb_text"],
            "arms": arms_out,
        }
    os.makedirs(os.path.dirname(os.path.abspath(out_pt_path)), exist_ok=True)
    tmp_pt = out_pt_path + ".tmp"
    torch.save(bundle, tmp_pt)
    os.replace(tmp_pt, out_pt_path)
    del pred
    torch.cuda.empty_cache()
    print(f"[bad_afdb] Built {out_pt_path} ({len(bundle['selected_pids'])} targets)", flush=True)


# ───────────────────────── Dynamic Queue Multiprocessing Helpers ─────────────────────────
def run_queue_workers(worker_fn, tasks, ngpus, extra_args=()):
    """Dispatch `tasks` dynamically across `ngpus` worker processes via mp.Queue (zero straggler barrier)."""
    if not tasks:
        return
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for t in tasks:
        q.put(t)
    for _ in range(ngpus):
        q.put(None)
    procs = []
    for g in range(ngpus):
        p = ctx.Process(target=worker_fn, args=(g, q, *extra_args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Dynamic queue GPU worker exited with code {p.exitcode}")


def q_stageA_worker(gpu_id, task_q, config_name, ema_ckpt, out_dir, seed, label_prefix):
    """Persistent Stage-A worker pulling (arm, chunk_id) dynamically from task_q."""
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
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
    print(f"[GPU {gpu_id}] Loaded {label_prefix} ({os.path.basename(ema_ckpt)}) in {time.time() - t0:.1f}s", flush=True)

    while True:
        item = task_q.get()
        if item is None:
            break
        arm, chunk_id = item
        label = f"{label_prefix}_{arm}_c{chunk_id}"
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
        f1_vals = [r["metrics"].get("contact_f1", 0.0) for r in rows]
        model._fixed_val_batches_cache = None
        torch.cuda.empty_cache()
        print(
            f"[GPU {gpu_id}] DONE {label}: {len(rows)} samples ({len(want_stems)} chains) in {time.time() - t_task:.1f}s "
            f"| P@L mean={np.mean(pl_vals):.4f} F1 mean={np.mean(f1_vals):.4f}",
            flush=True,
        )


def q_cebm_worker(gpu_id, task_q, pack_path, cebm_repo, cebm_ckpt):
    """Persistent ContactEBM worker pulling single tasks from task_q."""
    my_tasks = []
    # Drain one by one while keeping model resident by delegating batches of 1..N or running inline
    # Since cebm_worker loads model once for `tasks`, let's load once and process queue items directly!
    from scratchpad.run_mpnn_chunk9_s160k_and_cebm_a6000 import cebm_reverse_sample_from_t0, pack_reader

    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    if cebm_repo not in sys.path:
        sys.path.insert(0, cebm_repo)
    with open(cebm_ckpt, "rb") as fh:
        md5 = hashlib.file_digest(fh, "md5").hexdigest()
    assert md5 == CEBM_MD5, f"ContactEBM ckpt md5 {md5} != {CEBM_MD5}"
    cebm_model, cebm_diff = load_contact_ebm(cebm_ckpt, dev)
    residue_type = pack_reader(pack_path)

    while True:
        item = task_q.get()
        if item is None:
            break
        label, chains_file, dump_dirs, out_dump_dir = item
        if os.path.exists(os.path.join(out_dump_dir, "samples.jsonl")):
            stems_q = sorted({l.split()[0] for l in open(chains_file) if l.strip()})
            n_got = sum(1 for l in open(os.path.join(out_dump_dir, "samples.jsonl")) if l.strip())
            if n_got == len(stems_q) * 3:
                print(f"[GPU {gpu_id}] SKIP ContactEBM {label} (already complete: {n_got} maps)", flush=True)
                continue

        t0_task = time.time()
        print(f"[GPU {gpu_id}] START ContactEBM {label} ({len(dump_dirs)} pools) -> {out_dump_dir}", flush=True)
        rows = defaultdict(list)
        for d_idx, d in enumerate(dump_dirs):
            for ln in open(os.path.join(d, "samples.jsonl")):
                if ln.strip():
                    r = json.loads(ln)
                    rows[r["stem"]].append((d_idx, r, d))

        stems = sorted({l.split()[0] for l in open(chains_file) if l.strip()})
        os.makedirs(out_dump_dir, exist_ok=True)
        spath = os.path.join(out_dump_dir, "samples.jsonl")
        if os.path.exists(spath):
            os.remove(spath)
        mpath = os.path.join(out_dump_dir, "cebm.jsonl")
        if os.path.exists(mpath):
            os.remove(mpath)

        for stem in stems:
            got = sorted(rows[stem], key=lambda x: (x[0], x[1]["sample_index"]))
            maps, labels, ref_ids, gt, L_c = [], [], set(), None, None
            for d_idx, r, d in got:
                z = np.load(os.path.join(d, r["file"]))
                L_c = int(z["L"]) if L_c is None else L_c
                assert int(z["L"]) == L_c
                maps.append(z["contact_prob"].astype(np.float32))
                labels.append(f"src{d_idx}#{r['sample_index']}")
                ref_ids.add(r["ref_id"])
                if gt is None:
                    gt = z["contact_gt"]
            ref_id = sorted(ref_ids)[0]
            aat = residue_type(stem)
            assert int(aat.shape[0]) == L_c, f"{stem}: pack aatype {aat.shape[0]} != L {L_c}"
            pool = np.stack(maps)
            e25 = np.concatenate([
                score_clean(cebm_model, pool[i:i + 8], aat, L_c, T_EVAL, dev)
                for i in range(0, len(pool), 8)
            ])
            pick = int(np.argmin(e25))
            init_bmap = (pool[pick] >= 0.5).astype(np.uint8)
            res = cebm_reverse_sample_from_t0(
                cebm_model, cebm_diff, init_bmap, aat, L_c, t0=T0, K_corrector=0, dev=dev, t_eval=T_EVAL, seed=TRI_SEED
            )
            pred = [i for i, k in enumerate(res["step_kinds"]) if k == "predictor"]
            second_half = pred[len(pred) // 2:]
            min_e_idx = min(second_half, key=lambda i: res["step_e"][i])
            mine = res["step_maps"][min_e_idx].astype(np.float32)
            np.fill_diagonal(mine, 0.0)
            outs = (("minE", mine), ("pick", pool[pick]), ("final", res["final_binary"].astype(np.float32)))
            with open(spath, "a") as fh:
                for k, (tag, mp_arr) in enumerate(outs):
                    f = f"{stem}_s{k:02d}.npz"
                    np.savez_compressed(
                        os.path.join(out_dump_dir, f),
                        contact_prob=mp_arr,
                        contact_gt=gt,
                        L=np.int32(L_c),
                        stem=stem,
                        sample_index=np.int32(k),
                        ref_id=ref_id,
                        cebm_output=tag,
                    )
                    fh.write(
                        json.dumps({
                            "stem": stem,
                            "sample_index": k,
                            "L": L_c,
                            "ref_id": ref_id,
                            "file": f,
                            "metrics": got[pick][1].get("metrics", {}),
                            "cebm_output": tag,
                        })
                        + "\n"
                    )
            with open(mpath, "a") as fh:
                fh.write(
                    json.dumps({
                        "stem": stem,
                        "L": L_c,
                        "ref_id": ref_id,
                        "pick_idx": pick,
                        "pick_label": labels[pick],
                        "pick_e25": float(e25[pick]),
                        "minE_e25": float(res["step_e"][min_e_idx]),
                        "final_e25": float(res["final_e"]),
                    })
                    + "\n"
                )
        print(f"[GPU {gpu_id}] DONE ContactEBM {label} ({len(stems)} chains) in {time.time() - t0_task:.1f}s", flush=True)


def q_stageB_worker(gpu_id, task_q, ckpt, dataset, out_dir, usalign, python_bin):
    """Dynamic queue Stage-B c2c worker preserving exact stageB_c2c_from_maps.py RNG parity."""
    while True:
        item = task_q.get()
        if item is None:
            break
        stageB_worker_loop(gpu_id, [item], ckpt, dataset, out_dir, usalign, python_bin)


# ───────────────────────── Phase 4: 10-Target bad_afdb Benchmark ─────────────────────────
def _cebm_reverse_with_corrector(cebm_model, cebm_diff, init_binary_map, aatype, Lq, t0, K_corrector, dev, t_eval, seed):
    from contact_ebm.diffusion.schedules import symmetrize_matrix

    torch.manual_seed(seed)
    x0 = torch.from_numpy((init_binary_map >= 0.5).astype(np.int64)).to(dev)[None]
    aa = aatype[:Lq].long().to(dev)[None]
    rmask = torch.ones((1, Lq), dtype=torch.float32, device=dev)
    pair_mask = cebm_model._pair_mask(rmask) if hasattr(cebm_model, "_pair_mask") else _cebm_pair_mask(rmask)
    pm_b = pair_mask.bool()
    diag = torch.arange(Lq, device=dev)

    timesteps = int(getattr(cebm_diff, "timesteps", 50))
    i0 = 0
    for i in range(timesteps):
        _, t_grid = cebm_diff.get_sampling_grid(i, timesteps)
        if t_grid <= t0 + 1e-6:
            i0 = i
            break
    _, t_start = cebm_diff.get_sampling_grid(i0, timesteps)
    t_start_tensor = torch.full((1,), float(t_start), device=dev, dtype=torch.float32)

    with torch.no_grad():
        zt = cebm_diff.forward_sample(x0, t_start_tensor, pair_mask)
        zt = symmetrize_matrix(zt.float()).long()
        zt[:, diag, diag] = 0
        zt = zt * pm_b.long()

    sched_obj = getattr(cebm_diff, "noise_schedule", getattr(cebm_diff, "schedule", None))

    def _renoise(z_s, s_val, t_val):
        with torch.no_grad():
            s_t = torch.full((1,), float(s_val), device=dev)
            t_t = torch.full((1,), float(t_val), device=dev)
            alpha_t = sched_obj.alpha(t_t).view(1, 1, 1)
            alpha_s = sched_obj.alpha(s_t).view(1, 1, 1)
            keep_prob = (alpha_t / alpha_s.clamp(min=1e-6)).clamp(0.0, 1.0)
            keep = torch.bernoulli(keep_prob.expand_as(z_s)).bool()
            fresh = (
                torch.full_like(z_s, cebm_diff.mask_id)
                if hasattr(cebm_diff, "mask_id")
                else cebm_diff.prior_sample(z_s.shape, device=dev)
            )
            out = torch.where(keep, z_s, fresh)
            out = symmetrize_matrix(out.float()).long()
            out[:, diag, diag] = 0
            return out * pm_b.long()

    pred_e, pred_maps = [], []
    logits_d = None
    for i in range(i0, timesteps):
        s, t = cebm_diff.get_sampling_grid(i, timesteps)
        t_tensor = torch.full((1,), float(t), device=dev, dtype=torch.float32)
        if K_corrector > 0 and s > 0:
            for _ in range(K_corrector):
                with torch.enable_grad():
                    logits, _, _ = cebm_model.compute_outputs(cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False)
                logits_c = logits.detach()
                with torch.no_grad():
                    zc = cebm_diff.sample_step(zt, logits_c, s, t, pair_mask=pair_mask)
                    zt = _renoise(zc, s, t)
        with torch.enable_grad():
            logits, _, _ = cebm_model.compute_outputs(cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False)
        logits_d = logits.detach()
        b_clean = (torch.sigmoid(logits_d.float())[0, :Lq, :Lq].cpu().numpy() >= 0.5).astype(np.float32)
        np.fill_diagonal(b_clean, 0.0)
        with torch.no_grad():
            e_eval = float(
                cebm_model.score_energy(
                    torch.from_numpy(b_clean.astype(np.int64)).to(dev)[None], aa, rmask, t_eval=float(t_eval)
                )[0].item()
            )
            zt = cebm_diff.sample_step(zt, logits_d, s, t, pair_mask=pair_mask)
        pred_e.append(e_eval)
        pred_maps.append(b_clean)

    with torch.no_grad():
        z0 = (torch.sigmoid(logits_d.float()) >= 0.5).long() * pm_b.long()
        final_map = z0[0, :Lq, :Lq].detach().cpu().numpy().astype(np.float32)
        np.fill_diagonal(final_map, 0.0)
    second_half = list(range(len(pred_e) // 2, len(pred_e)))
    min_i = min(second_half, key=lambda idx: pred_e[idx])
    return pred_maps[min_i], final_map


def _build_mpnn_chunk9_ref(topo_tf, full_t0, amask_t0, aatype_t0, cm_tmpl0, Lq):
    Lt = int(amask_t0.shape[0])
    runs = chunk9_runs(Lt, chunk_size=9)
    he_contact, keep = sse_contact_reference(cm_tmpl0.float(), runs, keep_types=(DSSP_HELIX,))
    structural = sse_structural_pair_features(cm_tmpl0.float(), full_t0.float(), amask_t0.float(), runs, keep)
    ref = topo_tf.assemble_reference(runs, he_contact, structural, length=Lq, augment=False)
    cap_res = max(int(ref["topology_he_tokens"].numel()) * 9, 1)
    L_eff = min(Lt, cap_res)
    ref["template_backbone_coords"] = full_t0[:L_eff, list(MPNN_ATOM4_INDICES), :].float()
    ref["template_backbone_mask"] = amask_t0[:L_eff, list(MPNN_ATOM4_INDICES)].all(dim=-1).float()
    ref["template_aatype"] = aatype_t0[:L_eff].long().clamp(0, 20)
    ref["template_residue_idx"] = torch.arange(L_eff, dtype=torch.long)
    return ref


def _generate_mpnn_maps(model, cfg_exp, ref, aatype_q, Lq, k, seed, dev):
    vs = cfg_exp.validation_sampling
    topo_keys = model._topology_keys()
    topology = {kk: Proteina._stack_topology([ref[kk]] * k, key=kk).to(dev) for kk in topo_keys}
    mask_q = torch.ones((k, Lq), dtype=torch.bool, device=dev)
    res_q = aatype_q[:Lq].long().to(dev)[None].expand(k, Lq).clone()

    cath_code_indices = cath_code_indices_mask = None
    if cfg_exp.training.get("fold_cond", False):
        cath_code_dir = cfg_exp.model.nn.get("cath_code_dir")
        if cath_code_dir is not None:
            _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
            cath_code_indices = torch.zeros((k, 1, 3), device=dev, dtype=torch.long)
            cath_code_indices[:, 0, 0], cath_code_indices[:, 0, 1], cath_code_indices[:, 0, 2] = nC, nA, nT
            cath_code_indices_mask = torch.zeros((k, 1), device=dev, dtype=torch.bool)

    L.seed_everything(seed)
    with torch.no_grad():
        res = model.generate(
            nsamples=k,
            n=Lq,
            dt=float(vs.get("dt", 0.02)),
            self_cond=cfg_exp.training.self_cond,
            cath_code=None,
            cath_code_indices=cath_code_indices,
            cath_code_indices_mask=cath_code_indices_mask,
            residue_type=res_q,
            guidance_weight=1.0,
            autoguidance_ratio=0.0,
            dtype=torch.float32,
            schedule_mode=vs.get("schedule_mode", "uniform"),
            schedule_p=float(vs.get("schedule_p", 1.0)),
            sampling_mode=vs.get("sampling_mode", "sc"),
            sc_scale_noise=float(vs.get("sc_scale_noise", 0.45)),
            sc_scale_score=float(vs.get("sc_scale_score", 1.0)),
            gt_mode="us",
            gt_p=1.0,
            gt_clamp_val=None,
            mask=mask_q,
            x_motif=None,
            fixed_sequence_mask=None,
            fixed_structure_mask=None,
            topology=topology,
            zero_sin_pos_emb=bool(cfg_exp.training.get("zero_sin_pos_emb", False)),
        )
    return torch.sigmoid(res["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()


def _fold_and_tm(c2c, prob, aatype_q, Lq, c2c_seed, native_pdb, usalign, tmp_dir, tag, dev):
    mask = torch.ones((1, Lq), device=dev)
    aa = aatype_q[:Lq].long().to(dev)[None]
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aa, mask)
    contacts = torch.as_tensor(prob, dtype=torch.float32, device=dev)[None, :Lq, :Lq]
    with torch.no_grad():
        s, z, _ = c2c.model.encode(contacts, aa, mask)
        torch.manual_seed(c2c_seed)
        coords = c2c.model.rollout(s, z, mask, ref_feats, ref_pos, a2t, amask, ruid, n_steps=C2C_INFERENCE_STEPS)
    c14 = coords.reshape(-1, Lq, 14, 3)[0, :Lq, :, :].float().cpu().numpy()
    pred_pdb = os.path.join(tmp_dir, f"{tag}.pdb")
    write_atom14_pdb(pred_pdb, c14, aatype_q[:Lq], torch.ones(Lq))
    r = subprocess.run([usalign, native_pdb, pred_pdb, "-TMscore", "5", "-outfmt", "2"], capture_output=True, text=True, timeout=180)
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) >= 7 and not ln.startswith("#"):
            return float(p[2])
    raise RuntimeError(f"USalign failed on {pred_pdb}: {r.stdout[:200]} {r.stderr[:200]}")


def _contact_stats(prob_maps, cm_nat):
    L_c = int(cm_nat.shape[0])
    iu = np.triu_indices(L_c, k=6)
    g = (np.asarray(cm_nat, dtype=np.float64)[iu[0], iu[1]] >= 0.5).astype(np.float64)
    pls, f1s = [], []
    for p_mat in prob_maps:
        p = np.asarray(p_mat, dtype=np.float64)[iu[0], iu[1]]
        b = (p >= 0.5).astype(np.float64)
        tp = float((b * g).sum())
        fp = float((b * (1.0 - g)).sum())
        fn = float(((1.0 - b) * g).sum())
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1s.append(2.0 * prec * rec / max(prec + rec, 1e-8))
        k_top = min(L_c, len(p))
        top_idx = np.argpartition(-p, k_top - 1)[:k_top] if k_top > 0 else np.array([], dtype=int)
        pls.append(float(g[top_idx].mean()) if k_top > 0 else 0.0)
    return float(np.mean(pls)), float(np.mean(f1s))


def q_bad_afdb_worker(gpu_id, task_q, bundle_path, config_name, ckpt_189k, ckpt_219k, c2c_ckpt, cebm_repo, cebm_ckpt, usalign, out_dir):
    os.environ["SIGMA_DATA"] = "16.0"
    assert SIGMA_DATA == 16.0, f"SIGMA_DATA={SIGMA_DATA}"
    torch.cuda.set_device(gpu_id)
    dev = torch.device(f"cuda:{gpu_id}")
    if cebm_repo not in sys.path:
        sys.path.insert(0, cebm_repo)

    from gen_c2c_structures import MODEL_CFG

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

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
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    topo_tf = [t for t in dm.transform.transforms if type(t).__name__ == "TopologyReferenceTransform"][0]
    topo_tf._ensure_loaded()

    store_189 = os.path.join(out_dir, f".store_bafdb189_gpu{gpu_id}")
    store_219 = os.path.join(out_dir, f".store_bafdb219_gpu{gpu_id}")
    os.makedirs(store_189, exist_ok=True)
    os.makedirs(store_219, exist_ok=True)

    model_189 = Proteina(cfg_exp, store_dir=store_189)
    sd189 = torch.load(ckpt_189k, map_location="cpu", weights_only=False)
    model_189.load_state_dict(sd189["state_dict"] if "state_dict" in sd189 else sd189, strict=True)
    model_189 = model_189.to(dev).eval()

    model_219 = Proteina(cfg_exp, store_dir=store_219)
    sd219 = torch.load(ckpt_219k, map_location="cpu", weights_only=False)
    model_219.load_state_dict(sd219["state_dict"] if "state_dict" in sd219 else sd219, strict=True)
    model_219 = model_219.to(dev).eval()

    c2c = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    ck_c2c = torch.load(c2c_ckpt, map_location="cpu", weights_only=False)
    c2c.model.load_state_dict(ck_c2c["ema"]["params"], strict=True)
    del ck_c2c
    c2c = c2c.to(dev).eval()

    cebm_model, cebm_diff = load_contact_ebm(cebm_ckpt, dev)

    while True:
        item = task_q.get()
        if item is None:
            break
        arm, pid = item
        out_json = os.path.join(out_dir, "bad_afdb", arm, f"{pid}.json")
        if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
            print(f"[GPU {gpu_id}] SKIP bad_afdb {arm}/{pid}", flush=True)
            continue

        t0_p = time.time()
        tdata = bundle["targets"][pid]
        adata = tdata["arms"][arm]
        Lq = int(tdata["Lq"])
        ci = int(tdata["ci"])
        c2c_seed = C2C_SEED_BASE + ci
        aatype_q = tdata["aatype_q"].long()
        cm_nat = tdata["cm_nat"].float().numpy()

        ref = _build_mpnn_chunk9_ref(
            topo_tf,
            adata["full_t0"],
            adata["amask_t0"],
            adata["aatype_t0"],
            adata["cm_tmpl0"],
            Lq,
        )
        prob_189 = _generate_mpnn_maps(model_189, cfg_exp, ref, aatype_q, Lq, K_SAMPLES, TRI_SEED, dev)
        prob_219 = _generate_mpnn_maps(model_219, cfg_exp, ref, aatype_q, Lq, K_SAMPLES, TRI_SEED, dev)
        pl_189, f1_189 = _contact_stats(prob_189, cm_nat)
        pl_219, f1_219 = _contact_stats(prob_219, cm_nat)

        e_189 = score_clean(cebm_model, prob_189, aatype_q, Lq, T_EVAL, dev)
        e_219 = score_clean(cebm_model, prob_219, aatype_q, Lq, T_EVAL, dev)
        pool16 = np.concatenate([prob_189, prob_219], axis=0)
        e_pool16 = np.concatenate([e_189, e_219], axis=0)

        with tempfile.TemporaryDirectory(prefix=f"bafdb_{arm}_{pid}_") as td:
            nat_pdb = os.path.join(td, "native.pdb")
            with open(nat_pdb, "w") as fh:
                fh.write(tdata["native_pdb_text"])

            tms_189 = [_fold_and_tm(c2c, prob_189[i], aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, f"s189_{i}", dev) for i in range(K_SAMPLES)]
            tms_219 = [_fold_and_tm(c2c, prob_219[i], aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, f"s219_{i}", dev) for i in range(K_SAMPLES)]
            tms_pool16 = tms_189 + tms_219

            pick_189 = int(np.argmin(e_189))
            pick_219 = int(np.argmin(e_219))
            pick_p16 = int(np.argmin(e_pool16))

            mine_189_map, fin_189_map = _cebm_reverse_with_corrector(
                cebm_model, cebm_diff, (prob_189[pick_189] >= 0.5).astype(np.uint8), aatype_q, Lq, 0.3, 0, dev, T_EVAL, TRI_SEED
            )
            mine_219_map, fin_219_map = _cebm_reverse_with_corrector(
                cebm_model, cebm_diff, (prob_219[pick_219] >= 0.5).astype(np.uint8), aatype_q, Lq, 0.3, 0, dev, T_EVAL, TRI_SEED
            )
            mine_p16_map, fin_p16_map = _cebm_reverse_with_corrector(
                cebm_model, cebm_diff, (pool16[pick_p16] >= 0.5).astype(np.uint8), aatype_q, Lq, 0.3, 0, dev, T_EVAL, TRI_SEED
            )

            tm_189_minE = _fold_and_tm(c2c, mine_189_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "s189_minE", dev)
            tm_189_fin = _fold_and_tm(c2c, fin_189_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "s189_fin", dev)
            tm_219_minE = _fold_and_tm(c2c, mine_219_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "s219_minE", dev)
            tm_219_fin = _fold_and_tm(c2c, fin_219_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "s219_fin", dev)
            tm_p16_minE = _fold_and_tm(c2c, mine_p16_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "p16_minE", dev)
            tm_p16_fin = _fold_and_tm(c2c, fin_p16_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "p16_fin", dev)

            extra = {}
            if arm == "tmpl":
                mine_t50_map, fin_t50_map = _cebm_reverse_with_corrector(
                    cebm_model, cebm_diff, (pool16[pick_p16] >= 0.5).astype(np.uint8), aatype_q, Lq, 0.5, 20, dev, T_EVAL, TRI_SEED
                )
                extra["pool16_cebm_t050_K20_minE"] = _fold_and_tm(c2c, mine_t50_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "p16_t50_minE", dev)
                extra["pool16_cebm_t050_K20_final"] = _fold_and_tm(c2c, fin_t50_map, aatype_q, Lq, c2c_seed, nat_pdb, usalign, td, "p16_t50_fin", dev)

        rec = {
            "pid": pid,
            "arm": arm,
            "Lq": Lq,
            "ref_stem": adata["ref_stem"],
            "n_chunks": int(ref["topology_he_tokens"].numel()),
            "pl_s189k": pl_189,
            "f1_s189k": f1_189,
            "pl_s219k": pl_219,
            "f1_s219k": f1_219,
            "s189k_mean8": float(np.mean(tms_189)),
            "s189k_best8": float(np.max(tms_189)),
            "s189k_cebm_pick": float(tms_189[pick_189]),
            "s189k_cebm_minE": float(tm_189_minE),
            "s189k_cebm_final": float(tm_189_fin),
            "s219k_mean8": float(np.mean(tms_219)),
            "s219k_best8": float(np.max(tms_219)),
            "s219k_cebm_pick": float(tms_219[pick_219]),
            "s219k_cebm_minE": float(tm_219_minE),
            "s219k_cebm_final": float(tm_219_fin),
            "pool16_mean": float(np.mean(tms_pool16)),
            "pool16_best16": float(np.max(tms_pool16)),
            "pool16_cebm_pick": float(tms_pool16[pick_p16]),
            "pool16_cebm_minE": float(tm_p16_minE),
            "pool16_cebm_final": float(tm_p16_fin),
            "tms_189": [float(x) for x in tms_189],
            "tms_219": [float(x) for x in tms_219],
            "baselines": adata.get("baselines", {}),
            **extra,
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        tmp_json = out_json + ".tmp"
        with open(tmp_json, "w") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(tmp_json, out_json)
        print(
            f"[GPU {gpu_id}] DONE bad_afdb {arm}/{pid} (L={Lq}) in {time.time() - t0_p:.1f}s | "
            f"s189k={rec['s189k_mean8']:.4f} (minE {rec['s189k_cebm_minE']:.4f}) | "
            f"s219k={rec['s219k_mean8']:.4f} (minE {rec['s219k_cebm_minE']:.4f}) | "
            f"pool16_minE={rec['pool16_cebm_minE']:.4f}",
            flush=True,
        )


# ───────────────────────── Comprehensive Summary Report ─────────────────────────
def summarize_s219k(out_dir, bad_afdb_bundle=None):
    lines = ["=== MPNN_CHUNK9 S75K vs S160K vs S189K vs S208K vs S219K & CONTACTEBM + BAD_AFDB REPORT ==="]

    # 1. Stage-A comparison across t7, equal, self
    stageA_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tag, prefix in (
        ("s75k", "mpnn"),
        ("s160k", "mpnn_s160k"),
        ("s189k", "mpnn_s189k"),
        ("s208k", "mpnn_s208k"),
        ("s219k", "mpnn_s219k"),
    ):
        for arm, n_c in (("t7", 4), ("equal", 8), ("self", 8)):
            for c in range(n_c):
                d = os.path.join(out_dir, "dumps", f"{prefix}_{arm}_c{c}")
                sp = os.path.join(d, "samples.jsonl")
                if not os.path.exists(sp):
                    continue
                for ln in open(sp):
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    st = r["stem"]
                    m = r["metrics"]
                    stageA_data[arm][tag][st].append(float(m["contact_precision_at_L"]))
                    stageA_data[f"{arm}_f1"][tag][st].append(float(m.get("contact_f1", 0.0)))
                    stageA_data[f"{arm}_pl5"][tag][st].append(float(m.get("contact_precision_at_L5", 0.0)))
                    stageA_data[f"{arm}_lr_pl5"][tag][st].append(float(m.get("contact_long_range_precision_at_L5", 0.0)))

    for arm in ("t7", "equal", "self"):
        if "s75k" not in stageA_data[arm] or "s219k" not in stageA_data[arm]:
            continue
        stems_all = sorted(
            set(stageA_data[arm]["s75k"].keys())
            & set(stageA_data[arm].get("s189k", {}).keys())
            & set(stageA_data[arm]["s219k"].keys())
        )
        if not stems_all:
            continue
        subsets = [("all", stems_all)]
        if arm == "t7":
            clean = [s for s in stems_all if s not in T7_CHANGED_CHAINS]
            subsets = [("clean_70", clean), ("all_98", stems_all)]
        for sub_name, S in subsets:
            lines.append(f"\n--- Stage-A Contact Metrics: {arm} ({sub_name}, n={len(S)} chains) ---")
            for mkey, mlabel in (("", "P@L"), ("_pl5", "P@L/5"), ("_lr_pl5", "LR_P@L/5"), ("_f1", "F1")):
                rng = np.random.default_rng(0)
                a75 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s75k"][s]) for s in S])
                a160 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s160k"][s]) for s in S])
                a189 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s189k"][s]) for s in S])
                a219 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s219k"][s]) for s in S])
                s208_str = ""
                if "s208k" in stageA_data[arm] and all(s in stageA_data[f"{arm}{mkey}"]["s208k"] for s in S):
                    a208 = np.array([np.mean(stageA_data[f"{arm}{mkey}"]["s208k"][s]) for s in S])
                    s208_str = f" | s208k: {np.mean(a208):.4f} (med {np.median(a208):.4f})"
                mu219, lo219, hi219 = boot_ci(a219, rng)
                d_219_75 = a219 - a75
                mud75, lod75, hid75 = boot_ci(d_219_75, rng)
                d_219_189 = a219 - a189
                mud189, lod189, hid189 = boot_ci(d_219_189, rng)
                lines.append(
                    f"  {mlabel:8s} | s75k: {np.mean(a75):.4f} | s160k: {np.mean(a160):.4f} | s189k: {np.mean(a189):.4f}"
                    f"{s208_str} | s219k: {mu219:.4f} [{lo219:.4f}, {hi219:.4f}] (med {np.median(a219):.4f}) "
                    f"| s219k-s75k: {mud75:+.4f} [{lod75:+.4f}, {hid75:+.4f}] ({int((d_219_75>0).sum())}/{len(S)}) "
                    f"| s219k-s189k: {mud189:+.4f} [{lod189:+.4f}, {hid189:+.4f}] ({int((d_219_189>0).sum())}/{len(S)})"
                )

    # 2. Stage-B c2c TM comparison on T7-NEWEST
    sb_dir = os.path.join(out_dir, "stageB")
    tm_by_arm = defaultdict(lambda: defaultdict(list))
    for tag, prefix in (
        ("s75k", "mpnn_t7"),
        ("s160k", "mpnn_s160k_t7"),
        ("s189k", "mpnn_s189k_t7"),
        ("s208k", "mpnn_s208k_t7"),
        ("s219k", "mpnn_s219k_t7"),
    ):
        for c in range(4):
            p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip():
                        r = json.loads(ln)
                        tm_by_arm[tag][r["stem"]].append(float(r["tm"]))

    for pool_name, tags in (
        ("pool16_160_189_mean", ("s160k", "s189k")),
        ("pool16_189_219_mean", ("s189k", "s219k")),
        ("pool24_160_189_219_mean", ("s160k", "s189k", "s219k")),
        ("pool24_189_208_219_mean", ("s189k", "s208k", "s219k")),
        ("pool32_160_189_208_219_mean", ("s160k", "s189k", "s208k", "s219k")),
    ):
        if all(t in tm_by_arm for t in tags):
            common = set.intersection(*(set(tm_by_arm[t].keys()) for t in tags))
            for st in common:
                if all(len(tm_by_arm[t][st]) == 8 for t in tags):
                    tm_by_arm[pool_name][st] = sum((tm_by_arm[t][st] for t in tags), [])

    for cebm_tag, prefix in (
        ("s75k", "mpnn_cebm_t7"),
        ("s160k", "mpnn_s160k_cebm_t7"),
        ("s189k", "mpnn_s189k_cebm_t7"),
        ("s208k", "mpnn_s208k_cebm_t7"),
        ("s219k", "mpnn_s219k_cebm_t7"),
        ("pool16_160_189", "mpnn_pool16_160_189_cebm_t7"),
        ("pool16_189_219", "mpnn_pool16_189_219_cebm_t7"),
        ("pool24_160_189_219", "mpnn_pool24_160_189_219_cebm_t7"),
        ("pool24_189_208_219", "mpnn_pool24_189_208_219_cebm_t7"),
        ("pool32_160_189_208_219", "mpnn_pool32_160_189_208_219_cebm_t7"),
    ):
        for c in range(4):
            p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
            if os.path.exists(p):
                for ln in open(p):
                    if ln.strip():
                        r = json.loads(ln)
                        sub = CEBM_TAGS[int(r["sample_index"])]
                        tm_by_arm[f"{cebm_tag}_{sub}"][r["stem"]].append(float(r["tm"]))

    t7_all = sorted(tm_by_arm["s75k"].keys())
    if t7_all:
        t7_clean = [s for s in t7_all if s not in T7_CHANGED_CHAINS]
        for sub_name, S in (("clean_70", t7_clean), ("all_98", t7_all)):
            lines.append(f"\n=== T7-NEWEST Stage-B c2c 3D Structure TM-Score ({sub_name}, n={len(S)} chains) ===")
            lines.append(f"{'arm':<34} {'samples':>8} {'TM>=0.5':>9} {'chains>=0.5':>13}   mean TM [95% CI]          median   best_of_K")
            means = {}
            for arm_name in (
                "s75k",
                "s160k",
                "s189k",
                "s208k",
                "s219k",
                "pool16_189_219_mean",
                "pool24_189_208_219_mean",
                "pool32_160_189_208_219_mean",
                "s75k_cebm_pick",
                "s160k_cebm_pick",
                "s160k_cebm_minE",
                "s189k_cebm_pick",
                "s189k_cebm_minE",
                "s208k_cebm_pick",
                "s208k_cebm_minE",
                "s219k_cebm_pick",
                "s219k_cebm_minE",
                "s219k_cebm_final",
                "pool16_160_189_cebm_pick",
                "pool16_160_189_cebm_minE",
                "pool16_189_219_cebm_pick",
                "pool16_189_219_cebm_minE",
                "pool16_189_219_cebm_final",
                "pool24_160_189_219_cebm_pick",
                "pool24_160_189_219_cebm_minE",
                "pool24_189_208_219_cebm_pick",
                "pool24_189_208_219_cebm_minE",
                "pool24_189_208_219_cebm_final",
                "pool32_160_189_208_219_cebm_pick",
                "pool32_160_189_208_219_cebm_minE",
                "pool32_160_189_208_219_cebm_final",
            ):
                if arm_name not in tm_by_arm or any(s not in tm_by_arm[arm_name] for s in S):
                    continue
                rng = np.random.default_rng(0)
                m = np.array([np.mean(tm_by_arm[arm_name][s]) for s in S])
                b = np.array([np.max(tm_by_arm[arm_name][s]) for s in S])
                means[arm_name] = m
                allv = np.concatenate([tm_by_arm[arm_name][s] for s in S])
                mu, lo, hi = boot_ci(m, rng)
                lines.append(
                    f"{arm_name:<34} {len(allv):>8} {int((allv >= 0.5).sum()):>9} {int((m >= 0.5).sum()):>13}   "
                    f"{mu:.4f} [{lo:.4f}, {hi:.4f}]   {np.median(m):.4f}   {np.mean(b):.4f} ({int((b>=0.5).sum())}/{len(S)})"
                )
            lines.append("paired difference B - A (per chain)                     mean [95% CI]            chains B>A / n")
            for b_, a_ in (
                ("s208k", "s75k"),
                ("s219k", "s75k"),
                ("s219k", "s160k"),
                ("s219k", "s189k"),
                ("s219k_cebm_pick", "s219k"),
                ("s219k_cebm_minE", "s219k"),
                ("s219k_cebm_pick", "s75k"),
                ("s219k_cebm_minE", "s75k"),
                ("s219k_cebm_pick", "s189k_cebm_pick"),
                ("s219k_cebm_minE", "s189k_cebm_minE"),
                ("pool16_189_219_cebm_minE", "s75k"),
                ("pool16_189_219_cebm_minE", "pool16_160_189_cebm_minE"),
                ("pool24_189_208_219_cebm_minE", "s75k"),
                ("pool24_189_208_219_cebm_minE", "pool16_160_189_cebm_minE"),
                ("pool32_160_189_208_219_cebm_minE", "s75k"),
                ("pool32_160_189_208_219_cebm_minE", "pool16_160_189_cebm_minE"),
            ):
                if b_ in means and a_ in means:
                    rng = np.random.default_rng(0)
                    d = means[b_] - means[a_]
                    mu, lo, hi = boot_ci(d, rng)
                    lines.append(
                        f"  {b_:>32} - {a_:<26} {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]   {int((d > 0).sum())}/{len(d)}"
                    )

    # 3. toporef_xrun Stage-B c2c TM comparison (equal and self: s75k, s160k, s189k, s219k, pool16_189_219)
    for xarm in ("equal", "self"):
        x_tm = defaultdict(lambda: defaultdict(list))
        for tag, prefix in (
            ("s75k", f"mpnn_{xarm}"),
            ("s160k", f"mpnn_s160k_{xarm}"),
            ("s189k", f"mpnn_s189k_{xarm}"),
            ("s219k", f"mpnn_s219k_{xarm}"),
        ):
            for c in range(8):
                p = os.path.join(sb_dir, f"{prefix}_c{c}.jsonl")
                if os.path.exists(p):
                    for ln in open(p):
                        if ln.strip():
                            r = json.loads(ln)
                            x_tm[tag][r["stem"]].append(float(r["tm"]))
        if not x_tm["s75k"]:
            continue
        S = sorted(x_tm["s75k"].keys())
        if all(s in x_tm["s189k"] and s in x_tm["s219k"] for s in S):
            for s in S:
                x_tm["pool16_189_219"][s] = x_tm["s189k"][s] + x_tm["s219k"][s]
        lines.append(f"\n=== Held-Out Eval Set (toporef_xrun) Stage-B c2c TM: {xarm} (n={len(S)} chains) ===")
        means = {}
        for tag in ("s75k", "s160k", "s189k", "s219k", "pool16_189_219"):
            if any(s not in x_tm[tag] for s in S):
                continue
            rng = np.random.default_rng(0)
            m = np.array([np.mean(x_tm[tag][s]) for s in S])
            b = np.array([np.max(x_tm[tag][s]) for s in S])
            means[tag] = m
            allv = np.concatenate([x_tm[tag][s] for s in S])
            mu, lo, hi = boot_ci(m, rng)
            lines.append(
                f"  {tag:<16} samples={len(allv):4d}  TM>=0.5={int((allv>=0.5).sum()):4d}  "
                f"chains>=0.5={int((m>=0.5).sum()):2d}/{len(S)}  mean TM={mu:.4f} [{lo:.4f}, {hi:.4f}]  "
                f"med={np.median(m):.4f}  bestK={np.mean(b):.4f}"
            )
        for b_, a_ in (("s189k", "s75k"), ("s219k", "s75k"), ("s219k", "s160k"), ("s219k", "s189k")):
            if b_ in means and a_ in means:
                rng = np.random.default_rng(0)
                d = means[b_] - means[a_]
                mu, lo, hi = boot_ci(d, rng)
                lines.append(f"  paired diff {b_:<8} - {a_:<8}: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  wins={int((d>0).sum())}/{len(d)}")

    # 4. 10-Target bad_afdb Benchmark Summary (mpnn_chunk9 vs SSE cfft / tmalign_smooth / sse_ens24)
    bafdb_dir = os.path.join(out_dir, "bad_afdb")
    if os.path.isdir(bafdb_dir):
        for arm in ("lad_r250", "tmpl", "self"):
            adir = os.path.join(bafdb_dir, arm)
            if not os.path.isdir(adir):
                continue
            jfiles = sorted(glob.glob(os.path.join(adir, "*.json")))
            if not jfiles:
                continue
            recs = [json.load(open(jf)) for jf in jfiles]
            lines.append(f"\n=== 10-Target bad_afdb Benchmark ({arm}, n={len(recs)} targets) ===")
            lines.append(
                f"{'pid':<9} {'Lq':>4} | {'cfft_hb8':>8} {'cfft_tmS':>8} {'hb_pick':>8} | "
                f"{'s189_m8':>8} {'s219_m8':>8} {'s219_pk':>8} {'s219_mnE':>8} {'p16_mnE':>8} {'p16_b16':>8}"
            )
            for r in recs:
                bs = r.get("baselines", {})
                hb8 = bs.get("h_bond", {}).get("mean8_tm", float("nan"))
                tms = bs.get("tmalign_smooth", {}).get("mean8_tm", float("nan"))
                hb_pk = bs.get("h_bond", {}).get("cebm_pick_tm", float("nan"))
                lines.append(
                    f"{r['pid']:<9} {r['Lq']:>4d} | {hb8:>8.4f} {tms:>8.4f} {hb_pk:>8.4f} | "
                    f"{r['s189k_mean8']:>8.4f} {r['s219k_mean8']:>8.4f} {r['s219k_cebm_pick']:>8.4f} "
                    f"{r['s219k_cebm_minE']:>8.4f} {r['pool16_cebm_minE']:>8.4f} {r['pool16_best16']:>8.4f}"
                )
            metrics_to_avg = [
                ("cfft_h_bond_mean8", [r.get("baselines", {}).get("h_bond", {}).get("mean8_tm", np.nan) for r in recs]),
                ("cfft_h_bond_cebm_pick", [r.get("baselines", {}).get("h_bond", {}).get("cebm_pick_tm", np.nan) for r in recs]),
                ("cfft_tmalign_mean8", [r.get("baselines", {}).get("tmalign", {}).get("mean8_tm", np.nan) for r in recs]),
                ("cfft_tmalign_cebm_pick", [r.get("baselines", {}).get("tmalign", {}).get("cebm_pick_tm", np.nan) for r in recs]),
                ("cfft_tmalign_smooth_mean8", [r.get("baselines", {}).get("tmalign_smooth", {}).get("mean8_tm", np.nan) for r in recs]),
                ("cfft_tmalign_smooth_pick", [r.get("baselines", {}).get("tmalign_smooth", {}).get("cebm_pick_tm", np.nan) for r in recs]),
                ("mpnn_s189k_mean8", [r["s189k_mean8"] for r in recs]),
                ("mpnn_s189k_cebm_pick", [r["s189k_cebm_pick"] for r in recs]),
                ("mpnn_s189k_cebm_minE", [r["s189k_cebm_minE"] for r in recs]),
                ("mpnn_s219k_mean8", [r["s219k_mean8"] for r in recs]),
                ("mpnn_s219k_best8", [r["s219k_best8"] for r in recs]),
                ("mpnn_s219k_cebm_pick", [r["s219k_cebm_pick"] for r in recs]),
                ("mpnn_s219k_cebm_minE", [r["s219k_cebm_minE"] for r in recs]),
                ("mpnn_pool16_mean", [r["pool16_mean"] for r in recs]),
                ("mpnn_pool16_best16", [r["pool16_best16"] for r in recs]),
                ("mpnn_pool16_cebm_pick", [r["pool16_cebm_pick"] for r in recs]),
                ("mpnn_pool16_cebm_minE", [r["pool16_cebm_minE"] for r in recs]),
                ("mpnn_pool16_cebm_final", [r["pool16_cebm_final"] for r in recs]),
            ]
            if arm == "tmpl" and all("pool16_cebm_t050_K20_minE" in r for r in recs):
                metrics_to_avg.append(("mpnn_pool16_t050_K20_minE", [r["pool16_cebm_t050_K20_minE"] for r in recs]))
                metrics_to_avg.append(("mpnn_pool16_t050_K20_final", [r["pool16_cebm_t050_K20_final"] for r in recs]))
            lines.append("  Aggregate 10-target means:")
            for mname, vals in metrics_to_avg:
                arr = np.asarray(vals, dtype=float)
                if np.all(np.isfinite(arr)):
                    rng = np.random.default_rng(0)
                    mu, lo, hi = boot_ci(arr, rng)
                    lines.append(
                        f"    {mname:<28}: mean={mu:.4f} [{lo:.4f}, {hi:.4f}]  med={np.median(arr):.4f}  >=0.5={int((arr>=0.5).sum())}/{len(arr)}"
                    )

    text = "\n".join(lines)
    print(text, flush=True)
    with open(os.path.join(out_dir, "s219k_and_cebm_report.txt"), "w") as fh:
        fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/jupyter-chenxi/mpnn_chunk9_eval")
    ap.add_argument("--config_name", default="training_contact_tri_v2_confind_mpnn_chunk9")
    ap.add_argument(
        "--ckpt_189k",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_f1_epoch=00000468_step=000000189375-EMA.ckpt",
    )
    ap.add_argument(
        "--ckpt_208k",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_f1_epoch=00000514_step=000000207842-EMA.ckpt",
    )
    ap.add_argument(
        "--ckpt_219k",
        default="/home/jupyter-chenxi/proteina_mpnn_chunk9/store/tri_v2_confind_mpnn_chunk9_seqmask25_tmsmooth/checkpoints/chk_best_contact_f1_epoch=00000541_step=000000218852-EMA.ckpt",
    )
    ap.add_argument("--c2c_ckpt", default="/home/jupyter-chenxi/mpnn_chunk9_eval/ckpt/final_twin_s29000_ema_only.ckpt")
    ap.add_argument("--c2c_dataset", default="pdb_train_contact-confind_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--pack_path", default="/home/jupyter-chenxi/sse1d_data/orcd/compute/so3/002/chenxi/proteina_pack/tri_v2.pack")
    ap.add_argument("--cebm_repo", default="/home/jupyter-chenxi/ContactEBM")
    ap.add_argument("--cebm_ckpt", default="/home/jupyter-chenxi/contactebm_runs/confind_a1_prod100/epoch82.pt")
    ap.add_argument("--usalign", default="/home/jupyter-chenxi/.local/bin/USalign")
    ap.add_argument("--python_bin", default="/home/jupyter-chenxi/miniconda3/envs/proteina_sse1d/bin/python")
    ap.add_argument("--bad_afdb_raw", default="/home/jupyter-chenxi/mpnn_chunk9_eval/bad_afdb_10targets_raw.json")
    ap.add_argument("--bad_afdb_bundle", default="/home/jupyter-chenxi/mpnn_chunk9_eval/bad_afdb_10targets_inputs.pt")
    ap.add_argument("--f2c_ckpt", default="/home/jupyter-chenxi/contactebm_data/f2c/best.pt")
    ap.add_argument("--ngpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        summarize_s219k(args.out_dir, args.bad_afdb_bundle)
        return

    prepare_inputs(args.out_dir)

    # Phase 1a: Dynamic-queue Stage-A for s219k across T7-NEWEST (4 chunks) + toporef_xrun (16 chunks)
    print("=== PHASE 1A: Dynamic-Queue Stage-A for s219k (t7 + equal + self, 20 chunks) ===", flush=True)
    s219k_stageA_tasks = (
        [("t7", k) for k in range(4)]
        + [("equal", k) for k in range(8)]
        + [("self", k) for k in range(8)]
    )
    run_queue_workers(
        q_stageA_worker,
        s219k_stageA_tasks,
        args.ngpus,
        extra_args=(args.config_name, args.ckpt_219k, args.out_dir, args.seed, "mpnn_s219k"),
    )

    # Phase 1b: Dynamic-queue Stage-A for s208k on T7-NEWEST (4 chunks)
    if os.path.exists(args.ckpt_208k):
        print("=== PHASE 1B: Dynamic-Queue Stage-A for s208k on T7-NEWEST (4 chunks) ===", flush=True)
        run_queue_workers(
            q_stageA_worker,
            [("t7", k) for k in range(4)],
            args.ngpus,
            extra_args=(args.config_name, args.ckpt_208k, args.out_dir, args.seed, "mpnn_s208k"),
        )

    # Phase 2: Dynamic-queue ContactEBM on T7-NEWEST (largest pools first)
    print("=== PHASE 2: Dynamic-Queue ContactEBM on T7-NEWEST ===", flush=True)
    cebm_tasks = []
    c2c_cebm_tasks = []
    has_208 = os.path.exists(args.ckpt_208k)
    pool_specs = []
    if has_208:
        pool_specs.append(("mpnn_pool32_160_189_208_219_cebm_t7", ("mpnn_s160k", "mpnn_s189k", "mpnn_s208k", "mpnn_s219k")))
        pool_specs.append(("mpnn_pool24_189_208_219_cebm_t7", ("mpnn_s189k", "mpnn_s208k", "mpnn_s219k")))
    pool_specs.append(("mpnn_pool24_160_189_219_cebm_t7", ("mpnn_s160k", "mpnn_s189k", "mpnn_s219k")))
    pool_specs.append(("mpnn_pool16_189_219_cebm_t7", ("mpnn_s189k", "mpnn_s219k")))
    pool_specs.append(("mpnn_s219k_cebm_t7", ("mpnn_s219k",)))
    if has_208:
        pool_specs.append(("mpnn_s208k_cebm_t7", ("mpnn_s208k",)))

    for out_pref, src_prefs in pool_specs:
        for k in range(4):
            q_file = os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt")
            src_dirs = [os.path.join(args.out_dir, "dumps", f"{sp}_t7_c{k}") for sp in src_prefs]
            out_d = os.path.join(args.out_dir, "dumps", f"{out_pref}_c{k}")
            cebm_tasks.append((f"{out_pref}_c{k}", q_file, src_dirs, out_d))
            c2c_cebm_tasks.append((f"{out_pref}_c{k}", q_file, out_d))

    run_queue_workers(
        q_cebm_worker,
        cebm_tasks,
        args.ngpus,
        extra_args=(args.pack_path, args.cebm_repo, args.cebm_ckpt),
    )

    # Phase 3: Dynamic-queue Stage-B c2c folding (8-sample chunks first, then 3-sample cebm chunks)
    print("=== PHASE 3: Dynamic-Queue Stage-B c2c Folding ===", flush=True)
    stageB_tasks = []
    for k in range(4):
        q_file = os.path.join(args.out_dir, "chunks", f"t7_c{k}_q.txt")
        stageB_tasks.append((f"mpnn_s219k_t7_c{k}", q_file, os.path.join(args.out_dir, "dumps", f"mpnn_s219k_t7_c{k}")))
        if has_208:
            stageB_tasks.append((f"mpnn_s208k_t7_c{k}", q_file, os.path.join(args.out_dir, "dumps", f"mpnn_s208k_t7_c{k}")))
    for arm in ("equal", "self"):
        for k in range(8):
            q_file = os.path.join(args.out_dir, "chunks", f"{arm}_c{k}_q.txt")
            stageB_tasks.append((f"mpnn_s219k_{arm}_c{k}", q_file, os.path.join(args.out_dir, "dumps", f"mpnn_s219k_{arm}_c{k}")))
    stageB_tasks.extend(c2c_cebm_tasks)

    run_queue_workers(
        q_stageB_worker,
        stageB_tasks,
        args.ngpus,
        extra_args=(args.c2c_ckpt, args.c2c_dataset, args.out_dir, args.usalign, args.python_bin),
    )

    # Phase 4: 10-target bad_afdb benchmark (if raw json or bundle staged)
    if args.bad_afdb_raw and os.path.exists(args.bad_afdb_raw) and not os.path.exists(args.bad_afdb_bundle):
        ctx_b = mp.get_context("spawn")
        pb = ctx_b.Process(target=prepare_bad_afdb_bundle, args=(args.bad_afdb_raw, args.bad_afdb_bundle, args.f2c_ckpt))
        pb.start()
        pb.join()
        if pb.exitcode != 0:
            raise RuntimeError(f"prepare_bad_afdb_bundle exited with code {pb.exitcode}")
    if args.bad_afdb_bundle and os.path.exists(args.bad_afdb_bundle):
        print("=== PHASE 4: Dynamic-Queue 10-Target bad_afdb Benchmark (lad_r250, tmpl, self) ===", flush=True)
        bundle = torch.load(args.bad_afdb_bundle, map_location="cpu", weights_only=False)
        pids_by_len = sorted(bundle["selected_pids"], key=lambda p: -int(bundle["targets"][p]["Lq"]))
        bafdb_tasks = []
        # Put tmpl first (since tmpl also runs t0=0.5, K=20), then lad_r250, then self
        for arm in ("tmpl", "lad_r250", "self"):
            for pid in pids_by_len:
                bafdb_tasks.append((arm, pid))
        run_queue_workers(
            q_bad_afdb_worker,
            bafdb_tasks,
            args.ngpus,
            extra_args=(
                args.bad_afdb_bundle,
                args.config_name,
                args.ckpt_189k,
                args.ckpt_219k,
                args.c2c_ckpt,
                args.cebm_repo,
                args.cebm_ckpt,
                args.usalign,
                args.out_dir,
            ),
        )

    summarize_s219k(args.out_dir, args.bad_afdb_bundle)


if __name__ == "__main__":
    main()
