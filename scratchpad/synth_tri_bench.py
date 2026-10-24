"""Synthetic-template benchmark, stage 1: tri (ConFind FT, cfft) sampled on (query, template) pairs -> c2c structures.

Per pair of --pairs (one JSONL row: pid, template, mode, rewind, sample, qpath, tpath, qs, Lq, Lt):
  reference  the template PDB -> the topology reference EXACTLY as precompute_synthetic_topology_index built the
             training rows: parse_structure + PDB_TO_OPENFOLD reorder (atom37, missing atoms zero as `full[amask] = ...`),
             DSSP as the template-row call, Frame2ConFind via _f2c_inputs (prob >= CONFIND_THRESHOLD), topology_row,
             the index's byte encoding decoded like _he_contact_for / _he_structural_for, then the training transform's
             assemble_reference(..., length=Lq, augment=False) with every augmentation off.
  tri        model.generate with _run_validation_trajectory's kwargs, k samples of the pair in one batch, unmasked
             query sequence, null fold label; seed_everything(0) before each pair (T7). Align-head outputs of the FIRST
             and LAST network call are captured with the dump_align_calls forward hook.
  c2c        each of the k maps -> ContactToCoord (EMA, 20 steps, its own > 0.5 binarisation), one draw per map,
             torch.manual_seed(1234 + ci) with ci = the query's index in the sorted pid list (stageB: chain-only seed).
One npz per pair at <out>/<pid>/<template>.npz (atomic, skipped if present), then a row in <out>/done_shard<k>.jsonl.
One --out per arm: template ids collide across arms (ladder vs indel), so a pairs file must hold ONE mode.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import proteinfoundation
from Frame2ConFind.inference.api import Frame2ConFindPredictor
from gen_c2c_structures import MODEL_CFG
from proteinfoundation.datasets.atom_features import atom14_features
from proteinfoundation.datasets.cath_utils import load_cath_mapping
from proteinfoundation.datasets.sse_topology import DSSP_HELIX, DSSP_STRAND, STRUCTURAL_PAIR_FEATURES
from proteinfoundation.datasets.topology_reference import TopologyReferenceTransform
from proteinfoundation.nn.af3_diffusion import C2C_INFERENCE_STEPS, SIGMA_DATA
from proteinfoundation.openfold_stub.np.residue_constants import restype_3to1
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils import precompute_synthetic_topology_index as builder
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from proteinfoundation.utils.topology_from_structure import parse_structure

assert os.path.abspath(proteinfoundation.__file__).startswith(REPO + os.sep), \
    f"proteinfoundation imported from {proteinfoundation.__file__}, not this checkout {REPO}"

EXP_CFG = "training_contact_tri_full384_confindsynth_ft_v1"
F2C_RUN = "f2s_v384_smoothap-pearson"   # the builder's --checkpoint default: the production winner
TRI_SEED = 0                            # T7 precedent (seed 0)
C2C_SEED_BASE = 1234                    # stageB_c2c_from_maps: 1234 + chain_index (+ seed_idx * 1e6, seed_idx 0 here)
# the training transform's values (dataset config) the reference must use; checked against the config at start
REF_EXPECT = dict(reference_source="synthetic", sse_types=[1, 2], max_topology_len=128, max_topology_he_len=64, min_len=1)
REF_AUG_OFF = dict(mutate_prob=0.0, sigma_frac=0.0, drop_prob=0.0, type_mutate_prob=0.0, token_mask_prob=0.0)


# ───────────────────────────────── template reference (builder path) ─────────────────────────────────
def structure_atom37(path, chain="A"):
    """PDB -> (graph, atom37 coords [L, 37, 3] with unresolved atoms at 0, atom37 mask [L, 37] bool)."""
    g = parse_structure(path, chain=chain)
    coords = g.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :].float()
    amask = g.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR].bool()
    # the builder fills `full = zeros; full[amask] = coords`; parse_structure leaves FILL_VALUE_COORDS there instead
    full = torch.where(amask[..., None], coords, torch.zeros_like(coords))
    return g, full, amask


def check_index_matches_builder(index):
    """The index records how it was built; the reference below must be built the same way."""
    assert index["contact_def"] == builder.CONTACT_DEF_CONFIND, f"index contact_def {index['contact_def']!r}"
    assert int(index["min_len"]) == REF_EXPECT["min_len"], f"index min_len {index['min_len']}"
    assert tuple(index["element_types"]) == builder._ELEMENT_TYPES == (DSSP_HELIX, DSSP_STRAND), index["element_types"]
    assert index["loop_axis"] is None and builder._LOOP_AXIS is None, index["loop_axis"]
    assert index["dssp_def"] in (builder.DSSP_DEF, builder.DSSP_DEF_PRO), index["dssp_def"]
    return index["dssp_def"] == builder.DSSP_DEF_PRO


def template_reference(full, amask, aatype, predictor, transform, length, pro_donor):
    """atom37 template -> (reference dict as the model sees it, elem_t [Lt] on the capped element axis, n_el, truncated).

    Same calls as precompute_synthetic_topology_index._chain_job / cmd_f2c for one template rung.
    """
    Lt = int(amask.shape[0])
    donor = torch.from_numpy(aatype.numpy() != builder.PRO_IDX)[None] if pro_donor else None
    dssp_t = compute_dssp_target(full[None], amask[:, builder.CA_IDX][None], coord_mask=amask[None],
                                 coord_layout="atom37", donor_mask=donor)
    assert dssp_t is not None, "template DSSP is None"
    x, mask = builder._f2c_inputs(full[None], amask)
    probs = predictor.predict_batch(x, mask[None])[:, :Lt, :Lt].float().cpu()
    # builder stores (probs >= threshold) as bits; load_f2c_maps unpacks them to float32 0/1
    cm_t = (probs[0] >= builder.CONFIND_THRESHOLD).float()
    row = builder.topology_row(cm_t, full, amask, dssp_t[0], REF_EXPECT["min_len"])
    assert row is not None, "template DSSP carries no usable residue"
    runs, ref_bytes, n_el, feat_bytes, _stats, elem = row
    # decoded as the transform reads the merged index (runs_flat int16, he_flat uint8, feat_flat float16)
    runs = [(int(t), int(n)) for t, n in runs]
    n_struct = len(STRUCTURAL_PAIR_FEATURES)
    if n_el > 0:
        he_contact = torch.frombuffer(bytearray(ref_bytes), dtype=torch.uint8).reshape(n_el, n_el).float()
        structural = torch.frombuffer(bytearray(feat_bytes), dtype=torch.float16).reshape(n_el, n_el, n_struct).float()
    else:
        he_contact = torch.zeros(0, 0)
        structural = torch.zeros(0, 0, n_struct)
    ref = transform.assemble_reference(runs, he_contact, structural, length, augment=False)
    n_he = int(ref["topology_he_tokens"].numel())
    # training maps the USalign truth through topology_row's elem, then _align_for drops elements beyond the cap
    elem_t = torch.from_numpy(elem.astype(np.int64))
    elem_t = torch.where(elem_t < n_he, elem_t, torch.full_like(elem_t, builder.ALIGN_NONE))
    n_tok = sum(1 for t, n in runs if t in transform.alphabet.types and n >= transform.alphabet.min_len)
    truncated = n_el > transform.max_topology_he_len or n_tok > transform.max_topology_len
    return ref, elem_t, n_el, truncated


def build_transform(index_path):
    """TopologyReferenceTransform with the cfft dataset config's values and every augmentation off."""
    with hydra.initialize_config_dir(os.path.join(REPO, "configs", "experiment_config"), version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=EXP_CFG)
    ds_dir = os.path.join(REPO, "configs", "datasets_config", cfg_exp.dataset_config_subdir)
    with hydra.initialize_config_dir(ds_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=cfg_exp.dataset)
    ents = [t for t in cfg_data.datamodule.transforms if t._target_.endswith("TopologyReferenceTransform")]
    assert len(ents) == 1, f"{cfg_exp.dataset}: {len(ents)} TopologyReferenceTransform entries"
    kw = OmegaConf.to_container(ents[0], resolve=False)
    cfg_index = kw.pop("index_path")
    kw.pop("_target_")
    assert os.path.basename(cfg_index) == os.path.basename(index_path), (cfg_index, index_path)
    for k, v in REF_EXPECT.items():
        assert kw[k] == v, f"dataset config {k}={kw[k]!r}, expected {v!r}"
    kw.update(REF_AUG_OFF)
    print(f"[ref] TopologyReferenceTransform({kw}) index={index_path}", flush=True)
    return TopologyReferenceTransform(index_path=index_path, **kw)


def load_predictor(ckpt_dir, dtype):
    assert os.path.basename(os.path.normpath(ckpt_dir)) == F2C_RUN, f"--f2c-ckpt-dir must be the {F2C_RUN} run"
    ck = os.path.join(ckpt_dir, "best.pt")
    pred = Frame2ConFindPredictor(checkpoint=ck, amp_dtype=dtype, compile_model=False)
    print(f"[f2c] {ck} amp={dtype} device={pred.device}", flush=True)
    return pred


# ───────────────────────────────────────────── models ─────────────────────────────────────────────
def load_tri(ckpt, store):
    """tri_gen_eval.py: compose the cfft experiment config, Proteina(cfg), EMA state_dict with 0 missing / 0 unexpected."""
    with hydra.initialize_config_dir(os.path.join(REPO, "configs", "experiment_config"), version_base=hydra.__version__):
        cfg_exp = hydra.compose(config_name=EXP_CFG)
    OmegaConf.set_struct(cfg_exp, False)
    cfg_exp.hardware.ngpus_per_node_ = 1
    cfg_exp.hardware.nnodes_ = 1
    cfg_exp.log.log_wandb = False
    cfg_exp.log.checkpoint = False
    assert ckpt.endswith("-EMA.ckpt"), f"--tri-ckpt must be a frozen EMA checkpoint (*-EMA.ckpt): {ckpt}"
    with open(ckpt, "rb") as fh:
        md5 = hashlib.file_digest(fh, "md5").hexdigest()
    print(f"[tri] {ckpt} md5={md5}", flush=True)
    model = Proteina(cfg_exp, store_dir=store)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[tri] {ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    assert not missing and not unexpected, f"missing {list(missing)[:5]} unexpected {list(unexpected)[:5]}"
    return model, cfg_exp


def load_c2c(ckpt):
    """stageB_c2c_from_maps.py --weights ema."""
    model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "ema" in ck, f"{ckpt} carries no EMA"
    model.model.load_state_dict(ck["ema"]["params"], strict=True)
    print(f"[c2c] EMA(decay={ck['ema'].get('decay')}) from {ckpt} global_step={ck.get('global_step')} "
          f"SIGMA_DATA={SIGMA_DATA}", flush=True)
    return model


def generate_tri(model, cfg_exp, topology, residue_type, mask, k):
    """model_trainer_base._run_validation_trajectory's generate() call, with its align-head first/last-call hook."""
    vs = cfg_exp.validation_sampling
    assert vs.get("sampling_grid", None) is None, "sampling_grid override path not replicated"
    cath_code_indices = cath_code_indices_mask = None
    if cfg_exp.training.get("fold_cond", False):
        cath_code_dir = cfg_exp.model.nn.get("cath_code_dir")
        if cath_code_dir is not None:
            _, _, _, nC, nA, nT = load_cath_mapping(cath_code_dir)
            cath_code_indices = torch.zeros((k, 1, 3), device=model.device, dtype=torch.long)
            cath_code_indices[:, 0, 0], cath_code_indices[:, 0, 1], cath_code_indices[:, 0, 2] = nC, nA, nT
            cath_code_indices_mask = torch.zeros((k, 1), device=model.device, dtype=torch.bool)
    rec = {"n": 0, "first": None, "last": None}

    def _rec(_mod, _inp, out):
        if isinstance(out, dict) and "align_logits" in out:
            cur = {kk: out[kk].detach().float().cpu() for kk in ("align_logits", "align_none_logits")}
            rec["first"] = cur if rec["first"] is None else rec["first"]
            rec["last"] = cur
            rec["n"] += 1

    hooks = [m.register_forward_hook(_rec)
             for m in {id(m): m for m in (model.nn, getattr(model, "nn_sc", None)) if m is not None}.values()]
    with torch.no_grad():
        result = model.generate(
            nsamples=k, n=mask.shape[1], dt=float(vs.get("dt", 0.005)), self_cond=cfg_exp.training.self_cond,
            cath_code=None, cath_code_indices=cath_code_indices, cath_code_indices_mask=cath_code_indices_mask,
            residue_type=residue_type, guidance_weight=1.0, autoguidance_ratio=0.0, dtype=torch.float32,
            schedule_mode=vs.get("schedule_mode", "uniform"), schedule_p=float(vs.get("schedule_p", 1.0)),
            sampling_mode=vs.get("sampling_mode", "sc"), sc_scale_noise=float(vs.get("sc_scale_noise", 0.45)),
            sc_scale_score=float(vs.get("sc_scale_score", 1.0)), gt_mode="us", gt_p=1.0, gt_clamp_val=None,
            mask=mask, x_motif=None, fixed_sequence_mask=None, fixed_structure_mask=None, topology=topology,
            zero_sin_pos_emb=bool(cfg_exp.training.get("zero_sin_pos_emb", False)),
        )
    for h in hooks:
        h.remove()
    assert rec["first"] is not None, "align head produced no output"
    return result, rec


def c2c_ca(c2c, prob, aatype, L, Lp, seed, dev):
    """One stageB rollout: contact_prob on [:L, :L], zeros elsewhere, encode (> 0.5 inside), rollout -> CA [L, 3]."""
    mask = torch.zeros(1, Lp, device=dev)
    mask[0, :L] = 1.0
    aa = torch.zeros(1, Lp, dtype=torch.long, device=dev)
    aa[0, :L] = aatype.to(dev)
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aa, mask)   # the features ContactToCoordTrainer._prepare builds
    contacts = torch.zeros(1, Lp, Lp, device=dev)
    contacts[0, :L, :L] = torch.from_numpy(prob).to(dev)
    with torch.no_grad():
        s, z, _ = c2c.model.encode(contacts, aa, mask)
        torch.manual_seed(seed)
        coords = c2c.model.rollout(s, z, mask, ref_feats, ref_pos, a2t, amask, ruid, n_steps=C2C_INFERENCE_STEPS)
    return coords.reshape(-1, Lp, 14, 3)[0, :L, 1, :].float().cpu().numpy()


# ────────────────────────────────────────────── main ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--tri-ckpt", required=True, help="cfft EMA checkpoint")
    ap.add_argument("--c2c-ckpt", required=True, help="ConFind twin c2c checkpoint (EMA used)")
    ap.add_argument("--index", required=True, help="topology_index_confind_synth.pt (standardisation constants)")
    ap.add_argument("--f2c-ckpt-dir", required=True, help=f".../runs/{F2C_RUN} (best.pt inside)")
    ap.add_argument("--f2c-dtype", choices=("fp32", "bf16", "fp16"), default="fp32", help="V100 has no bf16")
    ap.add_argument("--out", required=True, help="one directory per arm")
    ap.add_argument("--k", type=int, default=8, help="tri samples per pair (user: 8)")
    # validation pads every query to the dataset's PaddingTransform max_size (384), and stageB's c2c batches too
    ap.add_argument("--pad-to", type=int, default=384, help="padded query width for tri and c2c; 0 = no padding")
    ap.add_argument("--max-pairs", type=int, default=0, help="shakedown: only the first N pairs of this shard (0 = all)")
    ap.add_argument("--pairs-filter", default=None, help="shakedown: file of 'pid<TAB>template' lines to keep")
    args = ap.parse_args()
    assert 0 <= args.shard < args.n_shards
    assert os.environ.get("SIGMA_DATA") is not None and SIGMA_DATA == 16.0, f"SIGMA_DATA={SIGMA_DATA} (env must be 16)"

    rows = [json.loads(ln) for ln in open(args.pairs) if ln.strip()]
    modes = {r["mode"] for r in rows}
    assert len(modes) == 1, f"one arm per --pairs/--out (template ids collide across arms): modes {sorted(modes)}"
    keys = [(r["pid"], r["template"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate (pid, template) in --pairs"
    pids = sorted({r["pid"] for r in rows})
    ci_of = {p: i for i, p in enumerate(pids)}
    per_pid = {p: 0 for p in pids}
    for r in rows:
        per_pid[r["pid"]] += 1
    assert max(per_pid.values()) <= 1024 and len(pids) + args.n_shards <= 1024, "max 1024 files per directory"
    order = sorted(rows, key=lambda r: (int(r["Lq"]), int(r["Lt"]), r["pid"], r["template"]))
    mine = order[args.shard::args.n_shards]
    if args.pairs_filter:
        keep = {tuple(ln.split()[:2]) for ln in open(args.pairs_filter) if ln.strip()}
        unknown = keep - set(keys)
        assert not unknown, f"--pairs-filter names pairs absent from --pairs: {sorted(unknown)[:5]}"
        mine = [r for r in mine if (r["pid"], r["template"]) in keep]
    if args.max_pairs > 0:
        mine = mine[: args.max_pairs]
    print(f"[pairs] {args.pairs}: mode={modes.pop()} {len(rows)} pairs, {len(pids)} queries; shard "
          f"{args.shard}/{args.n_shards} -> {len(mine)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    # a crashed write leaves <dst>.part; only this shard's pairs are cleaned, so the other shards' live writes are untouched
    stale = [p for r in mine for p in glob.glob(os.path.join(args.out, r["pid"], f"{r['template']}.npz.part"))]
    for p in stale:
        os.remove(p)
    print(f"[resume] removed {len(stale)} stale .part files", flush=True)
    transform = build_transform(args.index)
    transform._ensure_loaded()
    pro_donor = check_index_matches_builder(transform._index)
    print(f"[ref] index dssp_def={transform._index['dssp_def']!r} contact_def={transform._index['contact_def']!r}",
          flush=True)
    predictor = load_predictor(args.f2c_ckpt_dir, args.f2c_dtype)
    model, cfg_exp = load_tri(args.tri_ckpt, os.path.join(args.out, f".store_shard{args.shard}"))
    model = model.to(dev).eval()
    c2c = load_c2c(args.c2c_ckpt).to(dev).eval()
    topo_keys = model._topology_keys()
    done_path = os.path.join(args.out, f"done_shard{args.shard}.jsonl")

    n_new = 0
    for j, r in enumerate(mine):
        pid, tpl = r["pid"], r["template"]
        dst = os.path.join(args.out, pid, f"{tpl}.npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        Lq, Lt, k = int(r["Lq"]), int(r["Lt"]), args.k
        gq, _, _ = structure_atom37(r["qpath"], chain="A")
        seq = "".join(restype_3to1[x] for x in gq.residues)
        assert len(seq) == Lq and seq == r["qs"], f"{pid}: parsed query ({len(seq)}) != qs ({len(r['qs'])}) / Lq {Lq}"
        gt, full_t, amask_t = structure_atom37(r["tpath"], chain="A")
        assert int(amask_t.shape[0]) == Lt, f"{tpl}: parsed template {amask_t.shape[0]} residues != Lt {Lt}"
        ref, elem_t, n_el, truncated = template_reference(full_t, amask_t, gt.residue_type, predictor, transform,
                                                          Lq, pro_donor)
        T = int(ref["topology_he_tokens"].numel())
        topology = {kk: Proteina._stack_topology([ref[kk]] * k).to(dev) for kk in topo_keys}

        n = Lq if args.pad_to == 0 else args.pad_to
        assert Lq <= n, f"{pid}: Lq {Lq} > pad width {n}"
        mask = torch.zeros(k, n, dtype=torch.bool, device=dev)
        mask[:, :Lq] = True
        residue_type = torch.zeros(k, n, dtype=torch.long, device=dev)   # PaddingTransform fill_value 0
        residue_type[:, :Lq] = gq.residue_type.to(dev)

        L.seed_everything(TRI_SEED)
        result, rec = generate_tri(model, cfg_exp, topology, residue_type, mask, k)
        prob = torch.sigmoid(result["contact_map_logits"].detach().float())[:, :Lq, :Lq].cpu().numpy()
        for w in ("first", "last"):
            assert tuple(rec[w]["align_logits"].shape) == (k, n, T), rec[w]["align_logits"].shape

        seed = C2C_SEED_BASE + ci_of[pid]
        ca = np.stack([c2c_ca(c2c, prob[i], gq.residue_type, Lq, n, seed, dev) for i in range(k)])

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        fh = open(tmp, "wb")
        np.savez_compressed(
            fh, pid=pid, template=tpl, mode=r["mode"], k=np.int32(k),
            contact_prob=prob.astype(np.float16),
            align_last=rec["last"]["align_logits"][:, :Lq, :T].numpy().astype(np.float32),
            align_none_last=rec["last"]["align_none_logits"][:, :Lq].numpy().astype(np.float32),
            align_first=rec["first"]["align_logits"][:, :Lq, :T].numpy().astype(np.float32),
            align_none_first=rec["first"]["align_none_logits"][:, :Lq].numpy().astype(np.float32),
            n_nn_calls=np.int32(rec["n"]),
            he_tokens=ref["topology_he_tokens"].numpy(), he_pos_raw=ref["topology_he_pos_raw"].numpy(),
            elem_t=elem_t.numpy().astype(np.int32),
            tmpl_ca=full_t[:, builder.CA_IDX, :].numpy().astype(np.float32),
            c2c_ca=ca.astype(np.float32),
            seeds=np.full((k,), seed, dtype=np.int64),
            n_el=np.int32(n_el), truncated=np.bool_(truncated), pad_to=np.int32(n),
            f2c_dtype=args.f2c_dtype, tri_ckpt=args.tri_ckpt, c2c_ckpt=args.c2c_ckpt,
        )
        fh.close()
        os.replace(tmp, dst)
        sec = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if dev.type == "cuda" else float("nan")
        with open(done_path, "a") as fh:
            fh.write(json.dumps({"pid": pid, "template": tpl, "file": dst, "Lq": Lq, "Lt": Lt, "T": T, "n_el": n_el,
                                 "truncated": bool(truncated), "tri_seed": TRI_SEED,
                                 "c2c_seed": seed, "seconds": round(sec, 2), "max_mem_gib": mem}) + "\n")
        n_new += 1
        print(f"[pair {j + 1}/{len(mine)}] {pid} {tpl} Lq={Lq} Lt={Lt} T={T} n_el={n_el} trunc={truncated} "
              f"{sec:.1f}s max_mem={mem:.2f}GiB", flush=True)
    n_have = sum(os.path.exists(os.path.join(args.out, r["pid"], f"{r['template']}.npz")) for r in mine)
    print(f"[done] shard {args.shard}: {n_new} new, {n_have}/{len(mine)} present", flush=True)
    return 0 if n_have == len(mine) else 3


if __name__ == "__main__":
    sys.exit(main())
