"""T8b eval, arm (b) (user 2026-11-07 "sse_ens24 + ContactEBM selection and refinement"): pool the 8 T8b samples of each
template SSE classifier (h_bond, tmalign, tmalign_smooth = 24 maps per chain, from three xrun_pass dumps of the same chunk),
pick the lowest ContactEBM energy at t_eval 0.25, run ContactEBM reverse sampling from t0 0.3 with K 0, and take the
minimum-energy predictor step of the trajectory's second half (minE).

The recipe is the code that produced the regimes-session numbers (toporef-regimes 81bccff, scratchpad/toporef_combo_bench.py
"C2 sse_ens24_pick" + toporef_regimes_bench.py load_contact_ebm / cebm_reverse_sample_from_t0, copied below), NOT the
later synth_tri_bench.py sse_ens24_cebm path (e5c1210), which imports contact_ebm.models.* (absent from ContactEBM) and adds
an energy gate the reported numbers never used.

Output: one tri-style dump dir that stageB_c2c_from_maps.py reads as is (samples.jsonl + <stem>_sKK.npz):
  s00 = minE map (the regime's output), s01 = the unrefined pick (its prob map), s02 = the trajectory's final map;
contact_gt and ref_id copied from the h_bond dump. Per-chain selection metadata in <out>/cebm.jsonl.
"""

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import torch

SSE_MODES = ("h_bond", "tmalign", "tmalign_smooth")   # sse_ens24 pool order (combo bench ens_modes)
K_PER_MODE = 8                                          # user 2026-10-28 "8 samples"
T_EVAL = 0.25                                           # regimes session: ContactEBM ranking t_eval
T0, K_CORR = 0.3, 0                                     # regimes session: sse_ens24 + ContactEBM (t0=0.3, K=0, minE)
TRI_SEED = 0                                            # combo bench seed for cebm_reverse_sample_from_t0
CEBM_MD5 = "2affbf45e2b28fecbc5f4d2a1dbe3b09"          # confind_a1_prod100_epoch82.pt (SuperCloud, 2026-11-07)


def _cebm_pair_mask(rmask):
    m = rmask.bool()
    L = m.shape[1]
    diag = torch.eye(L, dtype=torch.bool, device=m.device)[None]
    return (m[:, :, None] & m[:, None, :] & ~diag).float()


def load_contact_ebm(cebm_ckpt, dev):
    """toporef_regimes_bench.load_contact_ebm (81bccff)."""
    from contact_ebm.model.contact_ebm import ContactEBM
    from contact_ebm.scripts.train import build_diffusion

    ck = torch.load(cebm_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    mc = cfg["model"]
    diffc = cfg["diffusion"]
    diffusion, c_in = build_diffusion(diffc)
    diffusion = diffusion.to(dev)
    model = ContactEBM(
        vocab_size=diffc.get("vocab_size", 2),
        contact_input_dim=c_in,
        arch=mc["arch"],
        token_s=mc["token_s"],
        token_z=mc["token_z"],
        depth=mc["depth"],
        heads=mc["heads"],
        dim_fourier=mc.get("dim_fourier", 256),
        energy_dim=mc.get("energy_dim", 8),
        activation_checkpointing=False,
    ).to(dev)
    state = ck["ema"] if "ema" in ck else ck["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[cebm] loaded {cebm_ckpt} arch={mc['arch']} step={ck.get('step', ck.get('gstep'))} ema={'ema' in ck}", flush=True)
    return model, diffusion


def score_clean(cebm_model, prob_maps, aatype, Lq, t_eval, dev):
    """toporef_combo_bench.score_maps_multi_t (81bccff) at one t_eval: clean energies of the >= 0.5 binarised maps."""
    k = prob_maps.shape[0]
    binary_tokens = torch.from_numpy((prob_maps >= 0.5).astype(np.int64)).to(dev)
    aa_batch = aatype[:Lq].long().to(dev)[None].expand(k, Lq)
    rmask = torch.ones((k, Lq), dtype=torch.float32, device=dev)
    with torch.no_grad():
        return cebm_model.score_energy(binary_tokens, aa_batch, rmask, t_eval=float(t_eval)).detach().float().cpu().numpy()


def cebm_reverse_sample_from_t0(cebm_model, cebm_diff, init_binary_map, aatype, Lq, t0, K_corrector, dev, t_eval, seed):
    """toporef_regimes_bench.cebm_reverse_sample_from_t0 (81bccff), returning the per-step clean maps unpacked."""
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
            if hasattr(cebm_diff, "mask_id"):
                fresh = torch.full_like(z_s, cebm_diff.mask_id)
            else:
                fresh = cebm_diff.prior_sample(z_s.shape, device=dev)
            out = torch.where(keep, z_s, fresh)
            out = symmetrize_matrix(out.float()).long()
            out[:, diag, diag] = 0
            return out * pm_b.long()

    step_times, step_kinds, step_e, step_maps = [], [], [], []

    def _call(t_tensor):
        with torch.enable_grad():
            logits, energy_t, _ = cebm_model.compute_outputs(cebm_model._onehot(zt), aa, rmask, t_tensor, create_graph=False)
        logits_d = logits.detach()
        b_clean = (torch.sigmoid(logits_d.float())[0, :Lq, :Lq].cpu().numpy() >= 0.5).astype(np.uint8)
        with torch.no_grad():
            e_eval = float(cebm_model.score_energy(torch.from_numpy(b_clean.astype(np.int64)).to(dev)[None], aa, rmask,
                                                   t_eval=float(t_eval))[0].item())
        return logits_d, b_clean, e_eval

    for i in range(i0, timesteps):
        s, t = cebm_diff.get_sampling_grid(i, timesteps)
        t_tensor = torch.full((1,), float(t), device=dev, dtype=torch.float32)
        if K_corrector > 0 and s > 0:
            for kk in range(K_corrector):
                logits_d, b_clean, e_eval = _call(t_tensor)
                with torch.no_grad():
                    zc = cebm_diff.sample_step(zt, logits_d, s, t, pair_mask=pair_mask)
                    zt = _renoise(zc, s, t)
                step_times.append(float(t)); step_kinds.append(f"corrector_{kk}"); step_e.append(e_eval); step_maps.append(b_clean)
        logits_d, b_clean, e_eval = _call(t_tensor)
        with torch.no_grad():
            zt = cebm_diff.sample_step(zt, logits_d, s, t, pair_mask=pair_mask)
        step_times.append(float(t)); step_kinds.append("predictor"); step_e.append(e_eval); step_maps.append(b_clean)

    with torch.no_grad():
        z0 = (torch.sigmoid(logits_d.float()) >= 0.5).long() * pm_b.long()
        final_binary = z0[0, :Lq, :Lq].detach().cpu().numpy().astype(np.uint8)
        final_e = float(cebm_model.score_energy(z0[:, :Lq, :Lq].long(), aa, rmask, t_eval=float(t_eval))[0].item())
    return {"t_start": float(t_start), "step_times": step_times, "step_kinds": step_kinds, "step_e": step_e,
            "step_maps": step_maps, "final_binary": final_binary, "final_e": final_e}


def pack_reader(pack):
    z = np.load(pack + ".idx.npz", allow_pickle=True)
    idx = {str(s): (int(o), int(l)) for s, o, l in zip(z["stems"], z["offsets"], z["lengths"])}
    fh = open(pack, "rb")

    def residue_type(stem):
        o, l = idx[stem]
        fh.seek(o)
        d = torch.load(io.BytesIO(fh.read(l)), map_location="cpu", weights_only=False)
        rt = d["residue_type"] if isinstance(d, dict) else d.residue_type
        return torch.as_tensor(rt).long()
    return residue_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True, help="comma list of the h_bond,tmalign,tmalign_smooth dump dirs (that order)")
    ap.add_argument("--chains", required=True, help="the chunk's _q list")
    ap.add_argument("--pack", required=True, help="T8b training pack (query residue types)")
    ap.add_argument("--cebm-repo", required=True)
    ap.add_argument("--cebm-ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.cebm_repo)
    dumps = a.dumps.split(",")
    assert len(dumps) == len(SSE_MODES), dumps
    for d, m in zip(dumps, SSE_MODES):
        assert d.rstrip("/").endswith(f"_t7_{m}_c" + d.rstrip("/").rsplit("_c", 1)[1]), f"dump {d} is not the {m} pass"
    assert not os.path.exists(a.out), f"{a.out} exists (one fresh dir per run)"
    with open(a.cebm_ckpt, "rb") as fh:
        md5 = hashlib.file_digest(fh, "md5").hexdigest()
    assert md5 == CEBM_MD5, f"ContactEBM ckpt md5 {md5} != {CEBM_MD5}"
    dev = torch.device("cuda")
    cebm_model, cebm_diff = load_contact_ebm(a.cebm_ckpt, dev)
    residue_type = pack_reader(a.pack)
    rows = {}
    for d, m in zip(dumps, SSE_MODES):
        for ln in open(os.path.join(d, "samples.jsonl")):
            if ln.strip():
                r = json.loads(ln)
                rows.setdefault(r["stem"], {}).setdefault(m, []).append((r, d))
    stems = sorted({l.split()[0] for l in open(a.chains) if l.strip()})
    tmp = a.out + ".part"
    assert not os.path.exists(tmp), f"{tmp} exists (crashed run?)"
    os.makedirs(tmp)
    for ci, stem in enumerate(stems):
        maps, labels, ref_ids, gt, L = [], [], set(), None, None
        for m in SSE_MODES:
            got = sorted(rows[stem][m], key=lambda x: x[0]["sample_index"])
            assert [r["sample_index"] for r, _ in got] == list(range(K_PER_MODE)), f"{stem} {m}: samples {len(got)}"
            for r, d in got:
                z = np.load(os.path.join(d, r["file"]))
                L = int(z["L"]) if L is None else L
                assert int(z["L"]) == L, f"{stem}: L differs across passes"
                maps.append(z["contact_prob"].astype(np.float32))
                labels.append(f"{m}#{r['sample_index']}")
                ref_ids.add(r["ref_id"])
                if gt is None:
                    gt = z["contact_gt"]
                assert np.array_equal(z["contact_gt"], gt), f"{stem}: contact_gt differs across passes"
        assert len(ref_ids) == 1, f"{stem}: ref_ids {ref_ids}"
        ref_id = ref_ids.pop()
        aat = residue_type(stem)
        assert int(aat.shape[0]) == L, f"{stem}: pack residue_type length {aat.shape[0]} != dump L {L}"
        pool = np.stack(maps)
        # per mode in batches of 8, as the reference scored them (combo bench: one score_maps_multi_t call per mode)
        e25 = np.concatenate([score_clean(cebm_model, pool[i:i + K_PER_MODE], aat, L, T_EVAL, dev)
                              for i in range(0, len(pool), K_PER_MODE)])
        pick = int(np.argmin(e25))
        init_bmap = (pool[pick] >= 0.5).astype(np.uint8)
        res = cebm_reverse_sample_from_t0(cebm_model, cebm_diff, init_bmap, aat, L, t0=T0, K_corrector=K_CORR, dev=dev,
                                          t_eval=T_EVAL, seed=TRI_SEED)
        pred = [i for i, k in enumerate(res["step_kinds"]) if k == "predictor"]
        second_half = pred[len(pred) // 2:]                  # combo bench: minE over the second half (t <= t0 / 2)
        min_e_idx = min(second_half, key=lambda i: res["step_e"][i])
        mine = res["step_maps"][min_e_idx].astype(np.float32)
        n_diag = int(np.diagonal(mine).sum())
        np.fill_diagonal(mine, 0.0)   # the reference fed c2c the upper-triangle pack/unpack round trip (zero diagonal)
        outs = (("minE", mine), ("pick", pool[pick]),
                ("final", res["final_binary"].astype(np.float32)))
        with open(os.path.join(tmp, "samples.jsonl"), "a") as fh:
            for k, (tag, mp) in enumerate(outs):
                f = f"{stem}_s{k:02d}.npz"
                np.savez_compressed(os.path.join(tmp, f), contact_prob=mp, contact_gt=gt, L=np.int32(L), stem=stem,
                                    sample_index=np.int32(k), ref_id=ref_id, cebm_output=tag)
                fh.write(json.dumps({"stem": stem, "sample_index": k, "L": L, "ref_id": ref_id, "file": f,
                                     "metrics": {}, "cebm_output": tag}) + "\n")
        meta = {"stem": stem, "L": L, "ref_id": ref_id, "pick_idx": pick, "pick_label": labels[pick],
                "pick_mode": labels[pick].split("#")[0], "e25_pool": [round(float(x), 4) for x in e25],
                "pool_labels": labels, "t_start": res["t_start"], "n_calls": len(res["step_kinds"]),
                "minE_idx": min_e_idx, "minE_t": res["step_times"][min_e_idx], "minE_e25": res["step_e"][min_e_idx],
                "final_e25": res["final_e"], "pick_e25": float(e25[pick]), "minE_diag_ones_zeroed": n_diag,
                "t_eval": T_EVAL, "t0": T0, "K": K_CORR, "seed": TRI_SEED, "cebm_ckpt": a.cebm_ckpt, "cebm_md5": md5}
        with open(os.path.join(tmp, "cebm.jsonl"), "a") as fh:
            fh.write(json.dumps(meta) + "\n")
        print(f"[{ci + 1}/{len(stems)}] {stem} L={L} pick={labels[pick]} E25 {e25[pick]:.2f} -> minE {meta['minE_e25']:.2f}"
              f"@t={meta['minE_t']:.3f} (final {res['final_e']:.2f})", flush=True)
    os.replace(tmp, a.out)
    print(f"[done] {len(stems)} chains -> {a.out}")


if __name__ == "__main__":
    main()
