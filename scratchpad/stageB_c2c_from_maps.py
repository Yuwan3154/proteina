"""Stage B (Directive B): tri-SAMPLED contact maps -> the matching c2c -> all-atom -> quality vs native.

Compares the ConFind and CB-8A pipelines by the STRUCTURES their sampled maps produce, not by P@L
across two contact definitions that cannot be compared with each other.

Per chain of --chains (sorted, de-duplicated; its position is chain_index):
  batch  built with the c2c dataset's OWN transforms, the model_trainer_base._fixed_val_batches path,
         and keyed by the REAL protein_id: PDBDataset silently moves on to the NEXT chain when a file
         is missing or a transform fails, so the returned id is asserted, never assumed.
  map    --maps_dir: each tri sample's contact_prob (sigmoid of the final-step logits) replaces
         b["contacts"] on [:L,:L], zeros elsewhere, diagonal as dumped. NOT binarised here:
         ContactToCoord.encode embeds (contacts > 0.5), and that is the only threshold (user decision).
         --native: b["contacts"] untouched -- the ceiling arm, control (a).
  draws  --n_seeds rollouts per map under torch.manual_seed(1234 + chain_index + seed_idx*1_000_000).
         The seed depends on the chain and seed_idx only, so every arm -- and every tri sample of one
         chain -- starts from the SAME noise, and arms are paired.
  score  USalign TM (native-normalised), handedness_metrics, dist_mae, rg_ratio, plus the tri sample's
         own metrics from samples.jsonl. One JSONL row per (chain, tri sample, seed), fsynced per row.

CONTROLS run on every chain BEFORE any rollout; if (b) or (c) fails the script exits 2, nothing sampled:
  (b) npz contact_gt == this dataset's native contact_map EXACTLY on [:L,:L] (same definition, same
      residue order). The map is rotation-invariant -- sample_left_handed.py measured ZERO differing
      entries across two random rotations -- so any difference is a real mismatch.
  (c) npz L == mask.sum(), and the mask is a contiguous prefix (else the [:L,:L] placement is wrong).
"""

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import hydra
from omegaconf import OmegaConf

from gen_c2c_structures import MODEL_CFG, usalign_tm  # scratchpad/, next to this file
from proteinfoundation.datasets.pdb_data import PDBDataset
from proteinfoundation.nn.af3_diffusion import C2C_INFERENCE_STEPS, SIGMA_DATA
from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.c2c_dump import handedness_metrics, write_atom14_pdb
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingDataLoader

MAX_FILES_PER_DIR = 1024


def offdiag_density(c):
    """Fraction of off-diagonal pairs that c2c's own (> 0.5) binarisation calls a contact."""
    off = ~np.eye(c.shape[0], dtype=bool)
    return float((c[off] > 0.5).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
    ap.add_argument("--dataset", required=True, help="the c2c's OWN dataset yaml (datasets_config/pdb)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--maps_dir", help="tri dump dir: samples.jsonl + <stem>_sKK.npz")
    src.add_argument("--native", action="store_true",
                     help="ceiling arm: the dataset's own contact map, untouched")
    ap.add_argument("--chains", required=True, help="chain list, first token per line")
    ap.add_argument("--n_seeds", type=int, default=1, help="c2c draws per map")
    ap.add_argument("--steps", type=int, default=C2C_INFERENCE_STEPS)
    ap.add_argument("--out", required=True, help="JSONL, one row per (chain, tri sample, seed)")
    ap.add_argument("--pdb_dir", default=None, help="also keep atom14 PDBs here (sharded)")
    ap.add_argument("--usalign", default="/home/chenxiou/.local/bin/USalign")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    assert os.access(args.usalign, os.X_OK), f"USalign not executable: {args.usalign}"
    assert os.path.isfile(args.ckpt), f"missing checkpoint: {args.ckpt}"
    # One fresh file per run: the wrapper's row-count gate assumes the file holds exactly this run.
    assert not (os.path.exists(args.out) and os.path.getsize(args.out) > 0), \
        f"--out {args.out} exists and is not empty"
    print(f"[edm] SIGMA_DATA={SIGMA_DATA} (env SIGMA_DATA={os.environ.get('SIGMA_DATA')!r}) "
          f"steps={args.steps}", flush=True)

    with open(args.chains) as fh:
        stems = sorted({ln.split()[0] for ln in fh if ln.strip()})
    assert stems, f"no chains in {args.chains}"

    tri = {}
    if args.maps_dir:
        with open(os.path.join(args.maps_dir, "samples.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    r = json.loads(ln)
                    tri.setdefault(r["stem"], []).append(r)
        missing = [st for st in stems if st not in tri]
        assert not missing, (f"{len(missing)}/{len(stems)} chains have no tri sample in "
                             f"{args.maps_dir}: {missing[:20]}")
        for st in stems:
            idx = [r["sample_index"] for r in tri[st]]
            assert len(idx) == len(set(idx)), f"{st}: duplicate sample_index in samples.jsonl {idx}"
            tri[st].sort(key=lambda r: r["sample_index"])
        per_chain = collections.Counter(len(tri[st]) for st in stems)
        print(f"[maps] {args.maps_dir}: {sum(len(tri[st]) for st in stems)} samples over "
              f"{len(stems)} chains, samples/chain {dict(per_chain)}; "
              f"{len(set(tri) - set(stems))} dumped chains not in --chains (ignored)", flush=True)
    else:
        print("[maps] --native: every chain is fed its own dataset contact map", flush=True)

    cfg_dir = os.path.join(REPO, "configs", "datasets_config", "pdb")
    with hydra.initialize_config_dir(cfg_dir, version_base=hydra.__version__):
        cfg_data = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg_data, False)
    cfg_data.datamodule.num_workers = 0
    cfg_data.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg_data.datamodule)
    cmt = [t for t in dm.transform.transforms if type(t).__name__ == "ContactMapTransform"]
    assert len(cmt) == 1, f"{args.dataset}: expected one ContactMapTransform, found {len(cmt)}"
    t = cmt[0]
    # Stamped into every row, so a reader takes the definition from the artefact, not from a default.
    contact_def = (f"method={t.contact_method} atom={t.contact_atom_type} "
                   f"cutoff={t.contact_distance_cutoff} cb_fill={t.cb_fill} "
                   f"confind_thr={t.confind_contact_threshold}")
    print(f"[data] dataset={args.dataset} contact_def: {contact_def}", flush=True)

    # Same construction as model_trainer_base._fixed_val_batches, one chain per batch.
    ds = PDBDataset(
        pdb_codes=[st.split("_", 1)[0] for st in stems],
        chains=[st.split("_", 1)[1] if "_" in st else None for st in stems],
        data_dir=str(dm.data_dir),
        transform=dm.transform,
        format=dm.format,
        in_memory=False,
        file_names=stems,
        num_workers=0,
    )
    loader = DensePaddingDataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        cath_code_dir=getattr(dm, "cath_code_dir", None),
        multilabel_mode=getattr(dm, "multilabel_mode", "sample"),
        cath_dedupe_codes=getattr(dm, "cath_dedupe_codes", True),
    )

    model = ContactToCoordTrainer(model_cfg=MODEL_CFG)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    gstep = ck.get("global_step")
    if args.weights == "ema":
        # Never fall back to raw silently: --weights says which model is being measured.
        assert "ema" in ck, f"--weights ema but {args.ckpt} carries no EMA"
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
        wsrc = f"EMA(decay={ck['ema'].get('decay')})"
    else:
        model.load_state_dict(ck["state_dict"], strict=True)
        wsrc = "RAW state_dict"
    del ck
    print(f"[load] {wsrc} from {args.ckpt} global_step={gstep}", flush=True)

    # ── build every batch and run controls (b)/(c) before any GPU work ───────────────────────────
    chains, bad_b, bad_c = [], [], []
    n_checked = 0
    for ci, (stem, raw) in enumerate(zip(stems, loader)):
        pid = str(raw["protein_id"][0])
        assert pid == stem, (f"chain_index {ci}: requested {stem}, loader returned {pid} "
                             f"(missing .pt or failed transform -> PDBDataset skipped ahead)")
        b = model._prepare(raw, train=False)
        m = b["mask"][0].bool()
        L = int(m.sum())
        prefix = bool(m[:L].all()) and not bool(m[L:].any())
        native = b["contacts"][0, :L, :L].numpy()
        if not prefix:
            bad_c.append(f"{stem}: mask is not a contiguous prefix (L={L})")
        for r in tri.get(stem, []):
            z = np.load(os.path.join(args.maps_dir, r["file"]))
            assert str(z["stem"]) == stem and int(z["sample_index"]) == r["sample_index"], \
                f"{r['file']}: npz stem/sample_index disagree with samples.jsonl"
            n_checked += 1
            Lt = int(z["L"])
            if Lt != L or int(r["L"]) != L:
                bad_c.append(f"{stem} s{r['sample_index']:02d}: npz L={Lt} jsonl L={r['L']} "
                             f"!= mask.sum()={L}")
                continue
            gt = z["contact_gt"]
            ndiff = int((gt != native).sum())
            if ndiff:
                bad_b.append(f"{stem} s{r['sample_index']:02d}: {ndiff}/{L * L} entries differ "
                             f"(off-diag density dumped {offdiag_density(gt):.4f} vs "
                             f"c2c native {offdiag_density(native):.4f})")
        chains.append((stem, b, L))
    assert len(chains) == len(stems), f"loader yielded {len(chains)} of {len(stems)} chains"

    if args.maps_dir:
        nb = len({x.split()[0] for x in bad_b})
        print(f"[control b] contact_gt vs c2c native on [:L,:L]: {nb}/{len(stems)} chains mismatch "
              f"({n_checked} maps read; a map failing (c) is not compared)", flush=True)
        for x in bad_b:
            print(f"  MISMATCH {x}", flush=True)
    else:
        print("[control b] n/a for --native (no dumped maps)", flush=True)
    print(f"[control c] L / contiguous-prefix failures: {len(bad_c)} "
          f"(over {len(stems)} chains, {n_checked} maps)", flush=True)
    for x in bad_c:
        print(f"  FAIL {x}", flush=True)
    if bad_b or bad_c:
        print("FATAL: control (b) or (c) failed -- no rollout run, no row written.", flush=True)
        return 2
    if args.maps_dir:
        print("[control a] run the --native arm with the same --ckpt/--chains/--n_seeds for the "
              "ceiling; it is a separate invocation.", flush=True)

    # ── rollouts ─────────────────────────────────────────────────────────────────────────────────
    dev = "cuda"
    model = model.to(dev).eval()
    tmp = None if args.pdb_dir else tempfile.mkdtemp(prefix="stageB_")
    n_rows, n_expected = 0, 0
    with open(args.out, "a") as fh:
        for ci, (stem, b, L) in enumerate(chains):
            Lp = b["mask"].shape[1]
            # CPU copies for scoring and PDB writing: write_atom14_pdb reads coordinates one by one.
            aa_c, m_c = b["aatype"][0], b["mask"][0]
            keep = m_c.bool()
            gt14 = b["atom_pos"].reshape(-1, Lp, 14, 3)[0].float()
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            ca_t = gt14[keep][:, 1, :].numpy()
            dt = np.linalg.norm(ca_t[:, None] - ca_t[None], axis=-1)
            dens_nat = offdiag_density(b["contacts"][0, :L, :L].cpu().numpy())
            # Sequence fingerprint: rotation- and definition-invariant, so rows join across arms.
            seq_fp = hashlib.sha1(aa_c[:L].numpy().astype(np.int8).tobytes()).hexdigest()[:12]
            maps = [None] if args.native else tri[stem]
            n_expected += len(maps) * args.n_seeds
            if args.pdb_dir:
                odir = os.path.join(args.pdb_dir, args.label, f"{ci // 1000:03d}", stem)
                assert 1 + len(maps) * args.n_seeds <= MAX_FILES_PER_DIR, \
                    f"{stem}: {1 + len(maps) * args.n_seeds} PDBs would exceed {MAX_FILES_PER_DIR}/dir"
            else:
                odir = tmp
            os.makedirs(odir, exist_ok=True)
            tp = os.path.join(odir, f"{stem}_native.pdb")
            write_atom14_pdb(tp, gt14, aa_c, m_c)

            for r in maps:
                if r is None:
                    contacts, dens_tri, tag = b["contacts"], None, "nativemap"
                else:
                    prob = np.load(os.path.join(args.maps_dir, r["file"]))["contact_prob"]
                    assert prob.shape == (L, L), f"{r['file']}: contact_prob {prob.shape} != ({L}, {L})"
                    contacts = torch.zeros_like(b["contacts"])
                    contacts[0, :L, :L] = torch.from_numpy(prob).to(dev)
                    dens_tri, tag = offdiag_density(prob), f"s{r['sample_index']:02d}"
                print(f"[map] {stem} {tag} L={L} off-diag density (>0.5): "
                      f"fed {dens_nat if dens_tri is None else dens_tri:.4f} native {dens_nat:.4f}",
                      flush=True)
                with torch.no_grad():
                    s, z, _ = model.model.encode(contacts, b["aatype"], b["mask"])
                for si in range(args.n_seeds):
                    seed = 1234 + ci + si * 1_000_000
                    torch.manual_seed(seed)
                    coords = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                                 b["atom_to_token"], b["atom_mask"],
                                                 b["ref_space_uid"], n_steps=args.steps)
                    gen14 = coords.reshape(-1, Lp, 14, 3)[0].float().cpu()
                    ca_g = gen14[keep][:, 1, :].numpy()
                    h = handedness_metrics(ca_g, ca_t)
                    assert h, f"{stem}: handedness_metrics returned nothing (L={len(ca_g)})"
                    dg = np.linalg.norm(ca_g[:, None] - ca_g[None], axis=-1)
                    iu = np.triu_indices(len(ca_g), 1)
                    mae = float(np.abs(dg[iu] - dt[iu]).mean())
                    rg = float(np.sqrt((dg ** 2).sum() / (2 * len(ca_g) ** 2)) /
                               np.sqrt((dt ** 2).sum() / (2 * len(ca_t) ** 2)))
                    gp = os.path.join(odir, f"{stem}_{tag}_seed{si:02d}_gen.pdb")
                    write_atom14_pdb(gp, gen14, aa_c, m_c)
                    tm = usalign_tm(gp, tp, args.usalign)
                    rec = dict(
                        label=args.label, stem=stem, chain_index=ci, seq_fp=seq_fp, L=L,
                        sample_index=None if r is None else r["sample_index"],
                        tri_file=None if r is None else r["file"],
                        seed=si, torch_seed=seed, weights=args.weights, ckpt=args.ckpt,
                        global_step=gstep, dataset=args.dataset, contact_def=contact_def,
                        map_source="native" if r is None else "tri", maps_dir=args.maps_dir,
                        steps=args.steps, sigma_data=SIGMA_DATA,
                        density_tri=dens_tri, density_native=dens_nat,
                        tm=tm, rmsd_proper=float(h["rmsd_proper"]),
                        rmsd_reflected=float(h["rmsd_reflected"]),
                        is_mirrored=float(h["is_mirrored"]),
                        refl_sign=1.0 if h["rmsd_proper"] > h["rmsd_reflected"] else 0.0,
                        helix_pos_frac=h.get("helix_pos_frac"),
                        rmsd_refl_gap=float(h["rmsd_refl_gap"]),
                        rmsd_refl_gap_frac=float(h["rmsd_refl_gap_frac"]),
                        dist_mae=mae, rg_ratio=rg,
                    )
                    if r is not None:
                        rec.update({f"tri_{k}": v for k, v in (r.get("metrics") or {}).items()})
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    n_rows += 1
                    print(f"  {stem} {tag} seed{si:02d}  tm {tm:.3f}  proper {rec['rmsd_proper']:.2f}"
                          f"  refl {rec['rmsd_reflected']:.2f}  dist_mae {mae:.2f}  rg {rg:.2f}",
                          flush=True)
    if tmp is not None:
        shutil.rmtree(tmp)

    if torch.cuda.is_available():
        print(f"[gpu] {torch.cuda.get_device_name(0)}  peak allocated "
              f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB, peak reserved "
              f"{torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB", flush=True)
    print(f"[done] {args.label}: {n_rows}/{n_expected} rows -> {args.out}", flush=True)
    return 0 if n_rows == n_expected else 3


if __name__ == "__main__":
    sys.exit(main())
