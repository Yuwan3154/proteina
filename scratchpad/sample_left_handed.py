"""Does c2c reproduce a NATIVE left-handed helix, or does it right-hand everything?

A contact map fixes the global fold only up to reflection, but a LOCAL left-handed segment inside an
otherwise right-handed chain IS determined by the map -- reflecting only that segment would break
its contacts. So native alpha-L segments separate "reproduces the handedness the map implies" from
"makes everything right-handed".

⛔⛔ THE CONTROL IS NOT OPTIONAL. This script builds a model input from a raw .pt file, and the
contact-map definition it must match is NOT the module default: the c2c dataset sets
`cb_fill: "pseudo_cb"` while ContactMapTransform defaults to "ca". Building on the default would
feed the model an off-distribution map that still looks plausible and the handedness verdict would
be meaningless. So the control takes a chain from the REAL dataloader, rebuilds it through this
script's path, and compares. Sampling refuses to run unless it passes.
[[feedback_pin_the_definition_in_the_data]]

⭐ ROOT CAUSE, FOUND 2026-09-21 AND FIXED (this cost four wrong hypotheses, recorded so nobody
repeats them). An earlier version of this script omitted the PDB->openfold atom reindex that
`pdb_data.py:1018-1019` applies to EVERY sample before the transforms. The .pt on disk is PDB-ordered
(N, CA, C, O, CB); openfold ordering is (N, CA, C, CB, O). ContactMapTransform reads index 3 as CB,
so without the reindex it silently built an OXYGEN-8A contact map: 1080/147456 entries (0.73%) wrong.
Three checks that FAILED to catch it, and why:
  - "the config resolves cb_fill=pseudo_cb" -- it parsed fine, but coord_mask[:,3] was then the
    OXYGEN mask (resolved almost everywhere), so missing_cb was all-False and pseudo_cb never fired.
    A config echo proves PARSING, not PARTICIPATION.
  - "coord_mask is IDENTICAL" -- the check read atom index 0 (N), a FIXED POINT of the permutation.
    Vacuous by construction. Any residue-level reduction (.any(-1), .sum()) is permutation-invariant too.
  - "CA distances differ by 0.044 A, ~400x float32 error" -- that was a torch.cdist artifact, not a
    coordinate difference: cdist uses the matmul identity and cancels catastrophically near d=0, with
    the max error on the DIAGONAL. Real rotation round-off is ~2e-5 A. CA is index 1, a fixed point,
    so CA coords were never affected at all.

⛔ A local handedness verdict is only interpretable if the GLOBAL fold was reproduced. Each row
reports proper/reflected RMSD so a failed generation cannot be read as a handedness result.
"""

import argparse
import json
import os

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from proteinfoundation.proteinflow.contact2coord_trainer import ContactToCoordTrainer
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.utils.dense_padding_data_loader import dense_padded_from_data_list

CA = 1
HELICAL_LO, HELICAL_HI = 30.0, 90.0


def ca_dihedrals(ca):
    b0, b1, b2 = ca[1:-2] - ca[:-3], ca[2:-1] - ca[1:-2], ca[3:] - ca[2:-1]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    return np.degrees(np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1)))


def kabsch_rmsd(a, b, allow_reflection=False):
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    if not allow_reflection and d < 0:
        u[:, -1] *= -1
    r = u @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(-1).mean()))


def build_transforms(cfg):
    """Instantiate ONLY the transforms this model's input depends on, from the config itself so the
    definition is pinned by the config and not restated here."""
    out = []
    for t in cfg.datamodule.transforms:
        tgt = t["_target_"].split(".")[-1]
        if tgt in ("ContactMapTransform", "PaddingTransform"):
            out.append((tgt, hydra.utils.instantiate(t)))
    return out


def load_and_build(path, transforms):
    """Reproduce PDBDataset.__getitem__ before applying transforms.

    ⛔⛔ THE REINDEX IS NOT OPTIONAL, AND OMITTING IT FAILS SILENTLY. The .pt on disk is in PDB atom
    ordering (N=0, CA=1, C=2, O=3, CB=4); the rest of the codebase expects openfold ordering
    (N, CA, C, CB, O), and `pdb_data.py:1018-1019` converts it for EVERY sample before the
    transforms run. `ContactMapTransform` then reads index 3 as CB.
    Skip this and index 3 is the backbone OXYGEN, so you silently build an oxygen-8A contact map
    instead of a pseudo-CB-8A one -- and because coord_mask[:,3] is then the oxygen mask (resolved
    almost everywhere), `missing_cb` is all-False and cb_fill="pseudo_cb" NEVER FIRES even though
    the config parsed it correctly. Measured cost: 1080/147456 entries (0.73%) wrong, with the
    config echo still printing the right thing. [[feedback_config_echo_proves_parsing_not_participation]]
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    d.coords = d.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    d.coord_mask = d.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
    for _, tf in transforms:
        d = tf(d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chains", required=True, help="comma-separated paths to processed .pt files")
    ap.add_argument("--spans", required=True,
                    help="comma-separated start:len (CA index, run length) per chain")
    ap.add_argument("--n", type=int, default=8, help="samples per chain")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dataset",
                    default="pdb_train_contact-CB8_S25_max384_purge-test_cutoff-190828")
    ap.add_argument("--out", default="/orcd/scratch/orcd/011/chenxiou/.tmp/lh_sampling.jsonl")
    args = ap.parse_args()

    with hydra.initialize("../configs/datasets_config/pdb", version_base=hydra.__version__):
        cfg = hydra.compose(config_name=args.dataset)
    OmegaConf.set_struct(cfg, False)
    tfs = build_transforms(cfg)
    print("[cfg] transforms applied:", [n for n, _ in tfs], flush=True)
    for n, tf in tfs:
        if n == "ContactMapTransform":
            print(f"[cfg] contact def: atom={tf.contact_atom_type} cutoff={tf.contact_distance_cutoff} "
                  f"cb_fill={getattr(tf, 'cb_fill', '?')} method={tf.contact_method}", flush=True)

    # ── CONTROL ────────────────────────────────────────────────────────────────────────────────
    # Rebuild a chain the real dataloader also yields and require an EXACT contact-map match.
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = None
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")
    ref = next(iter(dm.val_dataloader()))
    ref_id = ref["id"][0] if isinstance(ref["id"], (list, tuple)) else str(ref["id"])
    stem = str(ref_id)
    shard = stem[1:3]
    root = str(cfg.datamodule.data_dir).rstrip("/") + "/processed"
    cand = os.path.join(root, shard, stem + ".pt")
    print(f"[control] dataloader yielded {stem}; rebuilding from {cand}", flush=True)
    if not os.path.exists(cand):
        raise SystemExit(f"CONTROL IMPOSSIBLE: {cand} not found; refusing to sample unverified")
    mine = load_and_build(cand, tfs)
    mine_b = dense_padded_from_data_list([mine])
    a = ref["contact_map"].float()[0]
    b = mine_b["contact_map"].float()[0]
    n = min(a.shape[0], b.shape[0])
    diff = int((a[:n, :n] != b[:n, :n]).sum())
    # ⛔ STRICT equality is the right control now. The contact map is deterministic and
    # rotation-invariant (measured: the same chain under two random rotations gives ZERO differing
    # entries), so once the atom reindex is applied there is no legitimate source of disagreement.
    # Anything non-zero here means the reconstruction diverges from the pipeline again.
    if diff:
        raise SystemExit(
            f"CONTROL FAILED: {diff} of {n*n} contact entries differ from the dataloader's. "
            "Refusing to sample -- a handedness verdict on an unverified input is meaningless. "
            "First thing to check: is the PDB->openfold atom reindex still applied in "
            "load_and_build (pdb_data.py:1018-1019)?")
    print(f"[control] PASS -- contact map matches the dataloader EXACTLY "
          f"(0 of {n*n} entries differ).\n", flush=True)

    # ── model ──────────────────────────────────────────────────────────────────────────────────
    from scratchpad.gen_c2c_structures import MODEL_CFG  # single source for the arch
    model = ContactToCoordTrainer(model_cfg=dict(MODEL_CFG, n_diffusion_samples=8))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "ema" in ck:
        model.model.load_state_dict(ck["ema"]["params"], strict=True)
        src = f"EMA(decay={ck['ema'].get('decay')})"
    else:
        model.load_state_dict(ck["state_dict"], strict=True)
        src = "raw state_dict"
    print(f"[load] {src} @ global_step {ck.get('global_step')}", flush=True)
    model = model.to("cuda").eval()

    chains = args.chains.split(",")
    spans = [tuple(int(x) for x in s.split(":")) for s in args.spans.split(",")]
    assert len(chains) == len(spans), "one span per chain"

    fh = open(args.out, "w")
    print(f"{'chain':>10} {'smp':>4} {'prop_rmsd':>10} {'refl_rmsd':>10} {'gap':>7} "
          f"{'nat_pos':>8} {'gen_pos':>8} {'verdict':>22}")
    for path, (st, run) in zip(chains, spans):
        d = load_and_build(path, tfs)
        batch = dense_padded_from_data_list([d])
        # ⛔ Do NOT move the batch to cuda before _prepare: `mask_dict` is NESTED, so a flat
        # comprehension moves `aatype` but leaves `mask_dict["coords"]` on CPU, and
        # atom14_features then mixes devices. Build on CPU, move the RESULT.
        b = model._prepare(batch, train=False)
        b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
        keep = b["mask"][0].bool().cpu().numpy()
        nat = b["atom_pos"].reshape(1, -1, 14, 3)[0, :, CA, :].detach().cpu().numpy()[keep]
        nat_d = ca_dihedrals(nat)
        # ⛔⛔ DO NOT trust the externally supplied index. The scanner reports a start index into the
        # ORIGINAL residue array, while everything here lives in the MASK-COMPACTED CA array; with
        # any unresolved gap the two frames disagree and we would silently compare the WRONG
        # residues while every printed number still looked reasonable.
        # Re-derive the span from the native structure actually loaded here, in this frame.
        best, best_i, cur, cur_i = 0, -1, 0, -1
        for i, v in enumerate(nat_d):
            if HELICAL_LO < abs(v) < HELICAL_HI and v > 0:
                if cur == 0:
                    cur_i = i
                cur += 1
                if cur > best:
                    best, best_i = cur, cur_i
            else:
                cur = 0
        if best < run:
            print(f"  [warn] {os.path.basename(path)}: re-derived longest run {best} < scanner's "
                  f"{run}; using the re-derived span (frames differ only if the chain has gaps)",
                  flush=True)
        st, run = best_i, best
        if run <= 0:
            print(f"  [skip] {os.path.basename(path)}: no left-handed helical run in this frame")
            continue
        nat_seg = nat_d[st:st + run]
        for j in range(args.n):
            torch.manual_seed(1234 + j)
            with torch.no_grad():
                s, z, _ = model.model.encode(b["contacts"], b["aatype"], b["mask"])
                coords = model.model.rollout(s, z, b["mask"], b["ref_feats"], b["ref_pos"],
                                             b["atom_to_token"], b["atom_mask"],
                                             b["ref_space_uid"], n_steps=args.steps)
            gen = coords.reshape(1, -1, 14, 3)[0, :, CA, :].detach().cpu().numpy()[keep]
            p = kabsch_rmsd(gen, nat, False)
            r = kabsch_rmsd(gen, nat, True)
            gen_d = ca_dihedrals(gen)
            gen_seg = gen_d[st:st + run]
            nat_pos = float((nat_seg > 0).mean())
            gen_pos = float((gen_seg > 0).mean())
            # ⛔ only a well-reproduced global fold makes the local verdict meaningful
            if p > 8.0:
                verdict = "fold not reproduced"
            elif gen_pos >= 0.75:
                verdict = "LEFT-handed kept"
            elif gen_pos <= 0.25:
                verdict = "flipped to RIGHT"
            else:
                verdict = "mixed"
            print(f"{os.path.basename(path)[:10]:>10} {j:4d} {p:10.2f} {r:10.2f} {p-r:7.2f} "
                  f"{nat_pos:8.2f} {gen_pos:8.2f} {verdict:>22}", flush=True)
            fh.write(json.dumps({"chain": os.path.basename(path), "sample": j,
                                 "rmsd_proper": p, "rmsd_reflected": r,
                                 "native_pos_frac_seg": nat_pos, "gen_pos_frac_seg": gen_pos,
                                 "native_seg_deg": nat_seg.tolist(),
                                 "gen_seg_deg": gen_seg.tolist(),
                                 "span_start": st, "span_len": run,
                                 "verdict": verdict}) + "\n")
            fh.flush()
    fh.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
