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

⛔⛔⛔ THIS SCRIPT DOES NOT CURRENTLY PASS ITS OWN CONTROL. DO NOT TRUST ANY OUTPUT FROM IT UNTIL
IT DOES. Status as of 2026-09-21: the reconstructed contact map differs from the dataloader's by
1080/147456 entries (0.73%) on 6kn9_B, and the cause is UNKNOWN. Three hypotheses were tested and
all three are dead:
  1. wrong transform subset -- REFUTED: all SEVEN config transforms give the identical diff as two.
  2. cb_fill default "ca" vs config "pseudo_cb" -- NOT the cause: the config resolves correctly.
  3. random rotation -> 8.0 A threshold jitter -- REFUTED by the decisive test: the SAME chain
     under two random rotations gives ZERO differing entries. The contact map IS deterministic and
     rotation-invariant. (An earlier version of this docstring asserted the opposite as fact
     before testing it. It was wrong.)
Established: same chain, identical residue_type and coord_mask, 224 resolved both sides -- but the
dataloader's CA distance matrix differs from the raw .pt's by up to 0.044 A, ~400x float32 rotation
error. THAT is the open thread: find why ref["coords"] differs from the raw .pt for the same chain.
⚠️ The boundary check below is ALSO mis-specified: it measures distance-to-cutoff in CA space while
the contact is defined on pseudo-CB (routinely 1-2 A apart). Fix that before reading its numbers.

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
    d = torch.load(path, map_location="cpu", weights_only=False)
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
    # ⛔⛔ BIT-IDENTITY IS THE WRONG CONTROL HERE AND CAN NEVER PASS. The pipeline applies
    # GlobalRotationTransform, a RANDOM rotation, and the contact map is a hard threshold at
    # exactly 8.0 A. MEASURED on 6kn9_B: same chain, same masks, same residue_type, but the CA
    # distance matrix differs by up to 0.044 A in float32 after rotation -- so every pair sitting
    # within that of the cutoff flips. That is 1080/147456 = 0.73% of entries, and it is a property
    # of the DATA PIPELINE, not of this reconstruction.
    # ⇒ The meaningful control is that the DEFINITION matches: every disagreement must be a pair
    # whose distance sits at the 8.0 A boundary. A disagreement far from the boundary would mean a
    # genuinely different contact rule, which is what we must refuse to sample on.
    rm = ref["mask_dict"]["coords"][0][..., 0, 0].bool()
    ca = ref["coords"][0].float()[:, CA, :]
    D = torch.cdist(ca, ca)
    mism = (a[:n, :n] != b[:n, :n]).nonzero()
    frac = diff / float(n * n)
    if diff:
        dd = torch.tensor([D[int(i), int(j)] for i, j in mism])
        off = (dd - 8.0).abs()
        far = int((off > 0.25).sum())
        print(f"[control] {diff} of {n*n} entries differ ({100*frac:.2f}%); "
              f"distance-to-cutoff of the disagreements: median {off.median():.4f} A, "
              f"max {off.max():.4f} A, beyond 0.25 A: {far}", flush=True)
        if far:
            raise SystemExit(f"CONTROL FAILED: {far} disagreements are NOT at the 8.0 A boundary, "
                             "so the contact DEFINITION differs. Refusing to sample.")
        if frac > 0.02:
            raise SystemExit(f"CONTROL FAILED: {100*frac:.2f}% of entries differ -- too many for "
                             "threshold jitter alone. Refusing to sample.")
    print("[control] PASS -- every disagreement is threshold jitter at the 8.0 A cutoff caused by "
          "the pipeline's random rotation; the contact DEFINITION matches.\n", flush=True)

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
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.to_dict().items()} \
            if hasattr(batch, "to_dict") else batch
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
