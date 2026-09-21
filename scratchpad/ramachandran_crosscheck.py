"""Cross-check `helix_pos_frac` against the literature-standard Ramachandran alpha-L definition.

WHY: `helix_pos_frac` (fraction of CA pseudo-dihedrals in the 30-90 deg window that are POSITIVE)
is PROJECT-LOCAL -- introduced 2026-09-07, no literature citation -- yet the whole mirror programme
leans on its 0.12-native / 0.89-mirrored calibration. The underlying quantity (CA virtual torsion)
is standard and its sign genuinely inverts under reflection, but the summary statistic is ours.
The literature-standard way to identify a left-handed helix is the Ramachandran plot: alpha-L sits
in the POSITIVE-phi region, right-handed alpha at phi ~ -60 deg.

⛔ NO INVENTED THRESHOLD. The criterion is `phi > 0` -- the textbook coarse discriminator between
right- and left-handed backbone conformation. A published alpha-L phi/psi BOX would need numbers I
cannot cite from here, so it is deliberately not used; phi>0 needs nothing but zero.

⛔ SELF-VALIDATION FIRST. Before any comparison the script checks its own dihedral code against
textbook values: on real native chains the phi distribution must be overwhelmingly negative and
helical-region phi must centre near -60 deg. If that fails, the phi code is wrong and nothing
downstream is reported.

⛔ MASKS: phi(i) needs C(i-1),N(i),CA(i),C(i) and psi(i) needs N(i),CA(i),C(i),N(i+1); any residue
whose own or neighbour's backbone atoms are unresolved is EXCLUDED, not silently zero-filled.

Note on atom indexing: N, CA, C are at 0, 1, 2 in BOTH the PDB and openfold orderings (the
permutation only swaps CB and O), so phi/psi are unaffected by the reindex bug that corrupted the
contact maps. The reindex is still applied, for consistency with the pipeline.
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch

from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

N_IDX, CA_IDX, C_IDX = 0, 1, 2
HELICAL_LO, HELICAL_HI = 30.0, 90.0


def dihedral(p0, p1, p2, p3):
    """IUPAC-signed torsion in degrees (standard 'praxeolitic' formulation).

    ⛔⛔ THE LEADING NEGATION ON b0 IS THE WHOLE SIGN CONVENTION. Omitting it flips every angle,
    and the first version of this script did exactly that: it reported 93.8% of residues with
    phi > 0 where real proteins are ~90% phi < 0. The self-validation below caught it.
    ⚠️ Note the second self-check ("median phi in the (-90,-30) basin ~ -60") PASSED anyway while
    the code was wrong -- selecting on a flipped phi, that window captured the alpha-L population,
    whose median sits near -65 in the flipped frame. One criterion alone would have waved the bug
    through; it took the pair.
    ⚠️ `ca_dihedrals` below deliberately KEEPS the repo's own convention (no negation), because
    helix_pos_frac's 0.12-native calibration is defined in that frame. The two functions therefore
    carry OPPOSITE signs on purpose -- do not "harmonise" them.
    """
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def ca_dihedrals(ca):
    b0, b1, b2 = ca[1:-2] - ca[:-3], ca[2:-1] - ca[1:-2], ca[3:] - ca[2:-1]
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True))
    return np.degrees(np.arctan2((m * n2).sum(-1), (n1 * n2).sum(-1)))


def backbone(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    coords = d["coords"] if hasattr(d, "keys") else d.coords
    cm = d["coord_mask"] if (hasattr(d, "keys") and "coord_mask" in d) else getattr(d, "coord_mask", None)
    coords = coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
    if cm is not None:
        cm = cm[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
    xyz = np.asarray(coords, dtype=np.float64)
    if cm is None:
        ok = np.isfinite(xyz).all(-1)
    else:
        ok = np.asarray(cm).astype(bool) & np.isfinite(xyz).all(-1)
    return xyz, ok


def phi_psi(xyz, ok):
    """Per-residue phi/psi in degrees; NaN where any required atom is unresolved."""
    L = xyz.shape[0]
    phi = np.full(L, np.nan)
    psi = np.full(L, np.nan)
    have = ok[:, N_IDX] & ok[:, CA_IDX] & ok[:, C_IDX]
    for i in range(1, L):
        if have[i] and have[i - 1]:
            phi[i] = dihedral(xyz[i - 1, C_IDX], xyz[i, N_IDX], xyz[i, CA_IDX], xyz[i, C_IDX])
    for i in range(L - 1):
        if have[i] and have[i + 1]:
            psi[i] = dihedral(xyz[i, N_IDX], xyz[i, CA_IDX], xyz[i, C_IDX], xyz[i + 1, N_IDX])
    return phi, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/orcd/pool/006/chenxiou/proteina/data/pdb_train/processed")
    ap.add_argument("--probes", default="", help="comma-separated .pt paths to report in detail")
    ap.add_argument("--spans", default="", help="comma-separated start:len CA-dihedral spans")
    ap.add_argument("--n", type=int, default=400, help="random chains for the population check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/orcd/scratch/orcd/011/chenxiou/.tmp/rama_crosscheck.json")
    args = ap.parse_args()

    # ── SELF-VALIDATION against textbook values ────────────────────────────────────────────────
    files = []
    for sh in sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)):
        files.extend(glob.glob(os.path.join(sh, "*.pt")))
    rng = random.Random(args.seed)
    sample = rng.sample(files, min(args.n, len(files)))
    print(f"[pop] {len(sample)} chains sampled of {len(files)}", flush=True)

    all_phi, all_hpf, all_phipos = [], [], []
    for p in sample:
        xyz, ok = backbone(p)
        phi, psi = phi_psi(xyz, ok)
        good = ~np.isnan(phi)
        if good.sum() < 20:
            continue
        all_phi.append(phi[good])
        # helical-region phi, by the STANDARD psi companion window for alpha (psi in -70..-10 is
        # the right-handed alpha basin); used ONLY for the textbook self-check, not for alpha-L.
        resolved_ca = ok[:, CA_IDX] & np.isfinite(xyz[:, CA_IDX]).all(-1)
        if resolved_ca.sum() >= 8:
            d = ca_dihedrals(xyz[resolved_ca][:, CA_IDX, :])
            h = d[(np.abs(d) > HELICAL_LO) & (np.abs(d) < HELICAL_HI)]
            if len(h):
                all_hpf.append(float((h > 0).mean()))
                all_phipos.append(float((phi[good] > 0).mean()))

    cat = np.concatenate(all_phi)
    frac_neg = float((cat < 0).mean())
    # right-handed alpha basin: phi in (-90,-30) is where the textbook -60 sits
    alpha_like = cat[(cat > -90) & (cat < -30)]
    print("\n=== SELF-VALIDATION (must match textbook before anything else is reported) ===")
    print(f"  residues with phi < 0            : {100*frac_neg:.1f}%   (expect ~90%: proteins are right-handed)")
    print(f"  median phi in the (-90,-30) basin: {np.median(alpha_like):+.1f} deg   (expect ~ -60)")
    ok_val = frac_neg > 0.80 and -75 < np.median(alpha_like) < -45
    print(f"  VERDICT: {'PASS' if ok_val else 'FAIL'}")
    if not ok_val:
        raise SystemExit("phi/psi code disagrees with textbook values; refusing to report a cross-check")

    # ── POPULATION AGREEMENT ───────────────────────────────────────────────────────────────────
    hpf = np.array(all_hpf)
    ppos = np.array(all_phipos)
    r = float(np.corrcoef(hpf, ppos)[0, 1])
    print("\n=== POPULATION: helix_pos_frac vs fraction of residues with phi>0 ===")
    print(f"  n chains            : {len(hpf)}")
    print(f"  helix_pos_frac      : median {np.median(hpf):.4f}  mean {hpf.mean():.4f}")
    print(f"  frac(phi>0)         : median {np.median(ppos):.4f}  mean {ppos.mean():.4f}")
    print(f"  Pearson r           : {r:+.3f}")

    # ── PROBES: does the alpha-L span found by CA torsion have phi>0? ──────────────────────────
    detail = []
    if args.probes:
        paths = args.probes.split(",")
        spans = [tuple(int(x) for x in s.split(":")) for s in args.spans.split(",")]
        print("\n=== PROBES: does the CA-torsion alpha-L span carry phi>0? ===")
        print(f"{'chain':>12} {'span_res':>9} {'n':>3} {'phi>0 in span':>14} {'median phi':>11} "
              f"{'phi>0 elsewhere':>16}")
        for path, (st, run) in zip(paths, spans):
            xyz, ok = backbone(path)
            phi, _ = phi_psi(xyz, ok)
            resolved_ca = ok[:, CA_IDX] & np.isfinite(xyz[:, CA_IDX]).all(-1)
            idx = np.where(resolved_ca)[0]
            # CA dihedral j spans CA residues j..j+3 within the RESOLVED index space
            lo, hi = st, min(st + run + 3, len(idx))
            res = idx[lo:hi]
            seg = phi[res]
            seg = seg[~np.isnan(seg)]
            other = np.setdiff1d(idx, res)
            oth = phi[other]
            oth = oth[~np.isnan(oth)]
            if len(seg) == 0:
                continue
            row = {"chain": os.path.basename(path), "n_span": int(len(seg)),
                   "frac_phi_pos_span": float((seg > 0).mean()),
                   "median_phi_span": float(np.median(seg)),
                   "frac_phi_pos_elsewhere": float((oth > 0).mean()) if len(oth) else None}
            detail.append(row)
            print(f"{row['chain'][:12]:>12} {f'{res[0]}-{res[-1]}':>9} {len(seg):3d} "
                  f"{row['frac_phi_pos_span']:14.2f} {row['median_phi_span']:+11.1f} "
                  f"{row['frac_phi_pos_elsewhere']:16.3f}")

    json.dump({"n_chains": len(hpf), "pearson_r": r,
               "hpf_median": float(np.median(hpf)), "phipos_median": float(np.median(ppos)),
               "selfcheck_frac_phi_neg": frac_neg,
               "selfcheck_median_alpha_phi": float(np.median(alpha_like)),
               "probes": detail}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
