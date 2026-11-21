"""D36 W4 unit test: featurise ONE template under the 5 cells with the production scorer's own _featurize and assert
each cell changes exactly the intended template features. Positions are compared after removing the template's
CA-centroid translation (openfold zero-centres template coordinates over the atoms present, so stripping side chains
shifts the frame without changing geometry). Writes a per-key diff table to stdout.
"""
import argparse
import csv
import os
import sys

import torch
import openfold.np.residue_constants as rc

from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import OpenFoldAF2Rank, get_sequence_from_structure

CELLS = {"none": (False, "none"), "seqx": (False, "x"), "sc": (True, "none"), "both": (True, "x"),
         "gapsc": (True, "gap")}
TORS = {"template_torsion_angles_sin_cos", "template_alt_torsion_angles_sin_cos", "template_torsion_angles_mask"}
TOL = 2e-3  # PDB coordinates carry 3 decimals


def first(t):
    return t[0, ..., 0].float().cpu()


def centred(b, key):
    pos, msk = first(b["template_all_atom_positions"]), first(b["template_all_atom_mask"])
    ca = pos[:, 1][msk[:, 1] > 0].mean(0)
    x = first(b[key])
    return x - ca


def ndiff(a, b):
    if a.shape != b.shape:
        return -1
    if a.dtype.is_floating_point:
        return int((~torch.isclose(a.float(), b.float(), atol=1e-6, rtol=0, equal_nan=True)).sum())
    return int((a != b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--natives", default="/orcd/scratch/orcd/011/chenxiou/d36u/natives")
    args = ap.parse_args()
    man = list(csv.DictReader(open(args.manifest)))
    # deterministic pick: shortest target whose query has a Gly, its first all-atom template (C_ = colabfold AF2, E_ = c2c)
    by_len = sorted({(int(r["L"]), r["pid"]) for r in man})
    pick = None
    for L, p in by_len:
        cands = sorted((r for r in man if r["pid"] == p and r["source"][:2] in ("C_", "E_")), key=lambda r: r["template_id"])
        nat = [r for r in man if r["pid"] == p and r["template_id"] == "native"][0]
        if cands and "G" in get_sequence_from_structure(nat["path"], "A"):
            pick = cands[0]
            break
    p = pick["pid"]
    entry, chain = p.rsplit("_", 1)
    native = os.path.join(args.natives, f"{entry}.cif")
    scorer = OpenFoldAF2Rank(native, chain=chain, model_name="model_1_ptm", recycles=6,
                             use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=True,
                             use_cuequivariance_multiplicative_update=True, mask_sidechains=False,
                             usalign_path=None)
    assert "G" in scorer.reference_sequence, p
    feats = {}
    for cell, (msc, sm) in CELLS.items():
        scorer.mask_sidechains, scorer.template_seq_mask = msc, sm
        batch, _, _ = scorer._featurize(pick["path"], decoy_chain="A")
        feats[cell] = {k: v.detach().cpu() for k, v in batch.items() if torch.is_tensor(v)}
    print(f"[test] {p} L={len(scorer.reference_sequence)} template {pick['template_id']} ({pick['path']})")
    keys = sorted(feats["none"])
    pairs = [("none", "seqx"), ("sc", "both"), ("none", "sc"), ("both", "gapsc")]
    diff = {pr: {k: ndiff(feats[pr[0]][k], feats[pr[1]][k]) for k in keys} for pr in pairs}
    print(f"{'key':45s} " + " ".join(f"{a + '->' + b:>14s}" for a, b in pairs))
    for k in keys:
        if any(diff[pr][k] for pr in pairs):
            print(f"{k:45s} " + " ".join(f"{diff[pr][k]:>14d}" for pr in pairs))
    changed = {pr: {k for k in keys if diff[pr][k]} for pr in pairs}

    aat = feats["none"]["aatype"][..., 0].long()  # [N_res, R] -> recycle copy 0
    gly = aat == rc.restype_order["G"]
    n = aat.shape[0]
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    nontmpl = [k for k in keys if not k.startswith("template_")]
    check("non-template features identical in all 5 cells",
          all(ndiff(feats["none"][k], feats[c][k]) == 0 for c in CELLS for k in nontmpl))
    ta = {c: first(feats[c]["template_aatype"]).long() for c in CELLS}
    check("none: template_aatype == query aatype", torch.equal(ta["none"], aat))
    check("seqx/both: template_aatype all 20 (X)", bool((ta["seqx"] == 20).all() and (ta["both"] == 20).all()))
    check("gapsc: template_aatype all 21 (gap)", bool((ta["gapsc"] == 21).all()))
    check("sc: template_aatype == query aatype", torch.equal(ta["sc"], aat))
    check("none->seqx changes only template_aatype", changed[("none", "seqx")] == {"template_aatype"})
    check("sc->both changes only template_aatype", changed[("sc", "both")] == {"template_aatype"})

    allowed_sc = {"template_all_atom_positions", "template_all_atom_mask", "template_pseudo_beta"} | TORS
    check("none->sc changes only coordinates/masks/pseudo-beta(translation)/torsions",
          changed[("none", "sc")] <= allowed_sc)
    m_none, m_sc = first(feats["none"]["template_all_atom_mask"]), first(feats["sc"]["template_all_atom_mask"])
    check("sc: no atom beyond N/CA/C/CB/O (atom37 index >= 5 masked)", bool((m_sc[:, 5:] == 0).all()))
    check("none: has side-chain atoms (the strip had something to remove)", bool((m_none[:, 5:] > 0).any()))
    check("sc: CB present on every residue incl. Gly", bool((m_sc[:, 3] > 0).all()))
    x_none, x_sc = centred(feats["none"], "template_all_atom_positions"), centred(feats["sc"], "template_all_atom_positions")
    bb = [0, 1, 2, 4]
    check("sc: backbone N/CA/C/O geometry unchanged (CA-centred)",
          float((x_none[:, bb] - x_sc[:, bb]).abs().max()) < TOL)
    check("sc: non-Gly CB unchanged (CA-centred)", float((x_none[~gly, 3] - x_sc[~gly, 3]).abs().max()) < TOL)
    pb_none, pb_sc = centred(feats["none"], "template_pseudo_beta"), centred(feats["sc"], "template_pseudo_beta")
    check("sc: pseudo-beta unchanged up to translation (Gly still uses CA)", float((pb_none - pb_sc).abs().max()) < TOL)
    tm_sc = first(feats["sc"]["template_torsion_angles_mask"])
    tm_none = first(feats["none"]["template_torsion_angles_mask"])
    check("sc: chi torsion masks all 0", bool((tm_sc[:, 3:] == 0).all()))
    check("sc: backbone torsion masks unchanged", torch.equal(tm_sc[:, :3], tm_none[:, :3]))
    ts_sc = first(feats["sc"]["template_torsion_angles_sin_cos"])
    ts_none = first(feats["none"]["template_torsion_angles_sin_cos"])
    bbm = tm_none[:, :3] > 0
    check("sc: backbone torsion values unchanged where defined",
          float((ts_sc[:, :3][bbm] - ts_none[:, :3][bbm]).abs().max()) < 1e-2)

    allowed_gap = {"template_aatype", "template_pseudo_beta"} | TORS
    check("both->gapsc changes only template_aatype, pseudo-beta, torsions", changed[("both", "gapsc")] <= allowed_gap)
    pb_b, pb_g = first(feats["both"]["template_pseudo_beta"]), first(feats["gapsc"]["template_pseudo_beta"])
    moved = (pb_b - pb_g).abs().max(-1).values > 1e-4
    check("gapsc: pseudo-beta moves only at Gly positions", bool((moved <= gly).all()))
    x_g = first(feats["gapsc"]["template_all_atom_positions"])
    check("gapsc: Gly pseudo-beta == the added CB", float((pb_g[gly] - x_g[gly, 3]).abs().max()) < 1e-5)
    check("gapsc: Gly pseudo-beta moved (CA -> CB) on every Gly", bool(moved[gly].all()))
    tm_b, tm_g = first(feats["both"]["template_torsion_angles_mask"]), first(feats["gapsc"]["template_torsion_angles_mask"])
    check("gapsc vs both: torsion masks equal (chis already absent)", torch.equal(tm_b, tm_g))
    print(f"[test] {sum(ok)}/{len(ok)} checks passed (n_res {n}, Gly {int(gly.sum())})")
    print("D36W4_FEATURE_TEST_" + ("PASS" if all(ok) else "FAIL"), flush=True)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
