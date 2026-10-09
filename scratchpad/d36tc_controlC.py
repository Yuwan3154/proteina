"""D36tc CONTROL C: the production AF2Rank path on Engaging L40S must reproduce the D36ap D1 gate's O6w cell (SuperCloud
V100): the 130 shipped decoys, whole chain, model_1_ptm, recycles 6, mask_sidechains, cuEq attention + mult-update ON,
DeepSpeed OFF, mask_inter_segment False -- the same call as d36ap_gate_of.py (`score_structure(allatom, "A",
_original_pdb=ca)`). Manifest paths under /home/gridsan/cou/ are remapped onto --root (an rsync copy of those files).

--mode score   : write one row per decoy to --out (resumable: rows already present are skipped).
--mode compare : vs o6w.csv. GATE (user 2026-10-09): ANY 0.7 / 0.5 bin flip of tm_ref_pred = FAIL. Reports max |delta|
                 of ptm, plddt, composite, tm_template_pred, tm_ref_template, tm_ref_pred. TF32 caveat: the scorer's
                 default precision "tf32" is plain fp32 on V100 but real TF32 on L40S (recorded, not changed).
"""
import argparse
import csv
import os
import shutil
import sys
import time

SC_PREFIX = "/home/gridsan/cou/"
FIELDS = ["pid", "structure_file", "ptm", "plddt", "composite", "pae_mean", "tm_template_pred",
          "tm_ref_template", "tm_ref_pred", "seconds", "gpu"]
NUM = ["ptm", "plddt", "composite", "tm_template_pred", "tm_ref_template", "tm_ref_pred"]


def remap(p, root):
    assert p.startswith(SC_PREFIX), p
    return os.path.join(root, p[len(SC_PREFIX):])


def binof(x):
    return ">=0.7" if x >= 0.7 else ("0.5-0.7" if x >= 0.5 else "<0.5")


def score(args):
    import torch
    from proteinfoundation.prediction_pipeline.af2rank_openfold_scorer import KALIGN_BINARY_PATH, OpenFoldAF2Rank
    print(f"[controlC] kalign={KALIGN_BINARY_PATH} exists={os.path.exists(KALIGN_BINARY_PATH)} "
          f"usalign={shutil.which('USalign')}", flush=True)
    man = list(csv.DictReader(open(args.manifest)))
    assert len(man) == 130, f"manifest has {len(man)} rows, want 130"
    for r in man:
        for k in ("allatom", "ca", "ref_cif"):
            assert os.path.exists(remap(r[k], args.root)), f"missing {remap(r[k], args.root)}"
    done = set()
    if os.path.exists(args.out):
        done = {(r["pid"], r["structure_file"]) for r in csv.DictReader(open(args.out))}
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fh.tell() == 0:
        w.writeheader()
    gpu = torch.cuda.get_device_name(0)
    pids = list(dict.fromkeys(r["pid"] for r in man))
    first = man[0]
    scorer = OpenFoldAF2Rank(remap(first["ref_cif"], args.root), chain=first["chain"], model_name="model_1_ptm",
                             recycles=6, use_deepspeed_evoformer_attention=False, use_cuequivariance_attention=True,
                             use_cuequivariance_multiplicative_update=True, mask_sidechains=True,
                             mask_inter_segment=False, usalign_path=shutil.which("USalign"))
    for pid in pids:
        rows = [r for r in man if r["pid"] == pid]
        scorer.reset_reference(remap(rows[0]["ref_cif"], args.root), rows[0]["chain"])
        scorer.set_segments(None)
        scorer._mask_inter_segment = False
        for r in rows:
            if (pid, r["structure_file"]) in done:
                continue
            t0 = time.time()
            s = scorer.score_structure(remap(r["allatom"], args.root), decoy_chain="A", recycles=6,
                                       _original_pdb=remap(r["ca"], args.root))
            torch.cuda.synchronize()
            w.writerow({"pid": pid, "structure_file": r["structure_file"], **{k: s.get(k, "") for k in FIELDS[2:9]},
                        "seconds": round(time.time() - t0, 3), "gpu": gpu})
            fh.flush()
        print(f"[controlC] done {pid}", flush=True)
    print("D36TC_CONTROLC_SCORE_DONE", flush=True)


def compare(args):
    ref = {(r["pid"], r["structure_file"]): r for r in csv.DictReader(open(args.reference)) if r["cell"] == "O6w"}
    got = {(r["pid"], r["structure_file"]): r for r in csv.DictReader(open(args.out))}
    assert len(ref) == 130, f"reference O6w rows {len(ref)}"
    missing = sorted(set(ref) - set(got))
    flips, dmax = [], {k: (0.0, None) for k in NUM}
    for key, r in ref.items():
        if key not in got:
            continue
        g = got[key]
        for k in NUM:
            d = abs(float(g[k]) - float(r[k]))
            if d > dmax[k][0]:
                dmax[k] = (d, key)
        if binof(float(g["tm_ref_pred"])) != binof(float(r["tm_ref_pred"])):
            flips.append((key, float(r["tm_ref_pred"]), float(g["tm_ref_pred"])))
    print(f"[controlC] {len(got)} of 130 scored; missing {len(missing)}")
    for k in NUM:
        print(f"  max |delta| {k:<17} {dmax[k][0]:.4f}  at {dmax[k][1]}")
    print(f"  tm_ref_pred 0.7/0.5 bin flips: {len(flips)} of {len(ref) - len(missing)}")
    for key, a, b in flips:
        print(f"    FLIP {key}: V100 {a:.4f} -> L40S {b:.4f}")
    print("  caveat: scorer precision 'tf32' = fp32 on V100, TF32 on L40S")
    ok = not missing and not flips
    print("CONTROL_C " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("score", "compare"), required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--reference", default="")
a = ap.parse_args()
score(a) if a.mode == "score" else compare(a)
