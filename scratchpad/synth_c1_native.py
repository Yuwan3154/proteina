"""Synthetic-template benchmark, control C1: the c2c twin fed each target's NATIVE contact map (no tri), to split the tri -> c2c TM
cap between c2c and the tri's map.

Native map = synth_tri_bench.template_reference's map built on the native (the ladder rewind-0 template IS the native, tpath == qpath):
Frame2ConFind (F2C_RUN, fp32) on the native backbone, prob >= builder.CONFIND_THRESHOLD, the ConFind definition the tri and the twin use
(CONTACT_DEF_CONFIND). c2c = synth_tri_bench.c2c_ca (EMA, 20 steps, pad 384, its own > 0.5 binarisation) with stageB_c2c_from_maps'
draw seeds torch.manual_seed(1234 + ci + si * 1_000_000), si = 0..k-1, ci = the target's index in the sorted pid list of the ladder pairs
(si = 0 is the exact seed stage 1 used for every tri sample of that target). One npz per target: <out>/<pid>.npz.
--map confind (C1b, item 868 (B)): the map the c2c twin TRAINED on instead -- its dataset yaml (--dataset-yaml, the twin's launcher
DATASET) has ContactMapTransform(contact_method confind), i.e. the stored contact_map_confind of the processed .pt (the ConFind PROGRAM's raw
scores, written by precompute_confind_maps._process_one: confind_raw_contact_map(graph, renumber=True) cast to float16), thresholded >= the
yaml's confind_contact_threshold by the transform's own _contact_map_from_confind_precomputed. The 22 targets are not all in the dataset, so
the raw map is computed by those same calls on the parse_structure graph of the native; --stored-pt-dir (<pdb>_<chain>.pt files of the
dataset, where they exist) is the control: same residues -> stored vs recomputed map, entry by entry.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from synth_tri_bench import C2C_SEED_BASE, builder, c2c_ca, load_c2c, load_predictor, structure_atom37
from proteinfoundation.datasets.transforms import ContactMapTransform
from proteinfoundation.nn.af3_diffusion import SIGMA_DATA
from proteinfoundation.openfold_stub.np.residue_constants import restype_3to1
from proteinfoundation.utils.confind_utils import confind_raw_contact_map

SEED_STRIDE = 1_000_000   # stageB_c2c_from_maps: seed = 1234 + chain_index + seed_idx * 1_000_000


def dataset_contact_transform(yaml_path):
    """The ContactMapTransform of the twin's dataset yaml, with the yaml's own method and threshold (rotamer lib passed separately)."""
    ts = [t for t in OmegaConf.load(yaml_path).datamodule.transforms if t["_target_"].endswith(".ContactMapTransform")]
    assert len(ts) == 1 and ts[0]["contact_method"] == "confind", f"{yaml_path}: {ts}"
    return ContactMapTransform(contact_method="confind", confind_contact_threshold=float(ts[0]["confind_contact_threshold"]))


def stored_check(pt, g, raw16, cmt):
    """Control: the dataset's stored .pt of this chain vs the recomputed map (a one-line verdict; compared only when the residues are the same)."""
    s = torch.load(pt, map_location="cpu", weights_only=False)
    if getattr(s, "contact_map_confind", None) is None:
        return "stored .pt has no contact_map_confind"
    same = (len(s.residue_type) == len(g.residue_type) and bool((s.residue_type == g.residue_type).all())
            and bool(torch.equal(s.coords[:, 1].float(), g.coords[:, 1].float())))
    if not same:
        return f"residues differ (stored L {len(s.residue_type)} vs native {len(g.residue_type)})"
    st = s.contact_map_confind.float()
    nb = int((cmt._contact_map_from_confind_precomputed(s) != cmt._contact_map_from_confind_precomputed(g)).sum())
    return f"same residues; raw max |diff| {float((st - raw16.float()).abs().max()):.4g}, {int((st != raw16.float()).sum())} raw / {nb} binary entries differ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="the LADDER pairs JSONL of stage 1 (pid order -> c2c seeds)")
    ap.add_argument("--c2c-ckpt", required=True)
    ap.add_argument("--f2c-ckpt-dir", default=None, help="--map f2c")
    ap.add_argument("--f2c-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="c2c draws per native map")
    ap.add_argument("--pad-to", type=int, default=384, help="as stage 1")
    ap.add_argument("--map", choices=("f2c", "confind"), default="f2c", help="f2c = C1 (the tri's representation); confind = C1b")
    ap.add_argument("--dataset-yaml", default=None, help="--map confind: the twin's dataset yaml")
    ap.add_argument("--confind-bin", default=None, help="--map confind: the ConFind binary")
    ap.add_argument("--rotlib", default=None, help="--map confind: the rotamer library dir (dataset yaml: DATA_PATH/rotlibs)")
    ap.add_argument("--stored-pt-dir", default=None, help="--map confind control: dataset .pt files <pdb>_<chain>.pt")
    args = ap.parse_args()
    assert os.environ.get("SIGMA_DATA") is not None and SIGMA_DATA == 16.0, f"SIGMA_DATA={SIGMA_DATA} (env must be 16)"

    rows = [json.loads(ln) for ln in open(args.pairs) if ln.strip()]
    assert {r["mode"] for r in rows} == {"ladder"}, "C1 takes the ladder pairs (rewind-0 template = the native)"
    pids = sorted({r["pid"] for r in rows})
    r0 = {r["pid"]: r for r in rows if int(r["rewind"]) == 0 and int(r["sample"]) == 0}
    assert set(r0) == set(pids), "every target needs a rewind-0 sample-0 row"
    for p in pids:
        assert r0[p]["tpath"] == r0[p]["qpath"], f"{p}: ladder rewind-0 template is not the native file"
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.map == "confind":
        assert args.dataset_yaml and args.confind_bin and args.rotlib, "--map confind needs --dataset-yaml --confind-bin --rotlib"
        cmt = dataset_contact_transform(args.dataset_yaml)
        assert args.stored_pt_dir is None or os.path.isdir(args.stored_pt_dir), f"--stored-pt-dir {args.stored_pt_dir} missing"
        n_cmp = 0
        print(f"[c1b] {cmt} threshold {cmt.confind_contact_threshold}", flush=True)
    else:
        assert args.f2c_ckpt_dir, "--map f2c needs --f2c-ckpt-dir"
        predictor = load_predictor(args.f2c_ckpt_dir, args.f2c_dtype)
    c2c = load_c2c(args.c2c_ckpt).to(dev).eval()

    for ci, pid in enumerate(pids):
        r = r0[pid]
        Lq = int(r["Lq"])
        g, full, amask = structure_atom37(r["qpath"], chain="A")
        seq = "".join(restype_3to1[x] for x in g.residues)
        assert len(seq) == Lq and seq == r["qs"], f"{pid}: parsed native != qs"
        if args.map == "confind":
            g.contact_map_confind = torch.as_tensor(confind_raw_contact_map(g, rotlib_path=args.rotlib, confind_bin=args.confind_bin,
                                                                            omp_threads=1, renumber=True), dtype=torch.float16)
            raw16 = g.contact_map_confind.clone()
            cm = cmt._contact_map_from_confind_precomputed(g).numpy()
            _, pdb, ch = pid.split("_", 2)
            pt = os.path.join(args.stored_pt_dir, f"{pdb.lower()}_{ch}.pt") if args.stored_pt_dir else None
            verdict = stored_check(pt, g, raw16, cmt) if pt and os.path.exists(pt) else "no stored .pt"
            n_cmp += verdict.startswith("same residues")
            print(f"[c1b control] {pid}: {verdict}", flush=True)
        else:
            # the same calls as synth_tri_bench.template_reference for the template's map
            x, mask = builder._f2c_inputs(full[None], amask)
            probs = predictor.predict_batch(x, mask[None])[:, :Lq, :Lq].float().cpu()
            cm = (probs[0] >= builder.CONFIND_THRESHOLD).float().numpy()
        n = Lq if args.pad_to == 0 else args.pad_to
        assert Lq <= n
        seeds = [C2C_SEED_BASE + ci + si * SEED_STRIDE for si in range(args.k)]
        ca = np.stack([c2c_ca(c2c, cm, g.residue_type, Lq, n, s, dev) for s in seeds])
        assert np.isfinite(ca).all(), f"{pid}: non-finite c2c CA"
        dst = os.path.join(args.out, f"{pid}.npz")
        fh = open(dst + ".part", "wb")
        np.savez_compressed(fh, pid=pid, native_map=np.packbits(cm.astype(np.uint8)), L=np.int32(Lq),
                            c2c_ca=ca.astype(np.float32), seeds=np.asarray(seeds, dtype=np.int64), ci=np.int32(ci),
                            pad_to=np.int32(n), f2c_dtype=args.f2c_dtype, c2c_ckpt=args.c2c_ckpt, map_source=args.map)
        fh.close()
        os.replace(dst + ".part", dst)
        print(f"[c1] {pid} L={Lq} ci={ci} map density {cm.mean():.4f} seeds {seeds[0]}..{seeds[-1]}", flush=True)
    if args.map == "confind":
        print(f"[c1b control] stored map compared for {n_cmp} of {len(pids)} targets", flush=True)
    n_have = sum(os.path.exists(os.path.join(args.out, f"{p}.npz")) for p in pids)
    print(f"[done] {n_have}/{len(pids)} targets", flush=True)
    return 0 if n_have == len(pids) else 3


if __name__ == "__main__":
    sys.exit(main())
