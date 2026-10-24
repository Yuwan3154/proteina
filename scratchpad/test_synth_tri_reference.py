"""Gate C2: synth_tri_bench.template_reference (structure -> reference) vs the synthetic index's own rows.

For template rows of topology_index_confind_synth.pt, rebuild the reference from the row's SOURCE structure with the
benchmark's builder and compare with TopologyReferenceTransform._build_reference(row): token / position tensors must
be EXACTLY equal (they depend on DSSP only); he_contact agreement and max |he_feat| difference are reported (the stored
maps came from Frame2ConFind under bf16 autocast, this run uses --f2c-dtype).

Cluster mode: --index and --rows (lines "<index row id>\t<template source structure PDB>", e.g. "1abc_A@rw50#3\t.../x.pdb");
the PDB goes through the SAME parse path as the benchmark (structure_atom37). Gate = tokens + positions exact; he_contact
agreement and max |he_feat diff| are printed only (no threshold).
Local mode (--fixture-dir DIR): DIR/fixture.json lists {stem, native, templates:[pdb,...]}; a one-chain index is first
built through the builder's own topology_row + load_f2c_maps + cmd_merge, from the same PDBs, then compared.
"""

import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from proteinfoundation.utils import precompute_synthetic_topology_index as builder
from proteinfoundation.utils.dssp_utils import compute_dssp_target
from synth_tri_bench import (build_transform, check_index_matches_builder, load_predictor, structure_atom37,
                             template_reference)

EXACT_KEYS = ("topology_tokens", "topology_pos", "topology_pos_raw", "topology_he_tokens", "topology_he_pos",
              "topology_he_pos_raw")


def builder_maps(full, amask, predictor):
    """cmd_f2c for one rung: thresholded, bit-packed, then read back with the builder's load_f2c_maps."""
    L = int(amask.shape[0])
    x, mask = builder._f2c_inputs(full[None], amask)
    probs = predictor.predict_batch(x, mask[None])[:, :L, :L].float().cpu()
    bits = (probs >= builder.CONFIND_THRESHOLD).numpy().reshape(1, -1)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.npz")
        np.savez(p, rungs=np.asarray([0], dtype=np.int16), L=np.int32(L), packed=np.packbits(bits, axis=1))
        return builder.load_f2c_maps(p)[0]


def build_fixture_index(fx_dir, predictor):
    """One chain: native row + one template row per fixture PDB, merged by the builder's cmd_merge."""
    spec = json.load(open(os.path.join(fx_dir, "fixture.json")))
    out = os.path.join(fx_dir, "synth_index")
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    chains, sources = [], {}
    for ch in spec:
        _, full_n, amask_n = structure_atom37(ch["native"])
        dssp_n = compute_dssp_target(full_n[None], amask_n[:, builder.CA_IDX][None], coord_mask=amask_n[None],
                                     coord_layout="atom37")[0]
        native = builder.topology_row(builder_maps(full_n, amask_n, predictor), full_n, amask_n, dssp_n, 1)
        rows = []
        for k, tp in enumerate(ch["templates"]):
            _, full, amask = structure_atom37(tp)
            dssp_t = compute_dssp_target(full[None], amask[:, builder.CA_IDX][None], coord_mask=amask[None],
                                         coord_layout="atom37", donor_mask=None)
            row = builder.topology_row(builder_maps(full, amask, predictor), full, amask, dssp_t[0], 1)
            align = np.full((int(amask_n.shape[0]),), builder.ALIGN_NONE, dtype=np.int16)
            rows.append((row[:5], 0.7, 50, k, align.tobytes(), 0.7))
            sources[f"{ch['stem']}@rw50#{k}"] = tp
        chains.append({"stem": ch["stem"], "native": native, "rows": rows, "seq_hash": 0})
    torch.save({"chains": chains, "n_parts": 1, "part": 0, "min_len": 1, "tm_build_range": (0.0, 1.0),
                "contact_def": builder.CONTACT_DEF_CONFIND, "dssp_def": builder.DSSP_DEF,
                "element_types": list(builder._ELEMENT_TYPES), "loop_axis": None},
               os.path.join(parts, "part_0000.pt"))
    idx = os.path.join(out, "topology_index_confind_synth.pt")
    builder.cmd_merge(SimpleNamespace(parts_dir=parts, n_parts=1, out=idx, eligible_out="", tm_range=(0.5, 0.9)))
    return idx, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None)
    ap.add_argument("--rows", default=None, help="lines: index row id <TAB> template source PDB")
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--f2c-ckpt-dir", required=True)
    ap.add_argument("--f2c-dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    args = ap.parse_args()
    predictor = load_predictor(args.f2c_ckpt_dir, args.f2c_dtype)

    if args.fixture_dir:
        index_path, sources = build_fixture_index(args.fixture_dir, predictor)
        cases = [(rid, ("pdb", p)) for rid, p in sources.items()]
    else:
        index_path = args.index
        cases = [(f[0], ("pdb", f[1])) for f in (ln.split() for ln in open(args.rows) if ln.strip())]
    tf = build_transform(index_path)
    tf._ensure_loaded()
    idx = tf._index
    pro_donor = check_index_matches_builder(idx)
    id_to_row = tf._id_to_row

    n_exact, worst_feat, agree = 0, 0.0, []
    for rid, (_, src) in cases:
        assert rid in id_to_row and not bool(idx["row_is_native"][id_to_row[rid]]), f"{rid}: not a template row of the index"
        row = id_to_row[rid]
        g, full, amask = structure_atom37(src)
        aatype = g.residue_type
        Lt = int(amask.shape[0])
        new, elem_t, n_el, trunc = template_reference(full, amask, aatype, predictor, tf, Lt, pro_donor)
        old = tf._build_reference(row, Lt, augment=False)
        bad = [kk for kk in EXACT_KEYS if new[kk].shape != old[kk].shape or not torch.equal(new[kk], old[kk])]
        n_exact += not bad
        c_new, c_old = new["topology_he_contact"], old["topology_he_contact"]
        a = float((c_new == c_old).float().mean()) if c_new.shape == c_old.shape else float("nan")
        jac = (float(((c_new > 0) & (c_old > 0)).sum() / ((c_new > 0) | (c_old > 0)).sum().clamp_min(1))
               if c_new.shape == c_old.shape else float("nan"))
        fd = (float((new["topology_he_feat"] - old["topology_he_feat"]).abs().max())
              if new["topology_he_feat"].shape == old["topology_he_feat"].shape else float("inf"))
        agree.append(a)
        worst_feat = max(worst_feat, fd)
        n_he = int(new["topology_he_tokens"].numel())
        assert int(elem_t.max()) < n_he and int(elem_t.min()) >= builder.ALIGN_NONE
        print(f"{rid:<40s} Lt={Lt} n_el={n_el} trunc={trunc} exact={'OK' if not bad else 'FAIL ' + ','.join(bad)} "
              f"he_contact agree={a:.4f} jaccard={jac:.4f} max|he_feat diff|={fd:.4g} "
              f"elem_t assigned={int((elem_t >= 0).sum())}/{Lt}", flush=True)
    assert cases, "no index rows compared"
    print(f"[summary] {n_exact}/{len(cases)} rows token/position-exact; he_contact agreement min "
          f"{min(agree):.4f} mean {sum(agree) / len(agree):.4f}; max |he_feat diff| {worst_feat:.4g}", flush=True)
    return 0 if n_exact == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
