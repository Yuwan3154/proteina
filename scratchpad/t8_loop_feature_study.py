"""T8 loop-element feature study on REAL natives (user 2026-10-09: "Be careful about the features regarding loops and make
sure they look reasonable ... Thoroughly test and ask me for review"). Read-only, CPU.

For a seeded sample of T8 train chains, builds the reference the T8 index builder would build: ConFind contacts
(contact_map_confind >= 0.01), DSSP recomputed with the proline donor mask (Ca-only residues -> -1), every run of
type loop/helix/strand an ELEMENT (min_len 1), orientation under BOTH loop-axis options. Writes one npz with
  per chain:   L, element counts (all / helix / strand / loop);
  per element: type, length;
  per pair:    type pair, |i-j| in elements, contact_max, contact_frac, cos(zero), cos(end_to_end), 4 circuit
               channels, seq_gap.
Usage: python t8_loop_feature_study.py CHAINS.txt DATA_DIR N SEED OUT.npz
"""

import json
import pathlib
import random
import sys

import numpy as np
import torch

from proteinfoundation.datasets.pdb_data import _processed_path_sharded
from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_LOOP,
    DSSP_STRAND,
    assemble_pair_features,
    dssp_to_runs,
    element_lengths,
    sse_contact_reference,
    sse_structural_pair_features,
)
from proteinfoundation.utils.dssp_utils import compute_dssp_target

TYPES = (DSSP_LOOP, DSSP_HELIX, DSSP_STRAND)
PRO_IDX = 14            # openfold restype order (as the builder)
CONFIND_THRESHOLD = 0.01


def main(chains, data_dir, n, seed, out):
    stems = sorted(l.strip() for l in open(chains) if l.strip())
    random.Random(int(seed)).shuffle(stems)
    d = pathlib.Path(data_dir)
    man = json.load(open(d / "shard_manifest.json"))
    C, E, P = [], [], []    # chain rows, element rows, pair rows
    skipped = {}
    for s in stems:
        if len(C) >= int(n):
            break
        p = _processed_path_sharded(d / "processed", s, man)
        g = torch.load(p, map_location="cpu", weights_only=False)
        coords, cmask, rt = g.coords.float(), g.coord_mask.float(), getattr(g, "residue_type", None)
        L = int(coords.shape[0])
        cm = getattr(g, "contact_map_confind", None)
        if cm is None or tuple(cm.shape) != (L, L) or rt is None or len(rt) != L:
            skipped[s] = "no_confind_or_residue_type"
            continue
        cm = (cm.float() >= CONFIND_THRESHOLD).float()
        dssp = compute_dssp_target(coords[None], torch.ones(1, L, dtype=torch.bool), coord_mask=(cmask > 0.5)[None],
                                   coord_layout="pdb", donor_mask=(torch.as_tensor(rt) != PRO_IDX)[None])
        if dssp is None or bool((dssp[0] < 0).all()):
            skipped[s] = "dssp_none_or_all_invalid"
            continue
        runs = dssp_to_runs(dssp[0], min_len=1)
        ref, keep = sse_contact_reference(cm, runs, keep_types=TYPES)
        st_zero = sse_structural_pair_features(cm, coords, cmask, runs, keep, loop_axis="zero")
        st_e2e = sse_structural_pair_features(cm, coords, cmask, runs, keep, loop_axis="end_to_end")
        feat = assemble_pair_features(ref, st_zero, runs, keep)       # contact, frac, cos, 4 circuit, gap
        types = np.array([runs[i][0] for i in keep])
        lens = element_lengths(runs, keep).numpy()
        ci = len(C)
        C.append((ci, L, len(keep), int((types == DSSP_HELIX).sum()), int((types == DSSP_STRAND).sum()),
                  int((types == DSSP_LOOP).sum())))
        for a in range(len(keep)):
            E.append((ci, types[a], lens[a]))
        T = len(keep)
        iu, ju = np.triu_indices(T, k=1)
        f = feat.numpy()
        ce = st_e2e[..., 1].numpy()
        for a, b in zip(iu, ju):
            P.append((ci, types[a], types[b], b - a, *f[a, b, :2], f[a, b, 2], ce[a, b], *f[a, b, 3:]))
        if len(C) % 100 == 0:
            print(f"[study] {len(C)} chains, {len(E)} elements, {len(P)} pairs", flush=True)
    np.savez_compressed(out, chains=np.array(C, dtype=np.float64), elements=np.array(E, dtype=np.float64),
                        pairs=np.array(P, dtype=np.float32),
                        pair_cols=np.array(["chain", "type_a", "type_b", "elem_sep", "contact_max", "contact_frac",
                                            "cos_zero", "cos_e2e", "circ_series", "circ_contains", "circ_inside",
                                            "circ_cross", "seq_gap"]),
                        chain_cols=np.array(["chain", "L", "T", "n_helix", "n_strand", "n_loop"]),
                        skipped=np.array([f"{k}\t{v}" for k, v in skipped.items()]))
    print(f"[study] done: {len(C)} chains ({len(skipped)} skipped), {len(E)} elements, {len(P)} pairs -> {out}")


if __name__ == "__main__":
    assert len(sys.argv) == 6, __doc__
    main(*sys.argv[1:6])
