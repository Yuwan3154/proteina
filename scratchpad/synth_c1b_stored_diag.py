"""C1b control diagnosis (item 868 (B)): the dataset's stored contact_map_confind vs ConFind recomputed (a) on the native's parse_structure graph
(what C1b fed) and (b) on the STORED graph itself, by precompute_confind_maps' own call (confind_raw_contact_map(graph, renumber=True) ->
float16). (b) == stored -> the C1b input graph differs; (b) != stored -> ConFind run-to-run / version difference. Prints per target: density,
nonzero counts, diagonal, raw / binary (>= the yaml threshold) disagreement for stored-vs-(a), stored-vs-(b), (a)-vs-(b)."""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from synth_c1_native import dataset_contact_transform
from proteinfoundation.utils.confind_utils import confind_raw_contact_map
from proteinfoundation.utils.topology_from_structure import parse_structure


def cmp(x, y, thr):
    return (f"raw differ {int((x != y).sum())}/{x.numel()} max {float((x - y).abs().max()):.3g}; "
            f"binary differ {int(((x >= thr) != (y >= thr)).sum())} of {int((x >= thr).sum())} / {int((y >= thr).sum())} contacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natives", nargs="+", required=True, help="native PDB paths (pid from the file name, badafdb36_<PDB>_<chain>.pdb)")
    ap.add_argument("--stored-pt-dir", required=True)
    ap.add_argument("--dataset-yaml", required=True)
    ap.add_argument("--confind-bin", required=True)
    ap.add_argument("--rotlib", required=True)
    a = ap.parse_args()
    thr = dataset_contact_transform(a.dataset_yaml).confind_contact_threshold
    run = lambda g: torch.as_tensor(confind_raw_contact_map(g, rotlib_path=a.rotlib, confind_bin=a.confind_bin, omp_threads=1, renumber=True),
                                    dtype=torch.float16).float()
    for path in a.natives:
        _, pdb, ch = os.path.basename(path)[:-4].split("_", 2)
        s = torch.load(os.path.join(a.stored_pt_dir, f"{pdb.lower()}_{ch}.pt"), map_location="cpu", weights_only=False)
        st = s.contact_map_confind.float()
        g = parse_structure(path, chain="A")
        ra, rb, rb2 = run(g), run(s), run(s)
        print(f"== {pdb}_{ch} L {len(g.residue_type)}: stored dtype {s.contact_map_confind.dtype} nonzero {int((st != 0).sum())} diag mean "
              f"{float(st.diagonal().mean()):.3g} sym {bool(torch.equal(st, st.T))} | (a) nonzero {int((ra != 0).sum())} diag {float(ra.diagonal().mean()):.3g}"
              f" | (b) nonzero {int((rb != 0).sum())}", flush=True)
        print(f"   stored vs (a): {cmp(st, ra, thr)}\n   stored vs (b): {cmp(st, rb, thr)}\n   (a) vs (b): {cmp(ra, rb, thr)}\n"
              f"   (b) vs (b) rerun: {cmp(rb, rb2, thr)}", flush=True)


if __name__ == "__main__":
    main()
