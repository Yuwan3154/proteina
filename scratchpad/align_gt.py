"""Ground truth for the alignment probe: which reference SSE element each query residue aligns to.

For a (query, reference) chain pair this builds a binary L x T matrix A, where A[i, e] = 1 iff
USalign structurally aligns query residue i to a reference residue lying inside reference SSE
element e. Q = the number of query residues with any such alignment, and precision@Q is scored
against Q because Q bounds what is achievable.

Three things this has to get exactly right or the probe's targets silently misalign with the
model's own T axis:

  1. INDEXING. The model's residue i is the graph's residue i, so the PDBs handed to USalign are
     written FROM THE GRAPH COORDS, in graph order, and a written->graph index map is kept.
     Residues without a resolved CA are skipped and never silently renumbered.
  2. ELEMENT IDENTITY. `TopologyReferenceTransform.assemble_reference` computes
     `keep = [i for i,(t,_) in enumerate(runs) if t in (H,E)]` BEFORE augmentation and then takes
     `keep[:k]` with `k = min(len(keep), max_topology_he_len)`. Element e is therefore run
     `keep[e]` of the UNAUGMENTED runs -- which is what we want, since ground truth is about the
     real structure, not the length-jittered tokens the model was shown.
  3. CROSS-PROTEIN MODE. Query and reference are different chains, so USalign runs sequence
     -independently (`-TMscore 0`) per the standing project rule. Never `-TMscore 5`, which
     assumes a shared residue numbering.
"""

import os
import pathlib
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

USALIGN = os.environ.get("USALIGN", "/home/chenxiou/.local/bin/USalign")
DSSP_HELIX, DSSP_STRAND = 1, 2
CA = 1  # index of the CA atom in the graph's per-residue atom axis


def write_ca_pdb(coords, coord_mask, path):
    """CA-only PDB in graph order. Returns written_index -> graph_index."""
    idx_map = []
    with open(path, "w") as fh:
        n = 0
        for i in range(coords.shape[0]):
            if coord_mask is not None and not bool(coord_mask[i, CA]):
                continue
            x, y, z = (float(v) for v in coords[i, CA])
            n += 1
            fh.write(
                f"ATOM  {n:5d}  CA  GLY A{n:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            idx_map.append(i)
        fh.write("TER\nEND\n")
    return idx_map


def usalign_pairs(pdb_q, pdb_r):
    """(query_written_idx, ref_written_idx) pairs from USalign's alignment block."""
    out = subprocess.run(
        [USALIGN, pdb_q, pdb_r, "-TMscore", "0"],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"USalign rc={out.returncode}: {out.stderr[:400]}")
    lines = out.stdout.splitlines()
    anchor = next((i for i, l in enumerate(lines) if l.startswith('(":" denotes')), None)
    if anchor is None or anchor + 3 >= len(lines):
        raise RuntimeError("USalign produced no alignment block")
    seq_q, _marks, seq_r = lines[anchor + 1], lines[anchor + 2], lines[anchor + 3]
    if len(seq_q) != len(seq_r):
        raise RuntimeError(f"ragged alignment block {len(seq_q)} vs {len(seq_r)}")
    pairs, qi, ri = [], 0, 0
    for cq, cr in zip(seq_q, seq_r):
        q_res = cq != "-"
        r_res = cr != "-"
        if q_res and r_res:
            pairs.append((qi, ri))
        qi += q_res
        ri += r_res
    return pairs


def element_spans(runs, max_he):
    """[(start, end)] in reference residue indices, for the elements the model actually saw."""
    keep = [i for i, (t, _) in enumerate(runs) if t in (DSSP_HELIX, DSSP_STRAND)]
    k = min(len(keep), max_he)
    starts, acc = [], 0
    for _t, n in runs:
        starts.append(acc)
        acc += n
    return [(starts[keep[e]], starts[keep[e]] + runs[keep[e]][1]) for e in range(k)]


def build_alignment(q_graph, r_graph, r_runs, max_he, tmpdir):
    """Binary L x T alignment plus Q, for one (query, reference) pair."""
    pq = os.path.join(tmpdir, "q.pdb")
    pr = os.path.join(tmpdir, "r.pdb")
    qmap = write_ca_pdb(q_graph["coords"], q_graph.get("coord_mask"), pq)
    rmap = write_ca_pdb(r_graph["coords"], r_graph.get("coord_mask"), pr)
    if not qmap or not rmap:
        raise RuntimeError("a chain had no resolved CA atoms")

    spans = element_spans(r_runs, max_he)
    L = int(q_graph["coords"].shape[0])
    T = len(spans)
    A = torch.zeros(L, T, dtype=torch.uint8)
    if T == 0:
        return A, 0

    # reference residue -> element, -1 where the residue is in a loop
    r_len = int(r_graph["coords"].shape[0])
    res_to_elem = torch.full((r_len,), -1, dtype=torch.long)
    for e, (a, b) in enumerate(spans):
        res_to_elem[a:min(b, r_len)] = e

    for q_w, r_w in usalign_pairs(pq, pr):
        gi = qmap[q_w]
        gj = rmap[r_w]
        e = int(res_to_elem[gj])
        if e >= 0:
            A[gi, e] = 1
    Q = int((A.sum(dim=1) > 0).sum())
    return A, Q


def load_graph(processed_dir, stem, manifest):
    from proteinfoundation.datasets.pdb_data import _processed_path_sharded

    p = _processed_path_sharded(pathlib.Path(processed_dir), stem, manifest)
    if not p.exists():
        raise FileNotFoundError(str(p))
    g = torch.load(p, map_location="cpu", weights_only=False)
    return {"coords": g.coords, "coord_mask": getattr(g, "coord_mask", None)}


def self_test(processed_dir, manifest, index, stem, max_he=64):
    """A chain aligned to ITSELF must give a near-perfect, near-diagonal alignment.

    This is the gate: it catches an index-map bug, a USalign parse bug, or an element-span
    off-by-one, none of which would be visible from a cross-chain number alone.
    """
    row = index["_id_to_row"][stem]
    runs = index["runs"](row)
    g = load_graph(processed_dir, stem, manifest)
    with tempfile.TemporaryDirectory() as td:
        A, Q = build_alignment(g, g, runs, max_he, td)
    spans = element_spans(runs, max_he)
    covered = sum(b - a for a, b in spans)
    ok_rows = int((A.sum(dim=1) > 1).sum())
    print(f"  self-test {stem}: L={A.shape[0]} T={A.shape[1]} Q={Q} "
          f"residues_in_elements={covered} rows_with_multiple_elements={ok_rows}")
    return Q, covered, ok_rows
