"""Is the Q x T alignment target recoverable from POSITION alone? (D18)

The review flagged that a synthetic template is a partial-diffusion variant of the query's OWN native
and so keeps the query's length and residue indexing. The model sees query residue i at position i
and element e at its own-chain midpoint he_pos_raw[e] -- on the SAME index scale -- so
`align_head` could score well by simply picking the nearest element, without learning any
query<->reference threading. If so, `train/align_precision_at_q` measures position, not threading.

This answers it OFFLINE, with no training: score every (query residue, element) cell by
-|i - he_pos_raw[e]| and evaluate it with the trainer's OWN precision@Q definition
(model_trainer_base.py:3035-3045): rank the valid cells, take the top Q where Q is the number of
query rows carrying a ground-truth element, and report the fraction of those that are true.

Numbers to compare against: the trained head reached 0.531 in the 1500-step sweep; the 2026-09-02
probe measured 0.737 for a trained probe on a fully trained trunk against a 0.383 left-alignment
floor. A positional baseline near 0.53 would mean the head learned nothing beyond position.
"""

import argparse
import random

import torch

from proteinfoundation.datasets.topology_reference import ALIGN_NONE, TopologyReferenceTransform
from torch_geometric.data import Data

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--eligible", required=True)
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--augment", type=int, default=1, help="1 = as training serves it")
a = ap.parse_args()

t = TopologyReferenceTransform(
    index_path=a.index, reference_source="synthetic", tm_range=(0.5, 0.9),
    sse_types=[1, 2], max_topology_len=128, max_topology_he_len=64,
    sigma_frac=0.15 if a.augment else 0.0, mutate_prob=0.3 if a.augment else 0.0,
    type_mutate_prob=0.1 if a.augment else 0.0, token_mask_prob=0.1 if a.augment else 0.0,
    drop_prob=0.0, seed=0,
)
t._ensure_loaded()
idx = t._index
stems = [l.strip() for l in open(a.eligible) if l.strip()]
random.Random(0).shuffle(stems)

prec_pos, prec_rand, qs, ts = [], [], [], []
for stem in stems:
    if len(prec_pos) >= a.n:
        break
    row = t._id_to_row.get(stem)
    if row is None:
        continue
    lo, hi = int(idx["runs_offset"][row]), int(idx["runs_offset"][row + 1])
    L = sum(int(x[1]) for x in idx["runs_flat"][lo:hi])
    if L <= 0:
        continue
    g = Data()
    g.coords = torch.zeros(L, 37, 3)
    g.protein_id = stem
    out = t(g)
    tgt = out.ref_align_target.long()            # [L] element index or ALIGN_NONE
    pos = out.topology_he_pos_raw.float()        # [T]
    T = int(pos.numel())
    if T == 0:
        continue
    aligned = (tgt >= 0) & (tgt < T)
    Q = int(aligned.sum())
    if Q == 0:
        continue

    # ground-truth cell matrix, exactly as the trainer builds it
    A = torch.zeros(L, T)
    A[aligned, tgt[aligned]] = 1.0

    i = torch.arange(L, dtype=torch.float32)[:, None]
    score = -(i - pos[None, :]).abs()             # nearest element by position
    top = torch.topk(score.reshape(-1), min(Q, L * T)).indices
    prec_pos.append(float(A.reshape(-1)[top].mean()))

    # chance level for the same Q and cell count
    prec_rand.append(Q / float(L * T))
    qs.append(Q)
    ts.append(T)

assert len(prec_pos) >= 100, f"VACUOUS: only {len(prec_pos)} chains scored"
p = torch.tensor(prec_pos)
r = torch.tensor(prec_rand)
print(f"chains scored: {len(prec_pos)}   (augment={bool(a.augment)})")
print(f"mean Q {sum(qs) / len(qs):.1f} residues, mean T {sum(ts) / len(ts):.1f} elements\n")
print(f"POSITIONAL baseline precision@Q : mean {p.mean():.3f}  median {p.median():.3f} "
      f"p5 {torch.quantile(p, 0.05):.3f}  p95 {torch.quantile(p, 0.95):.3f}")
print(f"chance level (Q / cells)        : mean {r.mean():.3f}")
print("\nreference points: trained head in the 1500-step sweep = 0.531;")
print("                  2026-09-02 probe on a trained trunk   = 0.737 (floor 0.383)")
