"""A DROPPED topology reference must be a variable-length, fully-masked scratch pad.

User 2026-09-10: the reference used to collapse to ONE mask token when drop_prob fired. Because the
MLM head can write into the reference, the model should instead get a realistically-sized fully
masked reference to use as a scratch pad -- with the SSE MLM loss DISABLED for those samples.

What must hold:
  1. a dropped reference has MANY elements, all MASK, with a length inside drop_ref_len_range;
  2. its positions are real and spread across the query (not all zero);
  3. NO MLM target is emitted, so the trainer's `sel = m_tgt > 1` selects nothing;
  4. no alignment ground truth is emitted either;
  5. the length varies from sample to sample, and is INDEPENDENT of the chain;
  6. (1, 1) reproduces the old single-token behaviour, so the mechanism is disableable.
"""

import os
import sys
import tempfile

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import MASK_TOKEN
from proteinfoundation.datasets.topology_reference import ALIGN_NONE, TopologyReferenceTransform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_synthetic_reference import build_toy_index, L_A  # reuse the toy index

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def graph(stem, L):
    g = Data()
    g.coords = torch.zeros(L, 37, 3)
    g.protein_id = stem
    return g


def main():
    with tempfile.TemporaryDirectory() as td:
        idx = os.path.join(td, "toy_index.pt")
        build_toy_index(idx)

        def make(**kw):
            kw.setdefault("reference_source", "synthetic")
            kw.setdefault("tm_range", (0.5, 0.9))
            t = TopologyReferenceTransform(index_path=idx, seed=0, **kw)
            t._ensure_loaded()
            return t

        # drop_prob 1.0 -> every sample is the dropped case
        t = make(drop_prob=1.0, drop_ref_len_range=(5, 39))
        lens = []
        for i in range(60):
            g = t(graph("aaaa_A", L_A))
            lens.append(int(g.topology_he_tokens.numel()))
        n = lens[0]
        g = t(graph("aaaa_A", L_A))
        T = int(g.topology_he_tokens.numel())

        check("dropped reference has MANY elements, not 1", min(lens) > 1, f"min {min(lens)}")
        check("length stays inside drop_ref_len_range", 5 <= min(lens) and max(lens) <= 39,
              f"{min(lens)}-{max(lens)}")
        check("length VARIES across samples", len(set(lens)) > 1, f"{len(set(lens))} distinct")
        check("every token is MASK", bool((g.topology_he_tokens == MASK_TOKEN).all()))
        check("positions are spread, not all zero",
              float(g.topology_he_pos_raw.max()) > 0 and g.topology_he_pos_raw.numel() == T)
        check("positions are increasing", bool((g.topology_he_pos_raw.diff() > 0).all()) if T > 1 else True)
        check("last position is near the query length",
              0.5 * L_A <= float(g.topology_he_pos_raw.max()) <= float(L_A),
              f"{float(g.topology_he_pos_raw.max()):.1f} of {L_A}")
        check("NO MLM targets (all zero -> `m_tgt > 1` selects nothing)",
              bool((g.topology_he_tokens_target == 0).all()) and int((g.topology_he_tokens_target > 1).sum()) == 0)
        check("no alignment ground truth", bool((g.ref_align_target == ALIGN_NONE).all()))
        check("he_feat / he_contact match the element count",
              tuple(g.topology_he_feat.shape[:2]) == (T, T) and tuple(g.topology_he_contact.shape) == (T, T))
        check("topology_missing_ref is 0 for a DROP (not a coverage bug)", int(g.topology_missing_ref) == 0)

        # independence: a different chain must not change the drawn length distribution
        t2 = make(drop_prob=1.0, drop_ref_len_range=(5, 39))
        lens_b = [int(t2(graph("bbbb_B", 20)).topology_he_tokens.numel()) for _ in range(60)]
        check("length is drawn independently of the chain (no leak of its own topology)",
              min(lens_b) >= 5 and max(lens_b) <= 39 and len(set(lens_b)) > 1,
              f"other chain: {min(lens_b)}-{max(lens_b)}")

        # disableable
        t3 = make(drop_prob=1.0, drop_ref_len_range=(1, 1))
        g3 = t3(graph("aaaa_A", L_A))
        check("(1, 1) reproduces the old single-token behaviour", int(g3.topology_he_tokens.numel()) == 1)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
