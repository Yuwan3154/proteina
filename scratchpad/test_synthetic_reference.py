"""Gate for TopologyReferenceTransform's synthetic-reference mode and the new augmentations.

Built on a TOY index in the synthetic schema (natives + template rows grouped per chain), so it
runs anywhere. What must hold:
  1. synthetic draws come only from the query's OWN template rows with TM inside tm_range -- never
     the native row, never an out-of-range row;
  2. a chain with no in-range template falls back to MASK AND flags topology_missing_ref = 1;
  3. rates 0 change nothing (targets all zero); token_mask_prob = 1 masks every token and stores the
     originals; type_mutate_prob = 1 flips every helix/strand type;
  4. a helix+strand alphabet has vocab 44, and topology_tokens == topology_he_tokens;
  5. ref_align_target is the residue-axis element index, -1 where unaligned or beyond the he cap;
  6. self_reference -> the native row, nonself_reference -> an in-range template row.
"""

import os
import sys
import tempfile

import torch

from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.sse_topology import (
    DSSP_HELIX,
    DSSP_STRAND,
    MASK_TOKEN,
    N_PAIR_FEATURES,
)
from proteinfoundation.datasets.topology_reference import (
    ALIGN_NONE,
    MASK_REF_ID,
    TopologyReferenceTransform,
)

PASS, FAIL = [], []
N_STRUCT = 4
L_A, L_B = 30, 20
# native: loop3 helix10 loop2 strand8 loop7 (= 30 residues, 2 helix/strand elements)
RUNS_A = [(0, 3), (1, 10), (0, 2), (2, 8), (0, 7)]
RUNS_B = [(0, 5), (1, 8), (0, 7)]


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def build_toy_index(path):
    ids = ["aaaa_A", "aaaa_A@r300#0", "aaaa_A@r250#1", "aaaa_A@r200#2", "bbbb_B"]
    rows_runs = [RUNS_A, RUNS_A, RUNS_A, RUNS_A, RUNS_B]
    he_sizes = [2, 2, 2, 2, 1]
    runs_flat, runs_offset = [], [0]
    he_flat, he_offset = [], [0]
    feat_flat, feat_offset = [], [0]
    align_flat, align_offset = [], [0]
    # template alignment: residues 3..12 -> element 0, 15..22 -> element 1, rest unaligned
    tpl_align = torch.full((L_A,), ALIGN_NONE, dtype=torch.int16)
    tpl_align[3:13] = 0
    tpl_align[15:23] = 1
    for r, (runs, T) in enumerate(zip(rows_runs, he_sizes)):
        runs_flat += runs
        runs_offset.append(len(runs_flat))
        he_flat += [1] * (T * T)
        he_offset.append(len(he_flat))
        feat_flat += [0.0] * (T * T * N_STRUCT)
        feat_offset.append(len(feat_flat))
        if r in (1, 2, 3):
            align_flat += tpl_align.tolist()
        align_offset.append(len(align_flat))
    index = {
        "ids": ids,
        "cluster_of": torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32),
        "members_flat": torch.tensor([1, 2, 3], dtype=torch.int32),   # template rows of chain 0 only
        "members_offset": torch.tensor([0, 3, 3], dtype=torch.int64),
        "seq_hash": torch.tensor([11, 11, 11, 11, 22], dtype=torch.int64),
        "runs_flat": torch.tensor(runs_flat, dtype=torch.int16),
        "runs_offset": torch.tensor(runs_offset, dtype=torch.int64),
        "he_offset": torch.tensor(he_offset, dtype=torch.int64),
        "he_size": torch.tensor(he_sizes, dtype=torch.int16),
        "he_flat": torch.tensor(he_flat, dtype=torch.uint8),
        "feat_offset": torch.tensor(feat_offset, dtype=torch.int64),
        "feat_flat": torch.tensor(feat_flat, dtype=torch.float16),
        # derived, never hardcoded: the width changed 10 -> 8 when the CA-CA channels were dropped
        "pair_feature_mean": torch.zeros(N_PAIR_FEATURES),
        "pair_feature_std": torch.ones(N_PAIR_FEATURES),
        "row_tm": torch.tensor([1.0, 0.45, 0.6, 0.85, 1.0], dtype=torch.float16),
        "row_is_native": torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool),
        "row_rewind": torch.tensor([0, 300, 250, 200, 0], dtype=torch.int16),
        "align_flat": torch.tensor(align_flat, dtype=torch.int16),
        "align_offset": torch.tensor(align_offset, dtype=torch.int64),
        "min_len": 1,
        "contact_def": "CB<=8.0A, CA fallback",
    }
    torch.save(index, path)


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
            kw.setdefault("drop_prob", 0.0)
            kw.setdefault("reference_source", "synthetic")
            kw.setdefault("tm_range", (0.5, 0.9))
            t = TopologyReferenceTransform(index_path=idx, seed=0, **kw)
            t._ensure_loaded()
            return t

        # ---- 1. draws stay inside the chain's own in-range template rows ----
        t = make()
        refs = {t(graph("aaaa_A", L_A)).topology_ref_id for _ in range(200)}
        check("synthetic draws only from in-range template rows", refs == {"aaaa_A@r250#1", "aaaa_A@r200#2"}, str(refs))
        check("...never the native, never TM 0.45", "aaaa_A" not in refs and "aaaa_A@r300#0" not in refs)
        g = t(graph("aaaa_A", L_A))
        check("topology_missing_ref == 0 on a served reference", int(g.topology_missing_ref) == 0)
        check("ref_align_target has the residue length", tuple(g.ref_align_target.shape) == (L_A,))
        check("ref_align_target values are element indices or ALIGN_NONE",
              set(g.ref_align_target.tolist()) == {ALIGN_NONE, 0, 1})
        check("aligned residues 3..12 -> element 0, 15..22 -> element 1",
              bool((g.ref_align_target[3:13] == 0).all()) and bool((g.ref_align_target[15:23] == 1).all()))

        # ---- 2. no in-range template -> MASK + loud flag ----
        g = t(graph("bbbb_B", L_B))
        check("chain without templates -> MASK sentinel", g.topology_ref_id == MASK_REF_ID)
        check("...and topology_missing_ref == 1", int(g.topology_missing_ref) == 1)
        check("...and ref_align_target all NONE with residue length",
              tuple(g.ref_align_target.shape) == (L_B,) and bool((g.ref_align_target == ALIGN_NONE).all()))
        check("...and topology_he_tokens_target present", hasattr(g, "topology_he_tokens_target"))
        t_narrow = make(tm_range=(0.95, 0.99))
        g = t_narrow(graph("aaaa_A", L_A))
        check("templated chain with an EMPTY range -> MASK + flag", g.topology_ref_id == MASK_REF_ID and int(g.topology_missing_ref) == 1)
        g = t(graph("zzzz_Z", 12))
        check("chain absent from the index -> MASK + flag", g.topology_ref_id == MASK_REF_ID and int(g.topology_missing_ref) == 1)

        # ---- 3. augmentation rates ----
        t0 = make(mutate_prob=0.0, sigma_frac=0.0)
        g = t0(graph("aaaa_A", L_A))
        base_tokens = g.topology_he_tokens.clone()
        check("rate 0: no masked-token targets", bool((g.topology_he_tokens_target == 0).all()))
        check("rate 0: no MASK tokens", bool((g.topology_he_tokens != MASK_TOKEN).all()))
        tm = make(mutate_prob=0.0, sigma_frac=0.0, token_mask_prob=1.0)
        g = tm(graph("aaaa_A", L_A))
        check("token_mask_prob 1: every he token is MASK", bool((g.topology_he_tokens == MASK_TOKEN).all()))
        check("token_mask_prob 1: targets are the pre-mask tokens", torch.equal(g.topology_he_tokens_target, base_tokens))
        check("token_mask_prob 1: positions untouched", g.topology_he_pos_raw.numel() == base_tokens.numel())
        tt = make(mutate_prob=0.0, sigma_frac=0.0, type_mutate_prob=1.0)
        g = tt(graph("aaaa_A", L_A))
        types_base = [tt.alphabet.decode(int(x))[0] for x in base_tokens]
        types_mut = [tt.alphabet.decode(int(x))[0] for x in g.topology_he_tokens]
        check("type_mutate_prob 1: helix<->strand flipped for every element",
              types_base == [DSSP_HELIX, DSSP_STRAND] and types_mut == [DSSP_STRAND, DSSP_HELIX], f"{types_base}->{types_mut}")
        check("type mutation keeps element count", g.topology_he_tokens.numel() == base_tokens.numel())

        # ---- 4. helix+strand alphabet ----
        t44 = make(sse_types=(1, 2), mutate_prob=0.0, sigma_frac=0.0)
        check("sse_types (1,2) -> vocab 44", t44.alphabet.vocab_size == 44)
        g = t44(graph("aaaa_A", L_A))
        check("he-only alphabet: topology_tokens == topology_he_tokens", torch.equal(g.topology_tokens, g.topology_he_tokens))
        check("he-only alphabet: topology_pos == topology_he_pos", torch.equal(g.topology_pos, g.topology_he_pos))
        check("he-only alphabet: all tokens < 44", int(g.topology_he_tokens.max()) < 44)
        t65 = make(mutate_prob=0.0, sigma_frac=0.0)
        g65 = t65(graph("aaaa_A", L_A))
        check("default alphabet still emits loop tokens (5 runs -> 5 tokens)", g65.topology_tokens.numel() == 5)

        # ---- 5. alignment remap through the he cap ----
        tcap = make(max_topology_he_len=1, mutate_prob=0.0, sigma_frac=0.0)
        g = tcap(graph("aaaa_A", L_A))
        check("he cap 1: element 1 alignments become NONE, element 0 kept",
              bool((g.ref_align_target[15:23] == ALIGN_NONE).all()) and bool((g.ref_align_target[3:13] == 0).all()))

        # ---- 6. validation helpers ----
        sr = t.self_reference("aaaa_A", L_A)
        check("self_reference returns the native's own runs (2 he elements)", sr is not None and sr["topology_he_tokens"].numel() == 2)
        nr = t.nonself_reference("aaaa_A", L_A, seed=3)
        check("nonself_reference returns an in-range template", nr is not None and nr[1] in ("aaaa_A@r250#1", "aaaa_A@r200#2"), str(nr[1] if nr else None))
        check("nonself_reference is None without templates", t.nonself_reference("bbbb_B", L_B) is None)

        # ---- 7. cluster mode still works on the same file (regression of the old path) ----
        tc = TopologyReferenceTransform(index_path=idx, seed=0, drop_prob=0.0)
        tc._ensure_loaded()
        g = tc(graph("aaaa_A", L_A))
        check("cluster mode: members share seq_hash -> self fallback, flag 0",
              g.topology_ref_id == "aaaa_A" and int(g.topology_missing_ref) == 0)
        check("cluster mode: repr mentions source", "source=cluster" in repr(tc))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
