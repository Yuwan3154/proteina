# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Compressed secondary-structure-element (SSE) alphabet for topology conditioning.

Run-length-compresses a per-residue DSSP assignment into a short sequence of (type, length-bucket)
tokens, following the two-tier scheme of Lin & Ahnert, "Millisecond Prediction of Protein Contact
Maps from Amino Acid Sequences" (bioRxiv 10.64898/2026.03.15.711852, Method 4.2): short segments
get an exact token, longer ones are binned, and the compression is ~13x.

Three deliberate departures from that paper, each a recorded decision rather than an inference:
  * loops are tokenized as a third type, because this model must fill a fixed L-residue chain and
    therefore needs inter-element spacing, whereas the paper predicts an SSE-by-SSE map;
  * everything above ``catch_all_above`` collapses into one bucket per type (the paper defines a
    step-3 tier above 30 with no stated upper limit);
  * length 11 folds into the first bin, closing a gap the paper leaves between its exact tier
    (2-10) and its binned tier (12-30).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

# DSSP index convention used across this repo (see DSSPTargetTransform): 0=loop, 1=helix, 2=strand.
# -1 marks residues with an incomplete backbone and breaks a run.
DSSP_LOOP, DSSP_HELIX, DSSP_STRAND = 0, 1, 2
SSE_TYPES = (DSSP_LOOP, DSSP_HELIX, DSSP_STRAND)

PAD_TOKEN = 0
MASK_TOKEN = 1  # whole-reference dropout, for classifier-free guidance
N_SPECIAL_TOKENS = 2


class SSEAlphabet:
    """Maps (dssp_type, run_length) to a token id and back.

    Args:
        exact_max: longest run that still gets its own exact token.
        bin_step: bucket width applied between ``exact_max + 1`` and ``catch_all_above``.
        catch_all_above: runs longer than this collapse into a single per-type bucket.
        min_len: runs shorter than this are dropped from the encoding.
        types: DSSP indices to encode, in the order their token blocks are laid out.
    """

    def __init__(
        self,
        exact_max: int = 10,
        bin_step: int = 2,
        catch_all_above: int = 30,
        min_len: int = 1,
        types: Sequence[int] = SSE_TYPES,
    ):
        if min_len > exact_max:
            raise ValueError(f"min_len={min_len} exceeds exact_max={exact_max}")
        if catch_all_above < exact_max:
            raise ValueError(f"catch_all_above={catch_all_above} below exact_max={exact_max}")
        self.exact_max = exact_max
        self.bin_step = bin_step
        self.catch_all_above = catch_all_above
        self.min_len = min_len
        self.types = tuple(types)

        self.exact_lengths = list(range(min_len, exact_max + 1))
        # Bin edges are the INCLUSIVE upper bound of each bucket, so a run folds into the first
        # edge that is >= its length -- this is what closes the paper's gap at length 11.
        self.bin_edges = list(range(exact_max + 1 + (bin_step - 1), catch_all_above + 1, bin_step))
        if not self.bin_edges or self.bin_edges[-1] < catch_all_above:
            self.bin_edges.append(catch_all_above)
        self.slots_per_type = len(self.exact_lengths) + len(self.bin_edges) + 1

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL_TOKENS + len(self.types) * self.slots_per_type

    def _slot(self, length: int) -> int:
        if length <= self.exact_max:
            return length - self.min_len
        for k, edge in enumerate(self.bin_edges):
            if length <= edge:
                return len(self.exact_lengths) + k
        return self.slots_per_type - 1  # catch-all

    def token(self, dssp_type: int, length: int) -> int:
        if dssp_type not in self.types:
            raise ValueError(f"type {dssp_type} not in alphabet types {self.types}")
        return (
            N_SPECIAL_TOKENS
            + self.types.index(dssp_type) * self.slots_per_type
            + self._slot(length)
        )

    def decode(self, token: int) -> Tuple[int, str]:
        """Return (dssp_type, human-readable length range) for a non-special token."""
        if token < N_SPECIAL_TOKENS:
            return (-1, "special")
        idx = token - N_SPECIAL_TOKENS
        t = self.types[idx // self.slots_per_type]
        slot = idx % self.slots_per_type
        if slot < len(self.exact_lengths):
            return t, str(self.exact_lengths[slot])
        slot -= len(self.exact_lengths)
        if slot < len(self.bin_edges):
            lo = self.exact_max + 1 if slot == 0 else self.bin_edges[slot - 1] + 1
            return t, f"{lo}-{self.bin_edges[slot]}"
        return t, f">{self.catch_all_above}"

    def runs_to_tokens(self, runs: Sequence[Tuple[int, int]]) -> List[int]:
        return [self.token(t, n) for t, n in runs if t in self.types and n >= self.min_len]



def dssp_to_runs(dssp: torch.Tensor, min_len: int = 1) -> List[Tuple[int, int]]:
    """Run-length-compress a per-residue DSSP assignment into (type, length) pairs.

    Residues labelled -1 (incomplete backbone) terminate the current run and contribute nothing,
    so an unresolved stretch does not silently fuse the elements on either side of it.
    """
    runs: List[Tuple[int, int]] = []
    values = dssp.tolist()
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        if values[i] >= 0 and (j - i) >= min_len:
            runs.append((int(values[i]), j - i))
        i = j
    return runs


def perturb_runs(
    runs: Sequence[Tuple[int, int]],
    sigma_frac: float,
    mutate_prob: float,
    generator: torch.Generator,
    min_len: int = 1,
    max_len: int = 512,
) -> List[Tuple[int, int]]:
    """Jitter element lengths, keeping types and order intact.

    Sigma scales with each element's own length so a 4-residue strand and a 30-residue helix are
    perturbed comparably in relative terms. Perturbation happens in RESIDUE space and the caller
    re-tokenizes afterwards, so a jitter that stays inside a bucket correctly produces no change
    to the reference -- which is the point: within-bucket precision was never claimed.
    """
    out: List[Tuple[int, int]] = []
    for t, n in runs:
        if float(torch.rand(1, generator=generator, device=generator.device)) >= mutate_prob:
            out.append((t, n))
            continue
        sigma = max(sigma_frac * n, 1e-6)
        delta = float(torch.randn(1, generator=generator, device=generator.device)) * sigma
        out.append((t, int(min(max(round(n + delta), min_len), max_len))))
    return out


def encode_topology(
    dssp: torch.Tensor,
    alphabet: SSEAlphabet,
    sigma_frac: float = 0.0,
    mutate_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """DSSP assignment -> topology token sequence, optionally augmented."""
    runs = dssp_to_runs(dssp, min_len=alphabet.min_len)
    if mutate_prob > 0.0 and sigma_frac > 0.0:
        if generator is None:
            raise ValueError("a torch.Generator is required when augmentation is enabled")
        runs = perturb_runs(
            runs, sigma_frac, mutate_prob, generator, min_len=alphabet.min_len
        )
    return torch.tensor(alphabet.runs_to_tokens(runs), dtype=torch.long)


def runs_to_spans(runs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Residue interval [start, end) covered by each run, in the source chain's own coordinates."""
    spans = []
    pos = 0
    for _, n in runs:
        spans.append((pos, pos + n))
        pos += n
    return spans


def element_positions(
    runs: Sequence[Tuple[int, int]], target_len: Optional[int] = None
) -> torch.Tensor:
    """Residue-space midpoint of each element, optionally rescaled onto another chain's length.

    Query residue index and topology element index are different coordinate systems, so a relative
    position between them is only meaningful once both are expressed on the same grid. Rescaling by
    ``target_len / template_len`` is the mixed-resolution RoPE trick: equal physical distances along
    the two chains then produce equal positional offsets.
    """
    spans = runs_to_spans(runs)
    if not spans:
        return torch.zeros(0, dtype=torch.float32)
    mids = torch.tensor([(a + b) / 2.0 for a, b in spans], dtype=torch.float32)
    template_len = spans[-1][1]
    if target_len is not None and template_len > 0:
        mids = mids * (float(target_len) / float(template_len))
    return mids


def sse_contact_reference(
    contact_map: torch.Tensor,
    runs: Sequence[Tuple[int, int]],
    keep_types: Sequence[int] = (DSSP_HELIX, DSSP_STRAND),
) -> Tuple[torch.Tensor, List[int]]:
    """Coarse-grain a residue-level contact map onto an element-by-element one.

    Returns the [T, T] element contact map and the indices (into ``runs``) of the kept elements.
    Max-pooling over each element pair matches the paper's notion of an SSE contact: two elements
    are in contact if ANY of their residues are, rather than on average.
    """
    spans = runs_to_spans(runs)
    keep = [i for i, (t, _) in enumerate(runs) if t in keep_types]
    T = len(keep)
    if T == 0:
        return contact_map.new_zeros((0, 0)), keep

    # Two scatter-reduce passes (rows, then columns) instead of a T^2 Python loop over slices:
    # the loop version costs ~20ms per chain, which is hours across the full training set.
    L = contact_map.shape[0]
    elem = torch.full((L,), -1, dtype=torch.long, device=contact_map.device)
    for a, ia in enumerate(keep):
        s, e = spans[ia]
        elem[s:e] = a
    valid = elem >= 0
    sub = contact_map[valid][:, valid]
    row = elem[valid]

    rows = contact_map.new_zeros((T, sub.shape[1]))
    rows.scatter_reduce_(0, row[:, None].expand(-1, sub.shape[1]), sub, reduce="amax")
    out = contact_map.new_zeros((T, T))
    out.scatter_reduce_(1, row[None, :].expand(T, -1), rows, reduce="amax")
    return out, keep


# ── Element-pair featurization ────────────────────────────────────────────────────────────────
#
# The 2D reference started as an outer concat of the two element embeddings plus a binary contact,
# which describes WHICH elements touch but nothing about HOW. Two richer descriptions are available
# and are both computed here so they can be compared:
#
#   * circuit topology -- the Series / Parallel / Cross relation between chain loops (Mashaghi et
#     al., "Circuit topology of proteins and nucleic acids", Structure 2014). Every contacting
#     element pair (s, t) spans a loop along the chain, and its relation to every other contact is
#     one of disjoint (SERIES), nested (PARALLEL), or interlocked (CROSS). Note the naming: in this
#     framework Series is the non-intersecting case and Parallel is the encompassed one, which is
#     the opposite of the everyday reading of those two words. These relations, not the contact set
#     alone, are what distinguish folds that share a contact count.
#   * secondary-structure contact/proximity -- how strongly and how closely two elements pack:
#     the fraction of their residue pairs in contact, their closest and mean CA-CA distance, and
#     the number of residues separating them along the chain.
#   * relative orientation -- the cosine between the two elements' axes, which is what separates
#     parallel from antiparallel beta pairing and fixes helix packing angles. Contacts and
#     distances cannot express it: two strands at +1 and -1 can pack identically closely.
#
# Channels split by WHERE they can be computed. The three structural ones need residue-level data
# (the full contact map and the coordinates) that the dataloader never sees, so they are stored in
# the precomputed index. The rest are derived from the element contact map and the run lengths,
# both of which the transform already holds -- and the sequence gap MUST be derived there, because
# augmentation perturbs element lengths and a stored gap would describe the unperturbed topology.
PAIR_FEATURE_NAMES = (
    "contact_max",
    "contact_frac",
    "min_ca_dist",
    "mean_ca_dist",
    "orientation_cos",
    "circuit_series",
    "circuit_parallel_contains",
    "circuit_parallel_inside",
    "circuit_cross",
    "seq_gap",
)
N_PAIR_FEATURES = len(PAIR_FEATURE_NAMES)
STRUCTURAL_PAIR_FEATURES = ("contact_frac", "min_ca_dist", "mean_ca_dist", "orientation_cos")
ORIENTATION_PAIR_FEATURES = ("orientation_cos",)
CIRCUIT_PAIR_FEATURES = (
    "circuit_series",
    "circuit_parallel_contains",
    "circuit_parallel_inside",
    "circuit_cross",
)
PROXIMITY_PAIR_FEATURES = ("contact_frac", "min_ca_dist", "mean_ca_dist", "seq_gap")

# Which channels of the 2D SSE reference the pair track sees. "contact" is the original
# behaviour (which elements touch, nothing more); the other three add the descriptions of HOW they
# touch that sse_topology computes, and exist so the two can be compared empirically.
PAIR_FEATURE_MODES: Dict[str, Tuple[str, ...]] = {
    "contact": ("contact_max",),
    "circuit": ("contact_max",) + CIRCUIT_PAIR_FEATURES,
    "proximity": ("contact_max",) + PROXIMITY_PAIR_FEATURES,
    "both": PAIR_FEATURE_NAMES,
}

# Kept here rather than in a model file so a model can import it without pulling in the
# neural-network package, which imports cuequivariance and therefore needs a GPU driver present.

CA_ATOM_INDEX = 1  # ATOM_NUMBERING order, which is what the stored .pt uses


def pair_feature_indices(names: Sequence[str]) -> List[int]:
    return [PAIR_FEATURE_NAMES.index(n) for n in names]


def circuit_topology_features(contact: torch.Tensor) -> torch.Tensor:
    """[T, T] element contact map -> [T, T, 4] circuit-topology relation counts.

    Each element pair (s, t) delimits a loop along the chain. Its relation to another contact
    (u, v) is SERIES when the two loops are disjoint, PARALLEL when one encompasses the other
    (split here by which encompasses which, which the single P class does not distinguish), and
    CROSS when they interlock. Counting each relation over all contacts gives a per-pair
    fingerprint of how that pair sits within the fold rather than merely whether it touches.

    Two departures from the published definition, both deliberate: loops sharing an endpoint are
    the "concerted" subclass, folded here into whichever of Series or Parallel they border on so
    that every pair falls in exactly one of four channels; and a contact is not counted against
    itself, since "this pair is a contact" is already the contact_max channel. Counts are divided
    by the number of contacts, which keeps the channel scale-free across chains of very different
    sizes.
    """
    T = contact.shape[0]
    out = contact.new_zeros((T, T, 4))
    if T == 0:
        return out
    u, v = torch.triu_indices(T, T, offset=1, device=contact.device)
    sel = contact[u, v] > 0
    u, v = u[sel], v[sel]
    K = u.numel()
    if K == 0:
        return out

    a = torch.arange(T, device=contact.device)
    lo = torch.minimum(a[:, None], a[None, :])[..., None]  # [T, T, 1]
    hi = torch.maximum(a[:, None], a[None, :])[..., None]
    u, v = u[None, None, :], v[None, None, :]

    same = (u == lo) & (v == hi)  # a contact compared against its own loop
    series = (v <= lo) | (u >= hi)
    contains = (lo <= u) & (v <= hi) & ~same & ~series
    inside = (u <= lo) & (hi <= v) & ~same & ~series & ~contains
    cross = ~(series | contains | inside | same)

    for c, rel in enumerate((series, contains, inside, cross)):
        out[..., c] = rel.sum(dim=-1).to(out.dtype) / float(K)
    return out


def sse_sequence_gap(runs: Sequence[Tuple[int, int]], keep: Sequence[int]) -> torch.Tensor:
    """[T, T] residues lying strictly between each pair of kept elements.

    Derived from the run lengths in force at call time, so it tracks length augmentation instead
    of describing the template's original spacing.
    """
    T = len(keep)
    out = torch.zeros((T, T), dtype=torch.float32)
    if T == 0:
        return out
    spans = runs_to_spans(runs)
    starts = torch.tensor([spans[i][0] for i in keep], dtype=torch.float32)
    ends = torch.tensor([spans[i][1] for i in keep], dtype=torch.float32)
    gap = torch.maximum(starts[None, :] - ends[:, None], starts[:, None] - ends[None, :])
    return gap.clamp(min=0.0)


def sse_structural_pair_features(
    contact_map: torch.Tensor,
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    runs: Sequence[Tuple[int, int]],
    keep: Sequence[int],
) -> torch.Tensor:
    """[T, T, 4] contact fraction, closest/mean CA-CA distance and axis cosine per element pair.

    Args:
        contact_map: [L, L] binarised residue contact map.
        coords: [L, n_atoms, 3] as stored in the .pt (ATOM_NUMBERING order).
        coord_mask: [L, n_atoms] validity flags for those coordinates.
        runs, keep: the run-length decomposition and the indices retained in the 2D reference.

    Residues without a resolved CA are excluded from the distance reductions; a pair with no
    resolved CA on either side gets distance 0, which the contact channels already mark as
    uninformative.
    """
    T = len(keep)
    out = torch.zeros((T, T, len(STRUCTURAL_PAIR_FEATURES)), dtype=torch.float32)
    if T == 0:
        return out
    spans = runs_to_spans(runs)
    L = contact_map.shape[0]

    elem = torch.full((L,), -1, dtype=torch.long)
    for a, ia in enumerate(keep):
        s, e = spans[ia]
        elem[s:e] = a

    # Element-block reductions run as two scatter passes (rows, then columns) rather than a T^2
    # loop over slices, the same reason sse_contact_reference is written this way.
    valid = elem >= 0
    row = elem[valid]
    n_res = torch.zeros(T).scatter_add_(0, row, torch.ones(row.numel()))
    pair_res = n_res[:, None] * n_res[None, :]

    sub = contact_map[valid][:, valid].float()
    out[..., 0] = _block_reduce(sub, row, T, "sum") / pair_res.clamp(min=1.0)

    ca_ok = valid & (coord_mask[:, CA_ATOM_INDEX] > 0.5)
    if ca_ok.any():
        row_ca = elem[ca_ok]
        n_ca = torch.zeros(T).scatter_add_(0, row_ca, torch.ones(row_ca.numel()))
        d = torch.cdist(
            coords[ca_ok, CA_ATOM_INDEX, :].float()[None],
            coords[ca_ok, CA_ATOM_INDEX, :].float()[None],
        )[0]
        has_ca = (n_ca[:, None] > 0) & (n_ca[None, :] > 0)
        dmin = _block_reduce(d, row_ca, T, "amin")
        out[..., 1] = torch.where(has_ca, dmin, torch.zeros_like(dmin))
        dmean = _block_reduce(d, row_ca, T, "sum") / (n_ca[:, None] * n_ca[None, :]).clamp(min=1.0)
        out[..., 2] = torch.where(has_ca, dmean, torch.zeros_like(dmean))
        axes = _element_axes(coords[:, CA_ATOM_INDEX, :].float(), elem, ca_ok, T)
        out[..., 3] = (axes @ axes.T).clamp(-1.0, 1.0)
    # All three channels are symmetric by definition, but torch.cdist takes a matmul path for
    # larger inputs that is asymmetric at ~1e-5, and the float16 storage then amplifies that to a
    # whole ulp (0.03 A) whenever a mirrored pair straddles a rounding boundary. Averaging the two
    # halves restores the invariant exactly, at float32 precision, before it is ever cast.
    return 0.5 * (out + out.transpose(0, 1))


def _element_axes(
    ca: torch.Tensor, elem: torch.Tensor, ca_ok: torch.Tensor, T: int
) -> torch.Tensor:
    """[T, 3] unit axis per element, oriented from its first residue toward its last.

    The axis is the least-squares line through the element's CA atoms (its first principal
    component), which follows a curved helix better than a bare end-to-end vector while agreeing
    with it on a straight strand. Signing it N-to-C is what makes the pairwise cosine distinguish
    parallel from antiparallel rather than collapsing both onto |cos|. An element with fewer than
    two resolved CA atoms has no definable axis and is left at zero, giving cosine 0 everywhere.
    """
    axes = torch.zeros(T, 3)
    for a in range(T):
        m = (elem == a) & ca_ok
        if int(m.sum()) < 2:
            continue
        x = ca[m]
        v = torch.linalg.svd(x - x.mean(0, keepdim=True), full_matrices=False).Vh[0]
        if torch.dot(v, x[-1] - x[0]) < 0:
            v = -v
        axes[a] = v / v.norm().clamp(min=1e-8)
    return axes


def _block_reduce(m: torch.Tensor, row: torch.Tensor, T: int, reduce: str) -> torch.Tensor:
    """Reduce an [n, n] residue matrix onto its [T, T] element blocks."""
    init = float("inf") if reduce == "amin" else 0.0
    rows = torch.full((T, m.shape[1]), init, dtype=m.dtype)
    rows.scatter_reduce_(0, row[:, None].expand(-1, m.shape[1]), m, reduce=reduce)
    out = torch.full((T, T), init, dtype=m.dtype)
    out.scatter_reduce_(1, row[None, :].expand(T, -1), rows, reduce=reduce)
    return torch.nan_to_num(out, posinf=0.0)


def assemble_pair_features(
    contact: torch.Tensor,
    structural: torch.Tensor,
    runs: Sequence[Tuple[int, int]],
    keep: Sequence[int],
) -> torch.Tensor:
    """Stack every channel of PAIR_FEATURE_NAMES into one [T, T, N_PAIR_FEATURES] tensor."""
    circuit = circuit_topology_features(contact)
    gap = sse_sequence_gap(runs, keep)
    return torch.cat(
        [contact[..., None].float(), structural.float(), circuit.float(), gap[..., None]], dim=-1
    )


def alphabet_summary(alphabet: SSEAlphabet) -> Dict[str, object]:
    names = {DSSP_LOOP: "loop", DSSP_HELIX: "helix", DSSP_STRAND: "strand"}
    return {
        "vocab_size": alphabet.vocab_size,
        "slots_per_type": alphabet.slots_per_type,
        "exact": f"{alphabet.min_len}-{alphabet.exact_max}",
        "bins": alphabet.bin_edges,
        "catch_all_above": alphabet.catch_all_above,
        "types": [names.get(t, str(t)) for t in alphabet.types],
    }
