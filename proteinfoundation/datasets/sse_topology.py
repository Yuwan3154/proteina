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
