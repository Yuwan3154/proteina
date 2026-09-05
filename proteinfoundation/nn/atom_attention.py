"""AF3 atom-level attention for all-atom output (SI Algorithms 5 and 6).

⭐ Why this is cheap, which is the thing that makes all-atom affordable at all: attention is
SEQUENCE-LOCAL, not global. Each query atom attends only to a fixed window of keys -- AF3 uses
n_queries=32, n_keys=128, and Protenix, OpenFold3 and IntelliFold all use exactly those. So the cost
is O(N_atom * n_keys) rather than O(N_atom^2), and the encoder+decoder together are ~2.2 M of
Protenix's 203 M diffusion module (1.1%).

⛔ SCOPE: standard amino acids only, which is what our contact model handles. AF3's atom machinery
exists mainly to represent ligands and nucleic acids; for 20 canonical residues the per-atom
reference features come straight from the repo's own residue constants, so no new data is needed.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinfoundation.nn.af3_diffusion import DiffusionTransformerBlock

N_QUERIES = 32   # SI Alg. 5/6; identical in Protenix, OpenFold3, IntelliFold
N_KEYS = 128


def local_attention_mask(n_atom: int, device, n_queries: int = N_QUERIES, n_keys: int = N_KEYS):
    """Banded [N, N] bool mask: query i may attend to keys within its centred window.

    Built as an explicit band rather than as blocked gather/scatter. The band is the same
    computation; the blocked form is a memory optimisation that only pays off at the atom counts
    AF3 handles for large complexes, and it costs a great deal of index bookkeeping that is easy to
    get subtly wrong. At our sizes (<= 384 residues, so <= ~3000 atoms) the band is affordable and
    obviously correct. ⚠️ Revisit if atom counts grow.
    """
    idx = torch.arange(n_atom, device=device)
    centre = (idx // n_queries) * n_queries + n_queries // 2
    half = n_keys // 2
    d = (idx[None, :] - centre[:, None])
    return (d >= -half) & (d < half)


class AtomAttentionEncoder(nn.Module):
    """Atom features (+ optional noisy coords) -> atom repr, aggregated to per-token repr. SI Alg. 5."""

    def __init__(self, c_atom: int = 128, c_atompair: int = 16, c_token: int = 768,
                 c_s: int = 384, c_z: int = 128, n_blocks: int = 3, n_heads: int = 4,
                 n_ref_feats: int = 8, has_coords: bool = True):
        super().__init__()
        self.c_atom, self.has_coords = c_atom, has_coords
        self.ref_proj = nn.Linear(n_ref_feats, c_atom)
        self.pair_proj = nn.Linear(3, c_atompair)          # from reference offset vectors
        self.s_to_atom = nn.Linear(c_s, c_atom, bias=False)
        self.z_to_atompair = nn.Linear(c_z, c_atompair, bias=False)
        if has_coords:
            self.pos_proj = nn.Linear(3, c_atom, bias=False)
        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(c_atom, c_atom, c_atompair, n_heads) for _ in range(n_blocks)
        )
        self.to_token = nn.Linear(c_atom, c_token, bias=False)

    def forward(self, ref_feats, ref_pos, atom_to_token, s, z, atom_mask, noisy_pos=None):
        """ref_feats [B,A,F], ref_pos [B,A,3], atom_to_token [B,A] long, s [B,L,c_s],
        z [B,L,L,c_z], atom_mask [B,A]. Returns (a_token [B,L,c_token], q_atom [B,A,c_atom])."""
        B, A, _ = ref_pos.shape
        q = self.ref_proj(ref_feats)
        # Broadcast the token-level conditioning down to the atoms that belong to each token.
        idx = atom_to_token.clamp(min=0)
        q = q + torch.gather(self.s_to_atom(s), 1, idx[..., None].expand(-1, -1, self.c_atom))
        if self.has_coords and noisy_pos is not None:
            q = q + self.pos_proj(noisy_pos)

        p = self.pair_proj(ref_pos[:, :, None, :] - ref_pos[:, None, :, :])
        zt = self.z_to_atompair(z)
        p = p + zt[torch.arange(B, device=z.device)[:, None, None], idx[:, :, None], idx[:, None, :]]

        band = local_attention_mask(A, ref_pos.device)[None].expand(B, -1, -1)
        m = atom_mask.bool()[:, None, :] & band
        for blk in self.blocks:
            q = blk(q, q, p, m.any(dim=1).to(atom_mask.dtype))
        a_atom = self.to_token(q) * atom_mask[..., None]

        # Mean-pool atoms into their token. scatter_add then divide by the count -- a plain scatter
        # would keep only the last atom of each residue.
        L = s.shape[1]
        a_token = torch.zeros(B, L, a_atom.shape[-1], device=q.device, dtype=a_atom.dtype)
        a_token.scatter_add_(1, idx[..., None].expand(-1, -1, a_atom.shape[-1]), a_atom)
        cnt = torch.zeros(B, L, 1, device=q.device, dtype=a_atom.dtype)
        cnt.scatter_add_(1, idx[..., None], atom_mask[..., None].to(a_atom.dtype))
        return a_token / cnt.clamp_min(1.0), q


class AtomAttentionDecoder(nn.Module):
    """Per-token repr -> per-atom coordinate update. SI Alg. 6."""

    def __init__(self, c_atom: int = 128, c_atompair: int = 16, c_token: int = 768,
                 n_blocks: int = 3, n_heads: int = 4):
        super().__init__()
        self.c_atom = c_atom
        self.from_token = nn.Linear(c_token, c_atom, bias=False)
        self.blocks = nn.ModuleList(
            DiffusionTransformerBlock(c_atom, c_atom, c_atompair, n_heads) for _ in range(n_blocks)
        )
        self.norm = nn.LayerNorm(c_atom)
        self.to_pos = nn.Linear(c_atom, 3, bias=False)
        # Zero-init so the module starts as the identity on coordinates and the EDM skip connection
        # carries the first steps, rather than injecting noise before anything is learned.
        nn.init.zeros_(self.to_pos.weight)

    def forward(self, a_token, q_atom, atom_to_token, atom_mask, pair):
        idx = atom_to_token.clamp(min=0)
        q = q_atom + torch.gather(
            self.from_token(a_token), 1, idx[..., None].expand(-1, -1, self.c_atom)
        )
        for blk in self.blocks:
            q = blk(q, q, pair, atom_mask)
        return self.to_pos(self.norm(q)) * atom_mask[..., None]
