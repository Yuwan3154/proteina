"""Per-atom reference features for the AF3 atom-attention encoder, standard amino acids only.

AF3 builds these from CCD reference conformers because it must represent arbitrary ligands. We
handle 20 canonical residues, for which the repo's own `residue_constants` already carries
everything needed -- the atom14 layout, its validity mask, and idealised rigid-group positions. So
this needs no new data and no CCD dependency.

⛔ atom14, not atom37. atom14 is the DENSE per-residue layout (14 slots, all used by at least one
residue type), whereas atom37 is the sparse union over all residues and is ~62% padding for a
typical protein. For atom attention the difference is direct: cost scales with the atom count, and
atom37 would trade ~2.6x more atoms for no additional information.
"""

import torch

from proteinfoundation.openfold_stub.np import residue_constants as rc

# Element identity per atom14 slot, derived from the atom NAME (its first character is the element
# for every atom in the standard residues -- N, CA, C, O, CB, CG... -- with no two-letter elements).
_ELEMENTS = ["C", "N", "O", "S"]
N_REF_FEATS = 4 + 3 + 1   # element one-hot (4) + idealised position (3) + slot index (1)


def _build_tables():
    """[21, 14] element index, [21, 14] mask, [21, 14, 3] idealised positions."""
    elem = torch.zeros(21, 14, dtype=torch.long)
    for r, resname3 in enumerate(rc.restypes + ["UNK"]):
        name3 = rc.restype_1to3.get(resname3, "UNK") if len(resname3) == 1 else resname3
        names = rc.restype_name_to_atom14_names.get(name3, [""] * 14)
        for a, atom_name in enumerate(names):
            if not atom_name:
                continue
            e = atom_name[0]
            elem[r, a] = _ELEMENTS.index(e) if e in _ELEMENTS else 0
    mask = torch.as_tensor(rc.restype_atom14_mask, dtype=torch.float32)
    pos = torch.as_tensor(rc.restype_atom14_rigid_group_positions, dtype=torch.float32)
    return elem, mask, pos


_ELEM_TABLE, _MASK_TABLE, _POS_TABLE = _build_tables()


def atom14_features(aatype: torch.Tensor, mask: torch.Tensor):
    """Flatten a [B, L] residue sequence into per-atom reference features.

    Returns:
        ref_feats     [B, L*14, N_REF_FEATS]
        ref_pos       [B, L*14, 3]   idealised local-frame positions
        atom_to_token [B, L*14]      which residue each atom belongs to
        atom_mask     [B, L*14]      1 where the slot is a real atom of a real residue
    """
    B, L = aatype.shape
    dev = aatype.device
    a = aatype.long().clamp(0, 20)

    elem = _ELEM_TABLE.to(dev)[a]                       # [B, L, 14]
    amask = _MASK_TABLE.to(dev)[a] * mask[..., None]    # [B, L, 14]
    apos = _POS_TABLE.to(dev)[a]                        # [B, L, 14, 3]

    slot = torch.arange(14, device=dev, dtype=torch.float32)[None, None, :].expand(B, L, 14)
    feats = torch.cat(
        [
            torch.nn.functional.one_hot(elem, len(_ELEMENTS)).float(),
            apos,
            (slot / 13.0)[..., None],
        ],
        dim=-1,
    )
    tok = torch.arange(L, device=dev)[None, :, None].expand(B, L, 14)
    return (
        feats.reshape(B, L * 14, -1) * amask.reshape(B, L * 14, 1),
        apos.reshape(B, L * 14, 3) * amask.reshape(B, L * 14, 1),
        tok.reshape(B, L * 14),
        amask.reshape(B, L * 14),
    )
