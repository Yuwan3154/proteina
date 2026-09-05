"""Contact-map corruption for training the contact-to-coordinate model.

The model must tolerate imperfect contact maps, because at deployment it consumes maps sampled from
the tri model rather than ground truth. Phase 1 approximates that by corrupting GT contacts.

⛔ THE TWO MODES ARE NOT INTERCHANGEABLE, and the difference is roughly 4x in difficulty. ConFind
contact maps are about 2.5% positive, so:

  "balanced" (DEFAULT) -- drop `rate` of the TRUE contacts and raise an equal COUNT of non-contacts.
      The positive rate is preserved exactly, and the corruption is symmetric in the sense that
      matters: as many contacts invented as destroyed.

  "uniform"            -- flip `rate` of ALL pairs. At rate=0.1 that injects ~0.1*L^2 false contacts
      against ~0.025*L^2 true ones, i.e. FOUR TIMES more noise than signal. The resulting map is
      mostly wrong, not slightly wrong.

Both are implemented because the user asked for both; "balanced" is the default by their decision.
"""

from typing import Optional

import torch


def _symmetrise(m: torch.Tensor) -> torch.Tensor:
    """A contact map is symmetric; corrupt the upper triangle and mirror it."""
    upper = torch.triu(m, diagonal=1)
    return upper + upper.transpose(-1, -2)


def augment_contacts(
    contacts: torch.Tensor,
    mask: torch.Tensor,
    rate: float = 0.1,
    mode: str = "balanced",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Corrupt a binary contact map.

    Args:
        contacts: [B, L, L] binary (0/1), symmetric, zero on the diagonal.
        mask:     [B, L] 1 for real residues.
        rate:     fraction to corrupt; see the mode semantics above.
        mode:     "balanced" (default) or "uniform".
    Returns:
        [B, L, L] corrupted map, symmetric, diagonal zero, padded cells zero.
    """
    assert mode in ("balanced", "uniform"), mode
    B, L, _ = contacts.shape
    device = contacts.device
    c = contacts.clone()

    pair = (mask[:, :, None] * mask[:, None, :]).to(torch.bool)
    eye = torch.eye(L, device=device, dtype=torch.bool)[None].expand(B, L, L)
    # Only the strict upper triangle of the valid block is eligible; the mirror is applied after.
    elig = pair & ~eye & torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)[None]

    if mode == "uniform":
        r = torch.rand(c.shape, device=device, generator=generator)
        flip = elig & (r < rate)
        c = torch.where(flip, 1.0 - c, c)
        return _symmetrise(c * elig.to(c.dtype)) * pair.to(c.dtype)

    # balanced: per SAMPLE, drop `rate` of the 1s and raise the SAME COUNT of 0s.
    pos = elig & (c > 0.5)
    neg = elig & (c <= 0.5)
    out = c.clone()
    for b in range(B):
        p_idx = pos[b].nonzero(as_tuple=False)
        n_idx = neg[b].nonzero(as_tuple=False)
        k = int(round(rate * p_idx.shape[0]))
        if k == 0 or n_idx.shape[0] == 0:
            continue
        k = min(k, n_idx.shape[0])
        # randperm rather than rand<rate: it makes the counts EXACT, so the positive rate is
        # preserved per sample instead of only in expectation.
        drop = p_idx[torch.randperm(p_idx.shape[0], device=device, generator=generator)[:k]]
        add = n_idx[torch.randperm(n_idx.shape[0], device=device, generator=generator)[:k]]
        out[b, drop[:, 0], drop[:, 1]] = 0.0
        out[b, add[:, 0], add[:, 1]] = 1.0
    return _symmetrise(out * elig.to(out.dtype)) * pair.to(out.dtype)
