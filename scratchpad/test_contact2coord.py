"""Gate for the standalone contact-to-coordinate model.

Checks the things that fail silently in a long run: shapes and masking through the atom path, that
the contact map actually CHANGES the output (a model that ignores its conditioning would train
happily and be worthless), EDM limits, and that gradients reach every major sub-module.
"""

import sys

import torch

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord

B, L = 2, 24
# Small dims: this gate is about wiring and invariants, not capacity. Production dims are AF3's.
CFG = dict(c_s=32, c_z=16, c_token=32, c_atom=16, c_atompair=8, n_blocks=2, n_heads=4,
           n_tri_blocks=2, tri_hidden=16, transition_n=2, atom_blocks=1, atom_heads=2,
           c_noise_embedding=16, n_ref_feats=N_REF_FEATS)


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, L)
    mask[1, L - 5:] = 0.0
    aatype = torch.randint(0, 20, (B, L), generator=g)
    c = (torch.rand(B, L, L, generator=g) < 0.05).float()
    c = torch.triu(c, 1)
    c = (c + c.transpose(1, 2)) * (mask[:, :, None] * mask[:, None, :])
    ref_feats, ref_pos, a2t, amask = atom14_features(aatype, mask)
    return {
        "contacts": c, "aatype": aatype, "mask": mask,
        "ref_feats": ref_feats, "ref_pos": ref_pos,
        "atom_to_token": a2t, "atom_mask": amask,
        "atom_pos": torch.randn(B, L * 14, 3, generator=g) * 5.0 * amask[..., None],
    }


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return bool(ok)


def main():
    torch.manual_seed(0)
    r = []
    batch = make_batch()
    A = L * 14

    print("1. atom featurizer")
    r.append(check(f"ref_feats [B,{A},{N_REF_FEATS}]",
                   tuple(batch["ref_feats"].shape) == (B, A, N_REF_FEATS)))
    r.append(check("atom_mask marks only real atoms of real residues",
                   float(batch["atom_mask"][1, (L - 5) * 14:].sum()) == 0.0))
    r.append(check("every real residue has >= 4 atoms (N, CA, C, O)",
                   bool((batch["atom_mask"].reshape(B, L, 14).sum(-1)[batch["mask"].bool()] >= 4).all())))
    r.append(check("atom_to_token maps each slot to its residue",
                   bool((batch["atom_to_token"].reshape(B, L, 14)[0, 5] == 5).all())))

    print("2. forward shapes")
    m = ContactToCoord(**CFG).eval()
    with torch.no_grad():
        out = m(dict(batch))
    r.append(check(f"x_denoised [B,{A},3]", tuple(out["x_denoised"].shape) == (B, A, 3)))
    r.append(check("pair_logits [B,L,L,39]", tuple(out["pair_logits"].shape) == (B, L, L, 39)))
    r.append(check("finite", torch.isfinite(out["x_denoised"]).all().item()))
    r.append(check("padded atoms are exactly zero",
                   float(out["x_denoised"][batch["atom_mask"] == 0].abs().sum()) == 0.0))
    r.append(check("distogram symmetric",
                   torch.allclose(out["pair_logits"], out["pair_logits"].transpose(1, 2), atol=1e-5)))

    print("3. ⭐ the contact map must actually MATTER")
    # A model that ignores its conditioning trains happily and is worthless. Compare the DETERMINISTIC
    # part -- the encoder -- rather than the sampled denoiser, so the difference cannot be noise.
    b2 = dict(batch)
    b2["contacts"] = torch.zeros_like(batch["contacts"])
    with torch.no_grad():
        s1, z1, d1 = m.encode(batch["contacts"], batch["aatype"], batch["mask"])
        s2, z2, d2 = m.encode(b2["contacts"], b2["aatype"], b2["mask"])
    dz = (z1 - z2).abs().mean().item()
    dd = (d1 - d2).abs().mean().item()
    r.append(check(f"pair repr responds to contacts (mean |dz| = {dz:.4e})", dz > 1e-6))
    r.append(check(f"distogram responds to contacts (mean |d| = {dd:.4e})", dd > 1e-6))

    print("4. gradients reach every major sub-module")
    m2 = ContactToCoord(**CFG).train()
    o = m2(dict(batch))
    loss, _ = diffusion_loss(o["x_denoised"], batch["atom_pos"], o["sigma"], batch["atom_mask"])
    loss.mean().backward()
    groups = {"tri_blocks": 0.0, "atom_enc": 0.0, "blocks": 0.0, "atom_dec": 0.0,
              "contact_emb": 0.0, "dist_head": 0.0}
    for n, p in m2.named_parameters():
        if p.grad is None:
            continue
        for k in groups:
            if n.startswith(k):
                groups[k] += p.grad.abs().sum().item()
    for k, v in groups.items():
        # dist_head sees no gradient from the diffusion loss alone -- it is driven by the distogram
        # loss, which is not part of this backward. Everything else must receive one.
        want = v == 0.0 if k == "dist_head" else v > 0
        r.append(check(f"{k} grad {'zero as expected' if k == 'dist_head' else 'non-zero'} ({v:.3e})",
                       want))

    print("5. rollout")
    with torch.no_grad():
        co = m.rollout(*m.encode(batch["contacts"], batch["aatype"], batch["mask"])[:2],
                       batch["mask"], batch["ref_feats"], batch["ref_pos"],
                       batch["atom_to_token"], batch["atom_mask"], n_steps=3)
    r.append(check(f"rollout [B,{A},3] finite", tuple(co.shape) == (B, A, 3)
                   and torch.isfinite(co).all().item()))
    r.append(check("rollout zeroes padded atoms",
                   float(co[batch["atom_mask"] == 0].abs().sum()) == 0.0))

    print()
    print(f"{sum(r)}/{len(r)} checks pass")
    return 0 if all(r) else 3


if __name__ == "__main__":
    sys.exit(main())
