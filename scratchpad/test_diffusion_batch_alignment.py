"""Does row i of the diffusion mini-batch use row i's OWN structure and conditioning?

`forward()` builds the noised targets with
    x_gt[:, None].expand(B, n, A, 3).reshape(B*n, A, 3)
and the conditioning with
    t.repeat_interleave(n, dim=0)
Those are two DIFFERENT replication ops. If their row orders disagree (interleave vs tile), the
model is trained to denoise structure A's coordinates while conditioned on structure B's contact
map and sequence. The loss would still fall -- the model would learn population statistics -- but it
could never converge, which is indistinguishable from the failure we are chasing.

Reasoning says they agree (both map row b*n+j -> b). Reasoning has been wrong twice today, so this
checks it on tensors with distinguishable values, and checks the mask/gt pairing end to end.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def main():
    B, n, A, L = 3, 4, 8, 2

    # Row-order equivalence of the two replication ops, on values that identify their source row.
    x_gt = torch.arange(B, dtype=torch.float32)[:, None, None].expand(B, A, 3).contiguous()
    tag = torch.arange(B, dtype=torch.float32)[:, None].expand(B, L).contiguous()

    x_rep = x_gt[:, None].expand(B, n, A, 3).reshape(B * n, A, 3)   # as in forward()
    t_rep = tag.repeat_interleave(n, dim=0)                          # as in forward()

    src_from_x = x_rep[:, 0, 0]
    src_from_t = t_rep[:, 0]
    print(f"  row source id from x_gt_rep : {[int(v) for v in src_from_x]}")
    print(f"  row source id from rep(cond): {[int(v) for v in src_from_t]}")
    check("expand+reshape and repeat_interleave agree row for row",
          torch.equal(src_from_x, src_from_t))
    check("row b*n+j really comes from structure b",
          all(int(src_from_x[b * n + j]) == b for b in range(B) for j in range(n)))

    # ⛔ The failure mode this guards against, made explicit: a TILE ordering would also produce
    # B*n rows and pass every shape assertion, while pairing the wrong structure with the wrong
    # conditioning on all but the first block.
    tile = tag.repeat(n, 1)[:, 0]
    check("a TILE ordering would indeed be wrong (and is not what we use)",
          not torch.equal(tile, src_from_x),
          f"tile={[int(v) for v in tile]}")

    # End to end through the real model: does the loss see matching gt and mask per row?
    from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
    from proteinfoundation.nn.contact2coord import ContactToCoord

    torch.manual_seed(0)
    Lr = 6
    m = ContactToCoord(c_s=32, c_z=16, c_token=32, c_atom=16, c_atompair=8, n_blocks=1,
                       n_heads=2, n_tri_blocks=1, tri_hidden=16, transition_n=1,
                       atom_blocks=1, atom_heads=2, n_ref_feats=N_REF_FEATS,
                       n_diffusion_samples=n)
    aatype = torch.randint(0, 20, (B, Lr))
    mask = torch.ones(B, Lr)
    mask[2, -2:] = 0.0                       # a distinguishable per-structure mask
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aatype, mask)
    # ⛔ A per-structure CONSTANT (the old fixture) is degenerate now that the targets are centred:
    # a structure whose atoms all share one point centres to all-zeros, destroying the identity the
    # test keys on. Use a distinct rigid SHAPE per structure instead, identified by its internal
    # distance matrix, which is exactly the quantity a rigid augmentation must preserve.
    base = torch.randn(1, Lr * 14, 3, generator=torch.Generator().manual_seed(3)) * 8.0
    scales = torch.tensor([1.0, 2.0, 3.0])[:B, None, None]
    atom_pos = (base * scales).contiguous() + torch.tensor([40.0, -25.0, 15.0])
    batch = {"contacts": torch.zeros(B, Lr, Lr), "aatype": aatype, "mask": mask,
             "ref_feats": ref_feats, "ref_pos": ref_pos, "atom_to_token": a2t,
             "atom_mask": amask, "ref_space_uid": ruid, "atom_pos": atom_pos}
    out = m(batch)

    check("x_gt_rep has B*n rows", out["x_gt_rep"].shape[0] == B * n,
          str(tuple(out["x_gt_rep"].shape)))
    # ⛔ x_gt_rep is now the AUGMENTED target (centred + independently rotated/translated per
    # replica), which it MUST be so the loss target matches the noised input. So the invariant is no
    # longer "carries structure b's coordinates" but "is a RIGID TRANSFORM of structure b" --
    # checked through the internal distance matrix, which a rigid motion preserves exactly.
    def pdist(p):
        d = p[:, None, :].double() - p[None, :, :].double()
        return (d * d).sum(-1).sqrt()

    ok_src, detail = True, []
    for b in range(B):
        keep = amask[b].bool()
        d_ref = pdist(atom_pos[b][keep])
        for j in range(n):
            d_row = pdist(out["x_gt_rep"][b * n + j][keep])
            err = (d_row - d_ref).abs().max().item()
            ok_src &= err < 1e-3
            if j == 0:
                detail.append(f"b{b}:{err:.1e}")
    check("x_gt_rep row b*n+j is a RIGID transform of structure b", ok_src, " ".join(detail))

    # And the replicas must genuinely differ, or the per-sample augmentation is not happening.
    spread = max((out["x_gt_rep"][b * n] - out["x_gt_rep"][b * n + 1]).abs().max().item()
                 for b in range(B))
    check("replicas of one structure got DIFFERENT augmentations", spread > 1.0,
          f"max |diff| {spread:.2f} A")

    # Centring must have happened: the augmented COM sits near the origin, not at the raw offset.
    com = (out["x_gt_rep"] * out["atom_mask_rep"][..., None]).sum(1) / \
        out["atom_mask_rep"].sum(1, keepdim=True).clamp_min(1e-8)
    check("augmented targets are CENTRED (raw fixture sat ~50 A off-origin)",
          com.norm(dim=-1).max().item() < 6.0,
          f"max |COM| {com.norm(dim=-1).max().item():.2f} A")

    # The masks must be replicated in the SAME order, or the loss averages over the wrong atoms.
    mask_counts = out["atom_mask_rep"].sum(-1)
    per_struct = amask.sum(-1)
    check("atom_mask_rep row b*n+j carries structure b's mask",
          all(float(mask_counts[b * n + j]) == float(per_struct[b])
              for b in range(B) for j in range(n)),
          f"per-structure atom counts {[float(v) for v in per_struct]}")

    check("sigma is per row, not per structure", out["sigma"].shape == (B * n,),
          str(tuple(out["sigma"].shape)))
    check("sigma values differ within a structure (independent noise draws)",
          len({round(float(v), 6) for v in out["sigma"][:n]}) > 1)

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
