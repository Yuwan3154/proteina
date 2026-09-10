"""Gate for the checkpointed diffusion-sample chunking in ContactToCoord (diff_chunk).

The chunked path must be a pure memory optimisation: with the SAME noise draws, x_denoised and the
gradients of the diffusion loss must match the single-call path to floating-point tolerance, for a
chunk size that divides n and one that does not, and diff_chunk=0 must be the old code path.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.datasets.atom_features import N_REF_FEATS, atom14_features
from proteinfoundation.nn.af3_diffusion import diffusion_loss
from proteinfoundation.nn.contact2coord import ContactToCoord

PASS, FAIL = [], []
B, L, N = 2, 12, 6
CFG = dict(c_s=32, c_z=16, c_token=32, c_atom=16, c_atompair=8, n_blocks=2, n_heads=4,
           n_tri_blocks=2, tri_hidden=16, transition_n=2, atom_blocks=1, atom_heads=2,
           c_noise_embedding=16, n_ref_feats=N_REF_FEATS, n_diffusion_samples=N)


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def make_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(B, L)
    mask[1, L - 3:] = 0.0
    aatype = torch.randint(0, 20, (B, L), generator=g)
    c = (torch.rand(B, L, L, generator=g) < 0.1).float()
    c = torch.triu(c, 1)
    c = (c + c.transpose(1, 2)) * (mask[:, :, None] * mask[:, None, :])
    ref_feats, ref_pos, a2t, amask, ruid = atom14_features(aatype, mask)
    return {
        "contacts": c, "aatype": aatype, "mask": mask,
        "ref_feats": ref_feats, "ref_pos": ref_pos,
        "atom_to_token": a2t, "atom_mask": amask, "ref_space_uid": ruid,
        "atom_pos": torch.randn(B, L * 14, 3, generator=g) * 5.0 * amask[..., None],
    }


def run(chunk, seed=0):
    """forward + backward with identical model init and identical noise draws."""
    torch.manual_seed(0)
    m = ContactToCoord(**CFG, diff_chunk=chunk).train()
    batch = make_batch()
    torch.manual_seed(seed)                      # noise levels, augmentation, noise: same stream
    out = m(batch)
    dl, _ = diffusion_loss(out["x_denoised"], out["x_gt_rep"], out["sigma"], out["atom_mask_rep"],
                           use_smooth_lddt=False)
    loss = dl.mean()
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    return out, float(loss), grads


def main():
    ref_out, ref_loss, ref_grads = run(0)
    check("diff_chunk=0: x_denoised has B*n rows", tuple(ref_out["x_denoised"].shape)[0] == B * N)
    for chunk in (3, 4, 1):
        out, loss, grads = run(chunk)
        dx = float((out["x_denoised"] - ref_out["x_denoised"]).abs().max())
        check(f"chunk {chunk}: x_denoised matches the single call", dx < 1e-5, f"max|dx|={dx:.2e}")
        check(f"chunk {chunk}: loss matches", abs(loss - ref_loss) < 1e-6, f"{loss:.6f} vs {ref_loss:.6f}")
        # backward through the chunks sums the same terms in a different order (fp32 noise, ~1e-4
        # absolute on gradients of order 1), so the bar is RELATIVE to the gradient scale
        dg = float((grads - ref_grads).abs().max())
        scale = max(1.0, float(ref_grads.abs().max()))
        check(f"chunk {chunk}: gradients match", dg < 1e-3 * scale, f"max|dgrad|={dg:.2e} (scale {scale:.2e})")
        check(f"chunk {chunk}: atom_mask_rep / sigma unchanged",
              torch.equal(out["atom_mask_rep"], ref_out["atom_mask_rep"]) and torch.equal(out["sigma"], ref_out["sigma"]))
    m = ContactToCoord(**CFG, diff_chunk=4)
    check("diff_chunk adds no parameters", sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in ContactToCoord(**CFG).parameters()))
    with torch.no_grad():
        m.eval()
        o = m(make_batch())
        check("eval (no grad) path runs chunked without checkpoint", tuple(o["x_denoised"].shape)[0] == B * N)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
